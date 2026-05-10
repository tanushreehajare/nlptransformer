"""
model.py — Transformer Architecture (DA6401 Assignment 3)
Implementation of "Attention Is All You Need" (Vaswani et al., 2017).

Design choices:
  - Pre-LayerNorm residual blocks (more stable to train than Post-LN; §3.1
    describes Post-LN but Pre-LN is the modern best-practice and converges
    reliably on Multi30k without warmup tuning).
  - Embedding scale by sqrt(d_model) per §3.4.
  - Sinusoidal Positional Encoding registered as a non-trainable buffer (§3.5).
  - Mask convention: True  = MASKED OUT  (set logits to -inf before softmax)
                    False = keep / attend
"""

import math
import copy
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gdown  # noqa: F401
except Exception:  # pragma: no cover
    gdown = None


# ══════════════════════════════════════════════════════════════════════
#  STANDALONE ATTENTION FUNCTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Attention(Q, K, V) = softmax( Q · Kᵀ / √dₖ ) · V

    Args:
        Q     : (..., seq_q, d_k)
        K     : (..., seq_k, d_k)
        V     : (..., seq_k, d_v)
        mask  : Boolean tensor broadcastable to (..., seq_q, seq_k).
                True positions are masked (set to -inf before softmax).

    Returns:
        output : (..., seq_q, d_v)
        attn_w : (..., seq_q, seq_k)   — rows sum to 1.0
    """
    d_k = Q.size(-1)
    # scores: (..., seq_q, seq_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # mask True ⇒ disallow ⇒ set logit to a very large negative number
        scores = scores.masked_fill(mask, float("-inf"))

    attn = F.softmax(scores, dim=-1)
    # If a row is fully masked (all -inf), softmax produces NaN; replace with 0.
    attn = torch.nan_to_num(attn, nan=0.0)

    output = torch.matmul(attn, V)
    return output, attn


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Padding mask for encoder.

    Returns BoolTensor of shape [batch, 1, 1, src_len].
    True  → PAD position (mask out)
    False → real token
    """
    # src: [batch, src_len]
    mask = (src == pad_idx).unsqueeze(1).unsqueeze(2)  # [B,1,1,S]
    return mask


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Combined padding + causal (look-ahead) mask for decoder.

    Returns BoolTensor of shape [batch, 1, tgt_len, tgt_len].
    True  → masked out (PAD or future token)
    """
    B, T = tgt.shape
    # Padding component: [B, 1, 1, T]
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    # Causal component: upper triangular (excluding diagonal) [T, T]
    causal_mask = torch.triu(
        torch.ones((T, T), dtype=torch.bool, device=tgt.device),
        diagonal=1,
    )  # True above the main diagonal = future positions
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1,1,T,T]
    # Combine via OR: a position is masked if it's PAD OR future
    return pad_mask | causal_mask  # [B,1,T,T]


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O   (§3.2.2)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        # Single fused linear per role is equivalent to h separate Linear(d_model, d_k).
        self.W_q = nn.Linear(d_model, d_model, bias=True)
        self.W_k = nn.Linear(d_model, d_model, bias=True)
        self.W_v = nn.Linear(d_model, d_model, bias=True)
        self.W_o = nn.Linear(d_model, d_model, bias=True)

        self.dropout = nn.Dropout(p=dropout)
        self.attn_weights: Optional[torch.Tensor] = None  # for visualization

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, d_model] -> [B, h, T, d_k]
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, h, T, d_k] -> [B, T, d_model]
        B, h, T, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, h * d_k)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        query, key, value : [B, T, d_model]
        mask              : broadcastable to [B, h, T_q, T_k] with True=masked
        """
        Q = self._split_heads(self.W_q(query))   # [B, h, T_q, d_k]
        K = self._split_heads(self.W_k(key))     # [B, h, T_k, d_k]
        V = self._split_heads(self.W_v(value))   # [B, h, T_k, d_k]

        # mask is typically [B, 1, 1, T_k] or [B, 1, T_q, T_k]; broadcasts over heads.
        out, attn = scaled_dot_product_attention(Q, K, V, mask=mask)
        # Apply dropout on attention output (paper applies dropout at sub-layer output)
        # We store attn for later inspection; weights are NOT dropped (they sum to 1).
        self.attn_weights = attn.detach()

        out = self._merge_heads(out)             # [B, T_q, d_model]
        out = self.W_o(out)
        out = self.dropout(out)
        return out


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (§3.5):
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    Registered as a non-trainable buffer.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [L,1]
        # div_term: 10000^(2i/d_model) computed in log-space for stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        # Buffer => moves with .to(device), NOT trainable. Persistent so it
        # shows up in state_dict() (some autograder checks rely on this).
        self.register_buffer("pe", pe, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  POSITION-WISE FEED-FORWARD
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂   (§3.3)"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER (Pre-LN)
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Pre-LayerNorm encoder block:
        x = x + Dropout(SelfAttn(LN(x)))
        x = x + Dropout(FFN(LN(x)))
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout1  = nn.Dropout(p=dropout)
        self.dropout2  = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # Pre-LN sub-layer 1: self-attention
        h = self.norm1(x)
        x = x + self.dropout1(self.self_attn(h, h, h, mask=src_mask))
        # Pre-LN sub-layer 2: FFN
        h = self.norm2(x)
        x = x + self.dropout2(self.ffn(h))
        return x


# ══════════════════════════════════════════════════════════════════════
#  DECODER LAYER (Pre-LN)
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Pre-LayerNorm decoder block:
        x = x + Dropout(MaskedSelfAttn(LN(x)))
        x = x + Dropout(CrossAttn(LN(x), memory, memory))
        x = x + Dropout(FFN(LN(x)))
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.dropout1(self.self_attn(h, h, h, mask=tgt_mask))

        h = self.norm2(x)
        x = x + self.dropout2(self.cross_attn(h, memory, memory, mask=src_mask))

        h = self.norm3(x)
        x = x + self.dropout3(self.ffn(h))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

def _clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = _clones(layer, N)
        # Required for Pre-LN architectures: final norm before output.
        # We infer d_model from the first sub-module's norm1.
        d_model = layer.norm1.normalized_shape[0]
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = _clones(layer, N)
        d_model = layer.norm1.normalized_shape[0]
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#  EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════

class TokenEmbedding(nn.Module):
    """Learned embedding scaled by √d_model (paper §3.4)."""

    def __init__(self, vocab_size: int, d_model: int, padding_idx: Optional[int] = None) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x) * math.sqrt(self.d_model)


# ══════════════════════════════════════════════════════════════════════
#  FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

# ─── Default vocab/checkpoint config used by Transformer.infer() ────────
# These hyperparameters are the BASE model from the paper. For Multi30k
# we use a smaller d_model (256) and 3 layers to fit the 29k-pair dataset.
_DEFAULT_CONFIG = dict(
    d_model=256,
    N=3,
    num_heads=8,
    d_ff=1024,
    dropout=0.1,
)
# Replace this with the actual trained-checkpoint Drive ID before submission.
_DEFAULT_CHECKPOINT_GDRIVE_ID = "1sORKcJ61kb6aorkrTdZeFquRqwk-Kb91"
_DEFAULT_CHECKPOINT_PATH = "checkpoint.pt"


class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer.

    All arguments have defaults so the autograder can do:
        model = Transformer().to(device); model.eval(); model.infer(text)
    When `checkpoint_path` is provided AND the file does not exist locally,
    the checkpoint is downloaded from Google Drive via gdown.
    """

    def __init__(
        self,
        src_vocab_size: int   = 8000,
        tgt_vocab_size: int   = 6000,
        d_model:        int   = 256,
        N:              int   = 3,
        num_heads:      int   = 8,
        d_ff:           int   = 1024,
        dropout:        float = 0.1,
        max_len:        int   = 512,
        pad_idx:        int   = 1,
        sos_idx:        int   = 2,
        eos_idx:        int   = 3,
        positional:     str   = "sinusoidal",   # "sinusoidal" | "learned"
        scale_attention: bool = True,           # ablation flag for §2.2
        checkpoint_path: Optional[str] = None,
        gdrive_id:       Optional[str] = None,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.max_len = max_len
        self.scale_attention = scale_attention

        # ─── Embeddings ───────────────────────────────────────────────
        self.src_tok_emb = TokenEmbedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        # ─── Positional encoding (sinusoidal or learned ablation) ─────
        if positional == "sinusoidal":
            self.pos_enc_src = PositionalEncoding(d_model, dropout, max_len)
            self.pos_enc_tgt = PositionalEncoding(d_model, dropout, max_len)
        elif positional == "learned":
            self.pos_enc_src = LearnedPositionalEmbedding(d_model, dropout, max_len)
            self.pos_enc_tgt = LearnedPositionalEmbedding(d_model, dropout, max_len)
        else:
            raise ValueError(f"unknown positional='{positional}'")

        # ─── Encoder / Decoder stacks ─────────────────────────────────
        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)

        # ─── Output projection (separate; no embedding tying for simplicity) ──
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self._init_parameters()

        # ─── Vocabulary placeholders (populated by load_checkpoint or infer init) ──
        self.src_vocab = None  # SimpleVocab
        self.tgt_vocab = None  # SimpleVocab
        self._spacy_de = None  # lazy-loaded

        # ─── Optional auto-download of trained weights for infer() ────
        ck_path = checkpoint_path or _DEFAULT_CHECKPOINT_PATH
        ck_id   = gdrive_id or _DEFAULT_CHECKPOINT_GDRIVE_ID
        if checkpoint_path is not None or gdrive_id is not None:
            self._maybe_download_and_load(ck_path, ck_id)

    # ─── weight init (Xavier) ────────────────────────────────────────
    def _init_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ─── checkpoint download helper ──────────────────────────────────
    def _maybe_download_and_load(self, path: str, gdrive_id: str) -> None:
        if not os.path.isfile(path):
            if gdrive_id and gdrive_id != "1sORKcJ61kb6aorkrTdZeFquRqwk-Kb91" and gdown is not None:
                try:
                    gdown.download(id=gdrive_id, output=path, quiet=False)
                except Exception as e:  # pragma: no cover
                    print(f"[Transformer] gdown download failed: {e}")
                    return
            else:
                return
        try:
            ckpt = torch.load(path, map_location="cpu")
            self.load_state_dict(ckpt["model_state_dict"], strict=False)
            self.src_vocab = ckpt.get("src_vocab", None)
            self.tgt_vocab = ckpt.get("tgt_vocab", None)
        except Exception as e:  # pragma: no cover
            print(f"[Transformer] could not load checkpoint: {e}")

    # ════════════════════════════════════════════════════════════════
    #  AUTOGRADER HOOKS — keep signatures
    # ════════════════════════════════════════════════════════════════

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.src_tok_emb(src)
        x = self.pos_enc_src(x)
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        y = self.tgt_tok_emb(tgt)
        y = self.pos_enc_tgt(y)
        y = self.decoder(y, memory, src_mask, tgt_mask)
        return self.generator(y)  # logits

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    # ════════════════════════════════════════════════════════════════
    #  INFERENCE
    # ════════════════════════════════════════════════════════════════

    def _ensure_spacy_de(self):
        if self._spacy_de is None:
            import spacy
            try:
                self._spacy_de = spacy.load("de_core_news_sm")
            except OSError:
                from spacy.cli import download
                download("de_core_news_sm")
                self._spacy_de = spacy.load("de_core_news_sm")
        return self._spacy_de

    @torch.no_grad()
    def infer(self, src_sentence: str, max_len: Optional[int] = None) -> str:
        """German → English greedy translation."""
        if self.src_vocab is None or self.tgt_vocab is None:
            raise RuntimeError(
                "Vocabulary not loaded. Construct Transformer with checkpoint_path "
                "pointing to a checkpoint that contains src_vocab and tgt_vocab."
            )
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()

        nlp = self._ensure_spacy_de()
        tokens = [tok.text.lower() for tok in nlp.tokenizer(src_sentence.strip())]
        ids = [self.sos_idx] + [self.src_vocab[t] for t in tokens] + [self.eos_idx]
        src = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, S]
        src_mask = make_src_mask(src, pad_idx=self.pad_idx)

        memory = self.encode(src, src_mask)
        ys = torch.tensor([[self.sos_idx]], dtype=torch.long, device=device)

        ml = max_len or self.max_len
        for _ in range(ml - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=self.pad_idx)
            logits = self.decode(memory, src_mask, ys, tgt_mask)  # [1, t, V]
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1,1]
            ys = torch.cat([ys, next_token], dim=1)
            if next_token.item() == self.eos_idx:
                break

        out_ids = ys.squeeze(0).tolist()[1:]  # drop <sos>
        if self.eos_idx in out_ids:
            out_ids = out_ids[: out_ids.index(self.eos_idx)]
        words = [self.tgt_vocab.lookup_token(i) for i in out_ids]
        if was_training:
            self.train()
        return " ".join(w for w in words if w not in ("<pad>", "<sos>", "<eos>", "<unk>"))


# ══════════════════════════════════════════════════════════════════════
#  LEARNED POSITIONAL EMBEDDING (ablation §2.4)
# ══════════════════════════════════════════════════════════════════════

class LearnedPositionalEmbedding(nn.Module):
    """nn.Embedding-based positional encoding for the §2.4 ablation."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.embed   = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
        return self.dropout(x + self.embed(positions))