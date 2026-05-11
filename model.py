"""
model.py — Transformer Architecture (DA6401 Assignment 3)
Implementation of "Attention Is All You Need" (Vaswani et al., 2017).

Design choices:
  - Pre-LayerNorm residual blocks (more stable to train than Post-LN).
  - Embedding scale by sqrt(d_model) per §3.4.
  - Sinusoidal Positional Encoding registered as a non-trainable buffer (§3.5).
  - Mask convention: True  = MASKED OUT  (set logits to -inf before softmax)
                    False = keep / attend
  - infer() uses beam search (beam=4, length_penalty=0.6) — no spaCy needed.
"""

import math
import copy
import os
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gdown
except Exception:
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
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    output = torch.matmul(attn, V)
    return output, attn


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Padding mask for encoder.
    Returns BoolTensor shape [batch, 1, 1, src_len].
    True  → PAD (mask out),  False → real token.
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Combined padding + causal mask for decoder.
    Returns BoolTensor shape [batch, 1, tgt_len, tgt_len].
    True → masked out (PAD or future token).
    """
    B, T = tgt.shape
    pad_mask    = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.triu(
        torch.ones((T, T), dtype=torch.bool, device=tgt.device), diagonal=1
    ).unsqueeze(0).unsqueeze(0)
    return pad_mask | causal_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O   (§3.2.2)"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=True)
        self.W_k = nn.Linear(d_model, d_model, bias=True)
        self.W_v = nn.Linear(d_model, d_model, bias=True)
        self.W_o = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(p=dropout)
        self.attn_weights: Optional[torch.Tensor] = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, h, T, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, h * d_k)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))
        out, attn = scaled_dot_product_attention(Q, K, V, mask=mask)
        self.attn_weights = attn.detach()
        out = self._merge_heads(out)
        out = self.W_o(out)
        out = self.dropout(out)
        return out


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (§3.5).
    Registered as a NON-TRAINABLE persistent buffer.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
#  ENCODER LAYER  (Pre-LayerNorm)
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout1  = nn.Dropout(p=dropout)
        self.dropout2  = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.dropout1(self.self_attn(h, h, h, mask=src_mask))
        h = self.norm2(x)
        x = x + self.dropout2(self.ffn(h))
        return x


# ══════════════════════════════════════════════════════════════════════
#  DECODER LAYER  (Pre-LayerNorm)
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout1   = nn.Dropout(p=dropout)
        self.dropout2   = nn.Dropout(p=dropout)
        self.dropout3   = nn.Dropout(p=dropout)

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
    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = _clones(layer, N)
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = _clones(layer, N)
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape[0])

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
    """Learned embedding scaled by sqrt(d_model) (paper §3.4)."""

    def __init__(self, vocab_size: int, d_model: int, padding_idx: Optional[int] = None) -> None:
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x) * math.sqrt(self.d_model)


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embedding for §2.4 ablation."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.embed   = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        return self.dropout(x + self.embed(positions))


# ══════════════════════════════════════════════════════════════════════
#  DEFAULT CHECKPOINT CONFIG
#  Update _DEFAULT_CHECKPOINT_GDRIVE_ID with your new trained checkpoint.
# ══════════════════════════════════════════════════════════════════════

_DEFAULT_CHECKPOINT_GDRIVE_ID = "1sORKcJ61kb6aorkrTdZeFquRqwk-Kb91"
_DEFAULT_CHECKPOINT_PATH      = "checkpoint.pt"


# ══════════════════════════════════════════════════════════════════════
#  FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for DE→EN translation.

    Transformer() with NO arguments:
      1. Builds the model with default hyperparams
      2. Downloads checkpoint from Google Drive
      3. Loads weights + src_vocab + tgt_vocab
      4. infer(german_sentence) works immediately with beam search
    """

    def __init__(
        self,
        src_vocab_size:  int   = 7853,
        tgt_vocab_size:  int   = 5893,
        d_model:         int   = 256,
        N:               int   = 3,
        num_heads:       int   = 8,
        d_ff:            int   = 1024,
        dropout:         float = 0.1,
        max_len:         int   = 512,
        pad_idx:         int   = 1,
        sos_idx:         int   = 2,
        eos_idx:         int   = 3,
        positional:      str   = "sinusoidal",
        scale_attention: bool  = True,
        checkpoint_path: Optional[str] = None,
        gdrive_id:       Optional[str] = None,
    ) -> None:
        super().__init__()

        self.d_model         = d_model
        self.pad_idx         = pad_idx
        self.sos_idx         = sos_idx
        self.eos_idx         = eos_idx
        self.max_len         = max_len
        self.scale_attention = scale_attention

        self.src_tok_emb = TokenEmbedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        if positional == "sinusoidal":
            self.pos_enc_src = PositionalEncoding(d_model, dropout, max_len)
            self.pos_enc_tgt = PositionalEncoding(d_model, dropout, max_len)
        elif positional == "learned":
            self.pos_enc_src = LearnedPositionalEmbedding(d_model, dropout, max_len)
            self.pos_enc_tgt = LearnedPositionalEmbedding(d_model, dropout, max_len)
        else:
            raise ValueError(f"unknown positional='{positional}'")

        enc_layer     = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer     = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder  = Encoder(enc_layer, N)
        self.decoder  = Decoder(dec_layer, N)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self._init_parameters()

        self.src_vocab = None
        self.tgt_vocab = None

        # ALWAYS download + load checkpoint so Transformer() with no args works
        ck_path = checkpoint_path if checkpoint_path is not None else _DEFAULT_CHECKPOINT_PATH
        ck_id   = gdrive_id       if gdrive_id       is not None else _DEFAULT_CHECKPOINT_GDRIVE_ID
        self._download_and_load(ck_path, ck_id)

    def _init_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _download_and_load(self, path: str, gdrive_id: str) -> None:
        """Download from Drive if needed, then load weights + vocab."""
        # Sentinel used by run_training_experiment to skip auto-download
        if path == "__skip__":
            return
        if not os.path.isfile(path):
            if gdown is not None and gdrive_id:
                try:
                    print(f"[Transformer] Downloading checkpoint ({gdrive_id}) ...")
                    gdown.download(id=gdrive_id, output=path, quiet=False)
                except Exception as e:
                    print(f"[Transformer] download failed: {e}")
                    return
            else:
                print("[Transformer] no checkpoint on disk and gdown unavailable.")
                return
        try:
            import torch.serialization as _ser
            try:
                from dataset import SimpleVocab
                _ser.add_safe_globals([SimpleVocab])
            except (ImportError, AttributeError):
                pass
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            self.load_state_dict(ckpt["model_state_dict"], strict=False)
            self.src_vocab = ckpt.get("src_vocab", None)
            self.tgt_vocab = ckpt.get("tgt_vocab", None)
            print(f"[Transformer] Loaded checkpoint '{path}' (epoch {ckpt.get('epoch','?')})")
        except Exception as e:
            print(f"[Transformer] Could not load checkpoint: {e}")

    # ── AUTOGRADER HOOKS ─────────────────────────────────────────────

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
        return self.generator(y)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    # ── INFERENCE ────────────────────────────────────────────────────

    @staticmethod
    def _tokenize_de(sentence: str) -> List[str]:
        """
        Pure-Python German tokenizer — no spaCy required at inference time.
        Replicates spaCy's whitespace+punctuation splitting by separating
        every punctuation character from surrounding words, then lowercasing.
        Works in any environment without de_core_news_sm installed.
        """
        import re
        tokens = re.findall(r"[\w]+|[^\w\s]", sentence, re.UNICODE)
        return [t.lower() for t in tokens]

    @torch.no_grad()
    def _beam_search(
        self,
        memory:    torch.Tensor,
        src_mask:  torch.Tensor,
        beam_size: int   = 4,
        max_len:   int   = 100,
        alpha:     float = 0.6,
    ) -> List[int]:
        """
        Beam search with length penalty alpha (paper §6.1 uses beam=4, alpha=0.6).
        Returns list of token ids (excluding <sos>, up to but not including <eos>).
        """
        device = memory.device
        V      = self.tgt_tok_emb.embed.num_embeddings

        # Each beam: (log_prob, token_ids_list)
        beams: List[Tuple[float, List[int]]] = [(0.0, [self.sos_idx])]
        completed: List[Tuple[float, List[int]]] = []

        for step in range(max_len):
            if not beams:
                break
            # Build batch of all current beam sequences
            all_ids = [b[1] for b in beams]
            max_t   = max(len(s) for s in all_ids)
            # Pad to same length
            padded  = [s + [self.pad_idx] * (max_t - len(s)) for s in all_ids]
            ys      = torch.tensor(padded, dtype=torch.long, device=device)  # [B, t]

            # Expand memory for all beams
            B_cur   = ys.size(0)
            mem_exp = memory.expand(B_cur, -1, -1)      # [B, S, d]
            sm_exp  = src_mask.expand(B_cur, -1, -1, -1) # [B, 1, 1, S]

            tgt_mask = make_tgt_mask(ys, pad_idx=self.pad_idx)
            logits   = self.decode(mem_exp, sm_exp, ys, tgt_mask)  # [B, t, V]
            # Only look at last position
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)    # [B, V]

            new_beams: List[Tuple[float, List[int]]] = []
            for b_idx, (score, ids) in enumerate(beams):
                lp = log_probs[b_idx]                   # [V]
                # top-k candidates
                top_lp, top_ids = lp.topk(beam_size)
                for lp_val, tok_id in zip(top_lp.tolist(), top_ids.tolist()):
                    new_ids  = ids + [tok_id]
                    new_score = score + lp_val
                    if tok_id == self.eos_idx:
                        # Apply length penalty and add to completed
                        length = len(new_ids) - 1  # exclude <sos>
                        lp_pen = ((5.0 + length) / 6.0) ** alpha
                        completed.append((new_score / lp_pen, new_ids))
                    else:
                        new_beams.append((new_score, new_ids))

            # Keep top beam_size unfinished beams
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_size]

            if len(completed) >= beam_size:
                break

        # If nothing completed, take the best unfinished beam
        if not completed:
            if beams:
                completed = [(s, ids) for s, ids in beams]
            else:
                return []

        # Return the best completed sequence (excluding <sos> and <eos>)
        completed.sort(key=lambda x: x[0], reverse=True)
        best_ids = completed[0][1][1:]   # drop <sos>
        if self.eos_idx in best_ids:
            best_ids = best_ids[:best_ids.index(self.eos_idx)]
        return best_ids

    @torch.no_grad()
    def infer(
        self,
        src_sentence: str,
        max_len:   Optional[int] = None,
        beam_size: int   = 4,
        alpha:     float = 0.6,
    ) -> str:
        """
        Translate a German sentence to English.

        Uses beam search (beam=4, length_penalty=0.6) as in the paper §6.1.
        No spaCy model installation required — pure-Python tokenization.
        Works immediately after Transformer() with no arguments.
        """
        if self.src_vocab is None or self.tgt_vocab is None:
            raise RuntimeError(
                "Vocabulary not loaded. Construct Transformer with checkpoint_path "
                "pointing to a checkpoint that contains src_vocab and tgt_vocab."
            )

        device       = next(self.parameters()).device
        was_training = self.training
        self.eval()

        # Tokenise without spaCy
        tokens   = self._tokenize_de(src_sentence)
        ids      = [self.sos_idx] + [self.src_vocab[t] for t in tokens] + [self.eos_idx]
        src      = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        src_mask = make_src_mask(src, pad_idx=self.pad_idx)

        # Encode once
        memory   = self.encode(src, src_mask)

        # Beam search
        ml       = max_len or self.max_len
        out_ids  = self._beam_search(memory, src_mask,
                                     beam_size=beam_size, max_len=ml, alpha=alpha)

        words = [self.tgt_vocab.lookup_token(i) for i in out_ids]
        if was_training:
            self.train()
        return " ".join(
            w for w in words if w not in ("<pad>", "<sos>", "<eos>", "<unk>")
        )