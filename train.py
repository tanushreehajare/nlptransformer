"""
train.py — Training pipeline, inference & evaluation
DA6401 Assignment 3.
"""

import math
import os
import time
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import (
    Multi30kDataset, SimpleVocab,
    PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX,
    make_dataloaders, collate_fn,
)
from lr_scheduler import NoamScheduler


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    KL-divergence label smoothing as in "Attention Is All You Need".

        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    PAD positions receive zero probability mass and are excluded from the loss.
    """

    def __init__(self, vocab_size: int, pad_idx: int = PAD_IDX, smoothing: float = 0.1) -> None:
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits : [B*T, V]
        target : [B*T]
        """
        assert logits.size(1) == self.vocab_size
        log_probs = F.log_softmax(logits, dim=-1)

        # Build smoothed target distribution (excluding pad as a possible class).
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.pad_idx] = 0
            mask = (target == self.pad_idx)
            true_dist[mask] = 0  # zero rows at pad targets

        # Sum-then-divide-by-non-pad-tokens loss
        loss = -(true_dist * log_probs).sum(dim=1)  # [B*T]
        n_tokens = (~mask).sum().clamp(min=1)
        return loss.sum() / n_tokens


# ══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    clip_grad: float = 1.0,
    wandb_run=None,
    log_every: int = 50,
    pad_idx: int = PAD_IDX,
) -> float:
    """One epoch of training or evaluation. Returns the average per-token loss."""
    model.train(is_train)
    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()

    for step, (src, tgt) in enumerate(data_iter):
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        # Shift target for teacher forcing: input is tgt[:, :-1], gold is tgt[:, 1:]
        tgt_in  = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask = make_src_mask(src, pad_idx=pad_idx)
        tgt_mask = make_tgt_mask(tgt_in, pad_idx=pad_idx)

        with torch.set_grad_enabled(is_train):
            logits = model(src, tgt_in, src_mask, tgt_mask)        # [B, T-1, V]
            B, T, V = logits.shape
            loss = loss_fn(logits.reshape(B * T, V), tgt_out.reshape(B * T))

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Compute total grad norm BEFORE clipping (for W&B logging).
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad).item()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if wandb_run is not None and (step % log_every == 0):
                with torch.no_grad():
                    probs = F.softmax(logits, dim=-1)
                    pred_conf = probs.max(dim=-1).values.mean().item()
                wandb_run.log({
                    "train/loss":      loss.item(),
                    "train/lr":        optimizer.param_groups[0]["lr"],
                    "train/grad_norm": grad_norm,
                    "train/pred_conf": pred_conf,
                    "epoch":           epoch_num,
                })

        n_tok = (tgt_out != pad_idx).sum().item()
        total_loss   += loss.item() * n_tok
        total_tokens += n_tok

    avg_loss = total_loss / max(total_tokens, 1)
    elapsed  = time.time() - t0
    tag = "train" if is_train else "val"
    print(f"[epoch {epoch_num}] {tag} loss={avg_loss:.4f}  ppl={math.exp(min(avg_loss,20)):.2f}  ({elapsed:.1f}s)")
    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int = EOS_IDX,
    device: str = "cpu",
) -> torch.Tensor:
    """Token-by-token greedy decoding. Returns [1, out_len] including <sos> and (if reached) <eos>."""
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, pad_idx=PAD_IDX)
        logits = model.decode(memory, src_mask, ys, tgt_mask)   # [1, t, V]
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_tok], dim=1)
        if next_tok.item() == end_symbol:
            break
    return ys


# ══════════════════════════════════════════════════════════════════════
#  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def _ids_to_tokens(ids: List[int], vocab: SimpleVocab) -> List[str]:
    out = []
    for i in ids:
        if i == EOS_IDX:
            break
        if i in (SOS_IDX, PAD_IDX):
            continue
        out.append(vocab.lookup_token(i))
    return out


def _corpus_bleu(hypotheses: List[List[str]], references: List[List[str]]) -> float:
    """Corpus-level BLEU-4 in [0, 100]. Uses sacrebleu if available, else NLTK, else 0."""
    try:
        import sacrebleu
        hyp_strs = [" ".join(h) for h in hypotheses]
        ref_strs = [" ".join(r) for r in references]
        return float(sacrebleu.corpus_bleu(hyp_strs, [ref_strs], force=True).score)
    except Exception:
        pass
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        sm = SmoothingFunction().method1
        return 100.0 * corpus_bleu(
            [[r] for r in references], hypotheses, smoothing_function=sm
        )
    except Exception:
        return 0.0


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab: SimpleVocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """Greedy-decode the entire test set and return corpus BLEU (0–100)."""
    model.eval()
    hypotheses: List[List[str]] = []
    references: List[List[str]] = []

    with torch.no_grad():
        for src, tgt in test_dataloader:
            src = src.to(device)
            for i in range(src.size(0)):
                src_i = src[i:i+1]
                src_mask_i = make_src_mask(src_i, pad_idx=PAD_IDX)
                ys = greedy_decode(model, src_i, src_mask_i,
                                   max_len=max_len,
                                   start_symbol=SOS_IDX,
                                   end_symbol=EOS_IDX,
                                   device=device)
                pred_ids = ys.squeeze(0).tolist()
                ref_ids  = tgt[i].tolist()
                hypotheses.append(_ids_to_tokens(pred_ids, tgt_vocab))
                references.append(_ids_to_tokens(ref_ids,  tgt_vocab))

    return _corpus_bleu(hypotheses, references)


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
    src_vocab: Optional[SimpleVocab] = None,
    tgt_vocab: Optional[SimpleVocab] = None,
    extra: Optional[dict] = None,
) -> None:
    """Save full state (model, optim, scheduler, epoch, model_config, vocabs)."""
    model_config = {
        "src_vocab_size":  model.src_tok_emb.embed.num_embeddings,
        "tgt_vocab_size":  model.tgt_tok_emb.embed.num_embeddings,
        "d_model":         model.d_model,
        "N":               len(model.encoder.layers),
        "num_heads":       model.encoder.layers[0].self_attn.num_heads,
        "d_ff":            model.encoder.layers[0].ffn.linear1.out_features,
        "dropout":         model.encoder.layers[0].dropout1.p,
        "max_len":         model.max_len,
        "pad_idx":         model.pad_idx,
        "sos_idx":         model.sos_idx,
        "eos_idx":         model.eos_idx,
    }
    blob = {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "model_config":         model_config,
        "src_vocab":            src_vocab if src_vocab is not None else model.src_vocab,
        "tgt_vocab":            tgt_vocab if tgt_vocab is not None else model.tgt_vocab,
    }
    if extra:
        blob.update(extra)
    torch.save(blob, path)
    print(f"[checkpoint] saved → {path}")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict"):
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if "src_vocab" in ckpt:
        model.src_vocab = ckpt["src_vocab"]
    if "tgt_vocab" in ckpt:
        model.tgt_vocab = ckpt["tgt_vocab"]
    return int(ckpt.get("epoch", 0))


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment(
    *,
    epochs: int = 30,
    batch_size: int = 128,
    d_model: int = 256,
    N: int = 3,
    num_heads: int = 8,
    d_ff: int = 1024,
    dropout: float = 0.1,
    smoothing: float = 0.1,
    warmup_steps: int = 4000,
    use_noam: bool = True,
    fixed_lr: float = 1e-4,
    scale_attention: bool = True,    # §2.2 ablation
    positional: str = "sinusoidal",  # §2.4 ablation
    project: str = "da6401-a3",
    run_name: Optional[str] = None,
    save_path: str = "checkpoint.pt",
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    # ─── W&B (optional, soft import) ──────────────────────────────────
    try:
        import wandb
        wandb_run = wandb.init(
            project=project, name=run_name,
            config=dict(
                epochs=epochs, batch_size=batch_size, d_model=d_model, N=N,
                num_heads=num_heads, d_ff=d_ff, dropout=dropout,
                smoothing=smoothing, warmup_steps=warmup_steps, use_noam=use_noam,
                fixed_lr=fixed_lr, scale_attention=scale_attention,
                positional=positional,
            ),
        )
    except Exception as e:
        print(f"[wandb] disabled: {e}")
        wandb_run = None

    # ─── data ─────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = make_dataloaders(
        batch_size=batch_size,
    )
    print(f"[vocab] src={len(src_vocab)}  tgt={len(tgt_vocab)}")

    # ─── model / optim / sched / loss ─────────────────────────────────
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model, N=N, num_heads=num_heads, d_ff=d_ff, dropout=dropout,
        positional=positional, scale_attention=scale_attention,
    ).to(device)
    model.src_vocab = src_vocab
    model.tgt_vocab = tgt_vocab

    if use_noam:
        # Noam expects base LR = 1.0
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0,
                                     betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=fixed_lr,
                                     betas=(0.9, 0.98), eps=1e-9)
        scheduler = None

    loss_fn = LabelSmoothingLoss(vocab_size=len(tgt_vocab),
                                 pad_idx=PAD_IDX, smoothing=smoothing)

    # ─── train ────────────────────────────────────────────────────────
    best_bleu = -1.0
    for epoch in range(1, epochs + 1):
        run_epoch(train_loader, model, loss_fn, optimizer, scheduler,
                  epoch_num=epoch, is_train=True, device=device, wandb_run=wandb_run)
        run_epoch(val_loader, model, loss_fn, None, None,
                  epoch_num=epoch, is_train=False, device=device, wandb_run=wandb_run)

        val_bleu = evaluate_bleu(model, val_loader, tgt_vocab, device=device, max_len=80)
        print(f"[epoch {epoch}] val BLEU = {val_bleu:.2f}")
        if wandb_run is not None:
            wandb_run.log({"val/bleu": val_bleu, "epoch": epoch})

        if val_bleu > best_bleu:
            best_bleu = val_bleu
            save_checkpoint(model, optimizer, scheduler, epoch, path=save_path,
                            src_vocab=src_vocab, tgt_vocab=tgt_vocab)

    # ─── final test BLEU ─────────────────────────────────────────────
    load_checkpoint(save_path, model)
    model.to(device)
    test_bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device, max_len=100)
    print(f"[FINAL] test BLEU = {test_bleu:.2f}")
    if wandb_run is not None:
        wandb_run.log({"test/bleu": test_bleu})
        wandb_run.finish()


if __name__ == "__main__":
    run_training_experiment()