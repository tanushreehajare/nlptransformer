"""
dataset.py — Multi30k DE→EN dataset with spaCy tokenization
DA6401 Assignment 3.

Special tokens (fixed indices, used everywhere):
    <unk> = 0
    <pad> = 1
    <sos> = 2
    <eos> = 3
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]


# ══════════════════════════════════════════════════════════════════════
#  spaCy loaders (lazy, robust to missing model packages)
# ══════════════════════════════════════════════════════════════════════

def _load_spacy(name: str):
    import spacy
    try:
        return spacy.load(name)
    except OSError:
        from spacy.cli import download
        download(name)
        return spacy.load(name)


def get_tokenizers():
    """Return (de_tokenizer_callable, en_tokenizer_callable)."""
    nlp_de = _load_spacy("de_core_news_sm")
    nlp_en = _load_spacy("en_core_web_sm")

    def tok_de(text: str) -> List[str]:
        return [t.text.lower() for t in nlp_de.tokenizer(text.strip())]

    def tok_en(text: str) -> List[str]:
        return [t.text.lower() for t in nlp_en.tokenizer(text.strip())]

    return tok_de, tok_en


# ══════════════════════════════════════════════════════════════════════
#  Vocabulary
# ══════════════════════════════════════════════════════════════════════

class SimpleVocab:
    """Pickleable vocabulary with the same API the rest of the code expects."""

    def __init__(self, stoi: Dict[str, int]):
        self.stoi: Dict[str, int] = dict(stoi)
        self.itos: List[str] = [None] * len(self.stoi)
        for tok, idx in self.stoi.items():
            self.itos[idx] = tok

    def __len__(self) -> int:
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, UNK_IDX)

    def lookup_token(self, idx: int) -> str:
        if 0 <= idx < len(self.itos):
            return self.itos[idx]
        return "<unk>"

    @classmethod
    def from_counter(cls, counter: Counter, min_freq: int = 2) -> "SimpleVocab":
        stoi = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for tok, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            if freq < min_freq or tok in stoi:
                continue
            stoi[tok] = len(stoi)
        return cls(stoi)


# ══════════════════════════════════════════════════════════════════════
#  Multi30k dataset
# ══════════════════════════════════════════════════════════════════════

class Multi30kDataset(Dataset):
    """
    Loads the Multi30k DE→EN dataset (HF: bentrevett/multi30k) and provides:
      - tokenization via spaCy (de_core_news_sm / en_core_web_sm)
      - vocabulary build over the train split
      - integer token sequences with <sos>...<eos>
    """

    def __init__(
        self,
        split: str = "train",
        src_vocab: Optional[SimpleVocab] = None,
        tgt_vocab: Optional[SimpleVocab] = None,
        min_freq: int = 2,
    ) -> None:
        from datasets import load_dataset

        self.split = split
        ds = load_dataset("bentrevett/multi30k", split=split)
        self.de_sents: List[str] = [ex["de"] for ex in ds]
        self.en_sents: List[str] = [ex["en"] for ex in ds]

        self.tok_de, self.tok_en = get_tokenizers()

        # Pre-tokenize once (Multi30k is small ≤ 29k pairs).
        self.de_tokens: List[List[str]] = [self.tok_de(s) for s in self.de_sents]
        self.en_tokens: List[List[str]] = [self.tok_en(s) for s in self.en_sents]

        if src_vocab is None or tgt_vocab is None:
            assert split == "train", \
                "vocab can only be built from train split; pass src_vocab/tgt_vocab for val/test"
            self.src_vocab, self.tgt_vocab = self._build_vocabs(min_freq=min_freq)
        else:
            self.src_vocab, self.tgt_vocab = src_vocab, tgt_vocab

    def _build_vocabs(self, min_freq: int) -> Tuple[SimpleVocab, SimpleVocab]:
        src_counter, tgt_counter = Counter(), Counter()
        for toks in self.de_tokens:
            src_counter.update(toks)
        for toks in self.en_tokens:
            tgt_counter.update(toks)
        return (
            SimpleVocab.from_counter(src_counter, min_freq=min_freq),
            SimpleVocab.from_counter(tgt_counter, min_freq=min_freq),
        )

    # API expected by the skeleton (kept for backward compat) ─────────
    def build_vocab(self) -> Tuple[SimpleVocab, SimpleVocab]:
        return self.src_vocab, self.tgt_vocab

    def process_data(self):
        return self  # tokenization already done eagerly in __init__

    def encode(self, tokens: List[str], vocab: SimpleVocab) -> List[int]:
        return [SOS_IDX] + [vocab[t] for t in tokens] + [EOS_IDX]

    def __len__(self) -> int:
        return len(self.de_tokens)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src = self.encode(self.de_tokens[idx], self.src_vocab)
        tgt = self.encode(self.en_tokens[idx], self.tgt_vocab)
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


# ══════════════════════════════════════════════════════════════════════
#  Collate function (dynamic padding to longest in the batch)
# ══════════════════════════════════════════════════════════════════════

def collate_fn(batch):
    src_list, tgt_list = zip(*batch)
    src = pad_sequence(src_list, batch_first=True, padding_value=PAD_IDX)
    tgt = pad_sequence(tgt_list, batch_first=True, padding_value=PAD_IDX)
    return src, tgt


def make_dataloaders(
    batch_size: int = 64,
    min_freq: int = 2,
    num_workers: int = 2,
):
    """Helper to build train/val/test DataLoaders with shared train-built vocab."""
    train_ds = Multi30kDataset("train", min_freq=min_freq)
    val_ds   = Multi30kDataset("validation",
                               src_vocab=train_ds.src_vocab,
                               tgt_vocab=train_ds.tgt_vocab)
    test_ds  = Multi30kDataset("test",
                               src_vocab=train_ds.src_vocab,
                               tgt_vocab=train_ds.tgt_vocab)

    common = dict(collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **common)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **common)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader, train_ds.src_vocab, train_ds.tgt_vocab