"""Train a BPE tokenizer on a dataset and encode its splits to uint16 token IDs.

Usage:  uv run scripts/tokenize_and_encode.py {tinystories|owt}
"""

import argparse
import os
import pickle
import resource
import time
from dataclasses import dataclass

from cs336_basics.helpers import timer
from cs336_basics.tokenizer import Tokenizer, encode_to_uint16_bin
from cs336_basics.train_bpe import train_bpe

OUTPUT_DIR = "tokenizer"
SPECIAL_TOKENS = ["<|endoftext|>"]


@dataclass(frozen=True)
class DatasetConfig:
    """Everything that differs between datasets."""

    name: str
    train_txt: str
    valid_txt: str
    vocab_size: int
    num_processes: int

    @property
    def vocab_pkl(self) -> str:
        return os.path.join(OUTPUT_DIR, f"{self.name}_bpe_vocab_{self.vocab_size}.pkl")

    @property
    def merges_pkl(self) -> str:
        return os.path.join(OUTPUT_DIR, f"{self.name}_bpe_merges_{self.vocab_size}.pkl")

    def bin_path(self, split: str) -> str:
        return os.path.join(OUTPUT_DIR, f"{self.name}_{split}.uint16.bin")


CONFIGS = {
    c.name: c
    for c in [
        DatasetConfig(
            name="tinystories",
            train_txt="data/TinyStoriesV2-GPT4-train.txt",
            valid_txt="data/TinyStoriesV2-GPT4-valid.txt",
            vocab_size=10_000,
            num_processes=8,
        ),
        DatasetConfig(
            name="owt",
            train_txt="data/owt_train.txt",
            valid_txt="data/owt_valid.txt",
            vocab_size=32_000,
            num_processes=10,
        ),
    ]
}


def peak_memory_gb() -> float:
    """Peak RSS of this process and its workers. ru_maxrss is bytes on macOS, KiB on Linux."""
    scale = 1 << 30 if os.uname().sysname == "Darwin" else 1 << 20
    return sum(
        resource.getrusage(who).ru_maxrss
        for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
    ) / scale


def train(config: DatasetConfig) -> None:
    """Train BPE on the training split and pickle the vocab and merges."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start = time.perf_counter()
    vocab, merges = train_bpe(
        input_path=config.train_txt,
        vocab_size=config.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        num_processes=config.num_processes,
    )
    elapsed = time.perf_counter() - start

    with open(config.vocab_pkl, "wb") as f:
        pickle.dump(vocab, f)
    with open(config.merges_pkl, "wb") as f:
        pickle.dump(merges, f)

    longest_id, longest_bytes = max(vocab.items(), key=lambda kv: len(kv[1]))

    print(f"  vocab   -> {config.vocab_pkl} ({len(vocab):,} tokens)")
    print(f"  merges  -> {config.merges_pkl} ({len(merges):,} merges)")
    print(f"  elapsed  {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  peak RSS {peak_memory_gb():.2f} GB")
    print(f"  longest  id={longest_id} len={len(longest_bytes)} "
          f"{longest_bytes.decode('utf-8', errors='replace')!r}")


def encode(config: DatasetConfig) -> None:
    """Encode both splits to flat uint16 arrays."""
    tokenizer = Tokenizer.from_files(
        vocab_filepath=config.vocab_pkl,
        merges_filepath=config.merges_pkl,
        special_tokens=SPECIAL_TOKENS,
    )

    for split, source in [("train", config.train_txt), ("valid", config.valid_txt)]:
        destination = config.bin_path(split)
        if os.path.exists(destination):
            print(f"  {split}: {destination} exists, skipping")
            continue
        with timer(f"Encoding {config.name} {split}"):
            encode_to_uint16_bin(tokenizer, source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(CONFIGS))
    parser.add_argument("--retrain", action="store_true", help="retrain BPE even if it exists")
    args = parser.parse_args()

    config = CONFIGS[args.dataset]

    if args.retrain or not os.path.exists(config.merges_pkl):
        train(config)
    else:
        print(f"  BPE: {config.merges_pkl} exists, skipping")

    encode(config)


if __name__ == "__main__":
    main()
