import pickle
from collections.abc import Iterable, Iterator

import numpy as np
import regex as re

from cs336_basics.train_bpe import BYTE_TABLE, GPT_2_REGEX_COMPILED


class Tokenizer:
    """Byte-level BPE tokenizer: text <-> token IDs.

    Encoding is not a re-run of training. Training chose which pairs to merge
    by counting frequencies; encoding replays those choices in the order they
    were learned. So the only state needed is the merge order, and the same text
    always produces the same IDs.

    Args:
        vocab: Token ID to token bytes, as produced by BPE training.
        merges: Learned pairs in the order they were learned. Position in this
            list is the merge's rank, which is the only thing encoding needs.
        special_tokens: Strings to treat as atomic. Matched literally before the
            regex ever sees the text, and mapped straight to their vocab ID --
            never byte-split.
    """
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.vocab_sentinel = len(vocab) + 1
        
        # compiled regex object for pretokenization
        self.regex = GPT_2_REGEX_COMPILED

        # reverse index: encoding needs bytes -> id, decoding needs id -> bytes
        self.vocab_to_id = {pair: id for id, pair in vocab.items()}

        if special_tokens is None:
            self.special_tokens = []
            self.special_tokens_regex_compiled = re.compile("(?!)")
        else:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            self.special_tokens_regex_compiled = re.compile(f"({"|".join(re.escape(token) for token in self.special_tokens)})")

        # pair -> rank in vocab
        self.merges = {merge: id for id, merge in enumerate(merges)}

        # pretoken -> ids. bounded by the vocabulary of the corpus, not its size.
        self._bpe_cache = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        """Build a Tokenizer from the pickles written by the BPE training scripts.

        Args:
            vocab_filepath (str): Pickled dict[int, bytes].
            merges_filepath (str): Pickled list[tuple[bytes, bytes]], in merge order.
            special_tokens: Passed through to __init__.

        Returns:
            A Tokenizer ready to encode and decode.
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _get_bpe(self, token: str) -> list[int]:
        """Merge one pretoken down to token IDs, memoized on the pretoken.

        Repeatedly finds the lowest-ranked mergeable pair currently present and
        merges it, until no pair is a learned merge. Each merge shortens the
        sequence by one, so this runs at most len(token) - 1 times.

        Args:
            token (str): A single pretoken from the GPT-2 regex.

        Returns:
            The token IDs for this pretoken. The returned list is the cached
            object -- callers must not mutate it.
        """
        cached = self._bpe_cache.get(token, None)
        if cached is not None:
            return cached

        merged_tokens = [BYTE_TABLE[i] for i in  token.encode("utf-8")]

        while True:
            min_rank = self.vocab_sentinel
            min_idx = len(merged_tokens)
            
            for i in range(len(merged_tokens) - 1):
                a, b = merged_tokens[i], merged_tokens[i+1]
                rank = self.merges.get((a,b), self.vocab_sentinel)
                if rank == self.vocab_sentinel:
                    continue
                
                if rank < min_rank:
                    min_rank = rank
                    min_idx = i
                
            if min_rank == self.vocab_sentinel:
                break
            
            merged_tokens[min_idx:min_idx+2] = [merged_tokens[min_idx] + merged_tokens[min_idx+1]]

        result = [self.vocab_to_id[merge] for merge in merged_tokens]
        self._bpe_cache[token] = result

        return result

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs.

        Special tokens are carved out first so the regex can never split one,
        then each remaining fragment is pretokenized and merged independently.

        Args:
            text (str): Arbitrary text.

        Returns:
            One ID per token, in order.
        """
        token_ids = []

        # capturing group keeps the delimiters, so special tokens survive the split
        split_text = re.split(self.special_tokens_regex_compiled, text)
        for text in split_text:
            if match := re.match(self.special_tokens_regex_compiled, text):
                token_ids += [self.vocab_to_id[match.group().encode('utf-8')]]
            else:
                for token in self.regex.findall(text):
                    token_ids += self._get_bpe(token)

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode an iterable of strings, yielding one ID at a time.

        Args:
            iterable: Any iterable of strings, typically an open file handle.

        Yields:
            Token IDs, in order.
        """
        # specials first so alternation prefers them over their byte pieces --
        # a cut may never land inside a special token
        specials = "|".join(re.escape(s) for s in self.special_tokens)
        pattern = self.regex.pattern
        boundary = re.compile(f"{specials}|{pattern}" if specials else pattern)

        buf = ""
        keep = max(map(len, self.special_tokens), default=1) - 1

        for chunk in iterable:
            buf += chunk
            limit = len(buf) - keep      # leaves room for a partial special token
            pos = max(0, len(buf) - 1024)

            cut = 0
            for m in boundary.finditer(buf, pos):
                s = m.start()
                # the prefix must not end in whitespace: \s+(?!\S) re-tokenizes a
                # trailing run once it sits at end-of-string
                if 0 < s <= limit and not buf[s - 1].isspace():
                    cut = s

            if cut:
                yield from self.encode(buf[:cut])
                buf = buf[cut:]

        if buf:
            yield from self.encode(buf)

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back into text.

        All token bytes are concatenated before decoding, because a multi-byte
        UTF-8 character can straddle two tokens.

        Args:
            ids: Token IDs. Need not have come from encode().

        Returns:
            The decoded text. Byte sequences that aren't valid UTF-8 become
            U+FFFD rather than raising, so arbitrary ID sequences are safe.
        """
        result = b"".join(self.vocab[id] for id in ids)

        return result.decode(encoding="utf-8", errors="replace")

def encode_to_uint16_bin(
    tokenizer: Tokenizer, 
    input_path: str, 
    output_path: str, 
    chunk_tokens: int = 1_000_000
) -> None:
    """Stream-encode a text file into a flat uint16 array of token IDs on disk."""
    with open(input_path, encoding="utf-8") as fin, open(output_path, "wb") as fout:
        acc = []

        # 16MB reads
        for token_id in tokenizer.encode_iterable(iter(lambda: fin.read(1 << 24), "")):
            acc.append(token_id)
            if len(acc) == chunk_tokens:
                np.fromiter(acc, np.uint16, len(acc)).tofile(fout)
                acc.clear()
        np.fromiter(acc, np.uint16, len(acc)).tofile(fout)


