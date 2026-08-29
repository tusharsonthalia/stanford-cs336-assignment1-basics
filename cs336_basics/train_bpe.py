from collections import Counter, defaultdict
import regex as re
import os
from typing import BinaryIO
import multiprocessing as mp
import heapq
from cs336_basics.helpers import timer


# Global Flags
GPT_2_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT_2_REGEX_COMPILED = re.compile(GPT_2_REGEX)
FUZZY_CHUNK_BOUNDARY_SIZE = 4096
BYTE_TABLE = [bytes([i]) for i in range(256)]
TIMER_FLAG = False

# Custom Type Definitions
Word = tuple[bytes, ...]
Pair = tuple[bytes, bytes]

class RevPair:
    """Helper object for Max heap"""
    __slots__ = ('p',)
    def __init__(self, p): self.p = p
    def __lt__(self, o): return self.p > o.p

def find_chunk_boundaries(
    file: BinaryIO,
    special_token_pattern: bytes,
    num_processes: int,
) -> list[int]:
    """
    Determine the boundaries for chunking the text file for parallel
    pretokenization. Might return fewer than num_processes boundaries.
    """

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    # build the naive chunk boundaries - equally separated
    chunk_size = file_size // num_processes
    chunk_boundaries = [i * chunk_size for i in range(num_processes + 1)]
    chunk_boundaries[-1] = file_size

    # number of bytes to look ahead at chunk boundaries to find special
    # token to partition on
    mini_chunk_size = FUZZY_CHUNK_BOUNDARY_SIZE

    for idx in range(1, num_processes):
        boundary = chunk_boundaries[idx]
        file.seek(boundary) # finding the correct end position for the chunk
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[idx] = file_size
                break

            match = re.search(special_token_pattern, mini_chunk)
            if match:
                chunk_boundaries[idx] = boundary + match.start()
                break
            else:
                boundary += len(mini_chunk)

    # return the list of unique chunk boundaries. Might be less than num_processes
    return sorted(set(chunk_boundaries))

def pretokenize(
    input_path: str | os.PathLike,
    special_tokens_pattern: str,
    file_start: int,
    file_end: int,
) -> Counter[str]:
    """
    Function to pretokenize the word map based on the file chunk and generate
    a mapping of words to frequencies. The chunk is initially split on the
    special tokens to separate documents.
    """
    with open(input_path, "rb") as f:
        f.seek(file_start)
        # reading the normal representation of text for correctness in matching text
        text = f.read(file_end - file_start).decode()

    # removing all the special tokens from chunk
    text = re.split(special_tokens_pattern, text)

    # computing the frequency map for all words in chunk
    word_map = Counter()
    for doc in text:
        word_map.update(re.findall(GPT_2_REGEX_COMPILED, doc))

    return word_map

def make_pairs(word: Word | list[bytes]) -> list[Pair]:
    """Function to make list of pair from a given word of bytes"""
    return [(i, j) for i, j in zip(word, word[1:])]

def merge_pair_in_word(
    pair: Pair,
    word: Word,
    merged: bytes
) -> tuple[Word, Counter[Pair], Counter[Pair]]:
    """
    Function to merge a pair in the give word.
    
    Returns:
    
    - new_word: the word containing the merged pair
    - died: tokens removed from the old word
    - born: new tokens added to the new word
    """
    if len(word) <= 1:
        return word, Counter(), Counter()

    new_word: list[bytes] = []

    i = 0
    n = len(word)
    while i < n - 1:
        # matched pair in the word
        if word[i] == pair[0] and word[i+1] == pair[1]:
            new_word.append(merged)
            i += 1
        # adding the unmatched token
        else:
            new_word.append(word[i])
        # handling the last remaining token
        if i == n - 2:
            new_word.append(word[i+1])
        i += 1

    before = make_pairs(word)
    after = make_pairs(new_word)
    died = Counter(before) - Counter(after)
    born = Counter(after) - Counter(before)

    return tuple(new_word), died, born


def get_bpe(
    word_map: Counter[Word],
    special_tokens: list[str],
    vocab_size: int
) -> tuple[dict[int, bytes], list[Pair]]:
    """
    Function to perform the BPE training loop.
    
    1.  Creates a base vocab of the single byte values from utf-8 
        encoding + special tokens.
    2.  Runs the training loop until len(vocab) - vocab size is reached.
    3.  Create a frequency mapping for all the pairs in all the pretokenized words.
    4.  Selects the max frequency pair from the mapping.
    5.  Merges the max occuring pair across all the words it occurs in.
    6.  Adds the new token to the vocab and adds the specific pair to the 
        list of merges to help in encoding and decoding.

    Returns:
    
    - vocab:    Token ID to token mapping as part of the trained vocab
    - merges:   List of byte tokens merged in order.
    """
    
    byte_count = 256
    vocab: dict[int, bytes] = {i: BYTE_TABLE[i] for i in range(byte_count)}
    vocab.update({byte_count + idx: val.encode('utf-8') for idx, val in enumerate(special_tokens)})

    merge_count = vocab_size - len(vocab)
    merges: list[Pair] = []

    pairs: dict[Pair, int] = Counter()
    # index to map the pairs and the words they occur in
    pairs_to_words: dict[Pair, set[Word]] = defaultdict(set)

    for word, freq in word_map.items():
        word_pairs = make_pairs(word)
        for pair in word_pairs:
            pairs[pair] += freq
            pairs_to_words[pair].add(word)
            
    # max heap to track the max occuring pair efficiently
    max_heap = [(-freq, RevPair(pair), pair) for pair, freq in pairs.items()]
    heapq.heapify(max_heap)
    
    for idx in range(len(vocab), len(vocab) + merge_count):
        max_freq, _, max_pair = max_heap[0]
        while pairs.get(max_pair, 0) != -max_freq:
            heapq.heappop(max_heap)
            max_freq, _, max_pair = max_heap[0]

        merged = b"".join(max_pair)
        
        word_list = list(pairs_to_words[max_pair])
        touched = set()
        for word in word_list:
            freq = word_map[word] 
            new, died, born = merge_pair_in_word(max_pair, word, merged)

            for pair, count in died.items():
                pairs[pair] -= freq * count
                touched.add(pair) # tracking the pair updated to be used for the heap
            for pair, count in born.items():
                pairs[pair] += freq * count
                touched.add(pair) # tracking the pair updated to be used for the heap
                
            # removing the words from the pairs of the old word
            remove_pairs = make_pairs(word)
            for pair in remove_pairs:
                pairs_to_words[pair].discard(word)

            # adding the words to the pairs of the new word
            add_pairs = make_pairs(new)
            for pair in add_pairs:
                pairs_to_words[pair].add(new)

            if new != word:
                word_map[new] += word_map[word]
                del word_map[word]

        for pair in touched:
            freq = pairs[pair]
            if freq > 0:
                heapq.heappush(max_heap, (-freq, RevPair(pair), pair))
            else:
                # pruning dead pairs
                del pairs[pair]
                del pairs_to_words[pair]

        merges.append((max_pair[0], max_pair[1]))
        vocab[idx] = b"".join(max_pair)

    return vocab, merges


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    *,
    num_processes: int = 1
) -> tuple[dict[int, bytes], list[Pair]]:
    if vocab_size <= 256 + len(special_tokens):
        raise ValueError(f"""Vocab Size: {vocab_size} is too low to train
                         the tokenizer. It should be >= {256 + len(special_tokens)}""")

    with timer("Chunk Boundaries", TIMER_FLAG):
        # 1. Chunk the file for parallelly pre-tokenize
        split_special_tokens = "|".join((re.escape(s) for s in sorted(special_tokens, key=len, reverse=True)))
        with open(input_path, "rb") as f:
            chunks = find_chunk_boundaries(f, split_special_tokens.encode('utf-8'), num_processes)

    with timer("Pre Tokenization", TIMER_FLAG):
        # 2. Pre-tokenize the documents across multiple documents in the chunk
        process_count = len(chunks) - 1
        args = [(input_path, split_special_tokens, chunks[i], chunks[i + 1]) for i in range(process_count)]
        with mp.Pool(processes=process_count) as pool:
            word_map_partials = pool.starmap(pretokenize, args)

    with timer("Processing Pretokens", TIMER_FLAG):
        # 3. Merge all the pretokenized frequency maps
        word_map = Counter()
        for word_map_partial in word_map_partials:
            word_map.update(word_map_partial)

        # 3.1 convert all the string keys in the word map to tuple of bytes
        temp_map = Counter()
        for key, freq in word_map.items():
            temp_map[tuple(BYTE_TABLE[i] for i in key.encode('utf-8'))] = freq
        word_map = temp_map

    with timer("BPE Training", TIMER_FLAG):
        # 4. Begin the BPE proces
        vocab, merges = get_bpe(word_map, special_tokens, vocab_size)

    return vocab, merges