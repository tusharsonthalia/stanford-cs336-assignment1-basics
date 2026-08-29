import re

class Tokenizer:
    def __init__(self, special_tokens):
        self.special_tokens = special_tokens
        self.tokens = {}
        self.merges = []
        self.vocab_list = [i.encode('utf-8') for i in self.special_tokens]
        self.vocab_list += [bytes([i]) for i in range(256)]

    def _pretokenize(self, vocab):
        pattern = r'\S+'
        for token in re.finditer(pattern, vocab, 0):
            token = tuple(bytes([i]) for i in token.group().encode('utf-8'))
            self.tokens[token] = self.tokens.get(token, 0) + 1

    def _count_pairs(self):
        pair_to_freq = {}

        for token, freq in self.tokens.items():
            n = len(token)
            if n > 1:
                for i in range(n - 1):
                    pair = (token[i], token[i + 1])
                    pair_to_freq[pair] = pair_to_freq.get(pair, 0) + freq

        return pair_to_freq

    def _pick_max(self, pairs):
        max_freq = max(pairs.values())
        max_pair = max(pair for pair, freq in pairs.items() if freq == max_freq)

        return max_pair

    def _apply_merge(self, max_pair):
        staging_tokens = self.tokens.copy()
        result = b"".join(max_pair)

        for token, freq in self.tokens.items():
            n = len(token)
            if n > 1:
                temp_token = []
                i = 0
                while i < (n - 1):
                    pair = (token[i], token[i + 1])
                    if pair == max_pair:
                        temp_token.append(result)
                        i += 1
                    else:
                        temp_token.append(token[i])
                    if i == n - 2:
                        temp_token.append(token[i + 1])
                    i += 1
                temp_token = tuple(temp_token)
                if temp_token != token:
                    del staging_tokens[token]
                    staging_tokens[temp_token] = staging_tokens.get(temp_token, 0) + freq
        self.tokens = staging_tokens
        
    
    def _merge(self, vocab_size):
        merges = vocab_size - len(self.vocab_list)
        if merges <= 0:
            return

        for _ in range(merges):
            pairs = self._count_pairs()

            if len(pairs) == 0:
                return

            max_pair = self._pick_max(pairs)

            self._apply_merge(max_pair)

            self.merges.append(max_pair)
            self.vocab_list.append(b"".join(max_pair))

            # print(self.tokens)
            
        return

    def tokenize(self, vocab, vocab_size=500):
        self._pretokenize(vocab)
        self._merge(vocab_size)

if __name__ == "__main__":
    text =  """
            low low low low low
            lower lower widest widest widest
            newest newest newest newest newest newest
            """
    special_tokens = ["<|endoftext|>"]

    tokenizer = Tokenizer(special_tokens)
    tokenizer.tokenize(text, 263)

