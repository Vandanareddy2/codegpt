import json
import re
from pathlib import Path

PATTERN = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]|\s+")


def encode(text, merges, vocab, pattern):
    chunks = pattern.findall(text)
    token_ids = []

    for chunk in chunks:
        word = list(chunk)

        for pair in merges:
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word

        for token in word:
            token_ids.append(vocab[token])

    return token_ids


def decode(token_ids, vocab):
    id_to_token = {idx: token for token, idx in vocab.items()}
    tokens = [id_to_token[idx] for idx in token_ids]
    return "".join(tokens)


