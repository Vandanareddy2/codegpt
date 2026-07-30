def get_pair_counts(word_freqs):
    """
    word_freqs: dict mapping a tuple of tokens -> frequency count
                e.g. {("d","e","f"): 2, ("r","e","t"): 1}
    
    Returns: dict mapping a pair (tuple of 2 tokens) -> total count
             e.g. {("d","e"): 2, ("e","f"): 2, ("r","e"): 1, ("e","t"): 1}
    """
    pair_counts = {}

    for word, freq in word_freqs.items():          # go through each word and its frequency
        for i in range(len(word) - 1):              # walk through adjacent positions
            pair = (word[i], word[i + 1])            # grab the pair at this position
            pair_counts[pair] = pair_counts.get(pair, 0) + freq   # add this word's freq to the pair's total

    return pair_counts

def get_most_frequent_pair(pair_counts):
    """
    pair_counts: dict mapping pair -> count (output of get_pair_counts)
    Returns: the pair (tuple of 2 tokens) with the highest count
    """
    return max(pair_counts, key=pair_counts.get)

def merge_pair(pair, word_freqs):
    """
    pair: tuple of 2 tokens to merge, e.g. ("d", "e")
    word_freqs: dict mapping word-tuple -> frequency

    Returns: a NEW word_freqs dict where every occurrence of `pair`
             in each word has been merged into a single combined token.
             e.g. ("d","e","f") with pair ("d","e") becomes ("de","f")
    """
    new_word_freqs = {}
    merged_token = pair[0] + pair[1]        # e.g. "d" + "e" -> "de"

    for word, freq in word_freqs.items():
        new_word = []
        i = 0
        while i < len(word):
            # check if current position matches the pair we're merging
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(merged_token)   # combine into one token
                i += 2                           # skip both tokens we just merged
            else:
                new_word.append(word[i])         # keep token as-is
                i += 1                            # move ahead by one

        new_word_freqs[tuple(new_word)] = freq

    return new_word_freqs

def train_bpe(word_freqs, vocab_size):
    """
    word_freqs: dict mapping word-tuple -> frequency (starting point, e.g. character-level)
    vocab_size: target number of merge operations to perform

    Returns:
        merges: an ORDERED list of pairs, in the order they were merged
                e.g. [('d','e'), ('de','f')]
    """
    merges = []

    for _ in range(vocab_size):
        pair_counts = get_pair_counts(word_freqs)

        if not pair_counts:            # no more pairs left to merge (corpus exhausted)
            break

        best_pair = get_most_frequent_pair(pair_counts)
        merges.append(best_pair)        # record this merge, in order

        word_freqs = merge_pair(best_pair, word_freqs)   # apply it, update for next round

    return merges

def build_word_freqs(texts, pattern):
    """
    texts: list of raw strings (your Python code samples)
    pattern: compiled regex for pre-tokenization

    Returns: dict mapping character-tuple -> frequency count
             e.g. {('d','e','f'): 120, (' ',): 5000, ...}
    """
    word_freqs = {}

    for text in texts:
        chunks = pattern.findall(text)          # split text into initial chunks
        for chunk in chunks:
            char_tuple = tuple(chunk)            # break chunk into individual characters
            word_freqs[char_tuple] = word_freqs.get(char_tuple, 0) + 1

    return word_freqs

if __name__ == "__main__":
    import re
    from datasets import load_from_disk

    PATTERN = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]|\s+")

    train = load_from_disk("data/train")
    sample_texts = train["text"][:1000]        # small sample first

    word_freqs = build_word_freqs(sample_texts, PATTERN)
    print(f"Unique starting chunks: {len(word_freqs)}")

    merges = train_bpe(word_freqs, vocab_size=200)
    print(f"Learned {len(merges)} merges")
    print("First 20 merges:", merges[:20])