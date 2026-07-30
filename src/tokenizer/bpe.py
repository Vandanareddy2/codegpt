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


