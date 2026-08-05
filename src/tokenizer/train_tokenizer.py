import re
import json
import time
from pathlib import Path
from datasets import load_from_disk

from src.tokenizer.bpe import build_word_freqs, train_bpe_fast_verbose

PATTERN = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]|\s+")
VOCAB_SIZE = 8000
OUTPUT_DIR = Path("src/tokenizer/output")

def main():
    print("Loading training data...")
    train = load_from_disk("data/train")
    texts = train["text"]
    print(f"Total rows: {len(texts)}")

    print("Building word frequencies...")
    word_freqs = build_word_freqs(texts, PATTERN)
    print(f"Unique starting chunks: {len(word_freqs)}")

    print("Training BPE tokenizer (this will take a while)...")
    start = time.time()
    merges = train_bpe_fast_verbose(word_freqs, vocab_size=VOCAB_SIZE, log_every=500)
    elapsed = time.time() - start
    print(f"Training complete in {elapsed:.1f}s, learned {len(merges)} merges")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    merges_serializable = [list(pair) for pair in merges]
    with open(OUTPUT_DIR / "merges.json", "w") as f:
        json.dump(merges_serializable, f)

    base_chars = sorted(set(ch for word in word_freqs for ch in word))
    vocab = {ch: i for i, ch in enumerate(base_chars)}
    next_id = len(vocab)
    for pair in merges:
        merged_token = pair[0] + pair[1]
        if merged_token not in vocab:
            vocab[merged_token] = next_id
            next_id += 1

    with open(OUTPUT_DIR / "vocab.json", "w") as f:
        json.dump(vocab, f)

    print(f"Saved merges.json and vocab.json to {OUTPUT_DIR}")
    print(f"Final vocab size: {len(vocab)}")

if __name__ == "__main__":
    main()