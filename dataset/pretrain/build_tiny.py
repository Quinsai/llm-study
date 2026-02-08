from datasets import load_dataset

dataset = load_dataset(
    "wikitext",
    "wikitext-2-raw-v1",
    split="train"
)

N = 30000
texts = []

for i in range(N):
    text = dataset[i]['text'].strip()
    if len(text) > 0:
        texts.append(text)

with open("dataset/tiny_corpus.txt", "w", encoding="utf-8") as f:
    for line in texts:
        f.write(line + "\n")

print(f"Saved {len(texts)} lines to tiny_corpus.txt")