from datasets import load_dataset

dataset = load_dataset(
    "tatsu-lab/alpaca",
    split="train"
)

N = 30000
texts = []

for i in range(N):
    user_input = dataset[i]['instruction']
    if len(dataset[i]["input"]) > 0:
        user_input += ": " + dataset[i]["input"]
    agent_output = dataset[i]['output']
    texts.append([user_input, agent_output])

with open("dataset/tiny_sft_message.txt", "w", encoding="utf-8") as f:
    for message in texts:
        f.write("###User:\n" + message[0] + "\n")
        f.write("###Agent:\n" + message[1] + "\n")
        f.write("#----end----#\n")
        f.write("\n")

print(f"Saved {len(texts)} lines to tiny_sft_message.txt")