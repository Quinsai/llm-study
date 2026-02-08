from dataset.tokenizer.bpe_tokenizer import BPETokenizer
from trainer.tiny_trainer import TinyTrainer
import os
import argparse

block_size=128
batch_size=64
num_epoch_pretrain=500
num_epoch_sft=400
iter_per_epoch=10
lr=3e-4

fixed_prompt1 = "###User:\n"
fixed_prompt_len1 = len(fixed_prompt1)
fixed_prompt2 = "\n###Agent:\n"
fixed_prompt_len2 = len(fixed_prompt2)
fixed_prompt_len = fixed_prompt_len1 + fixed_prompt_len2

if __name__ == "__main__":
    corpus_path = "dataset/tiny_corpus.txt"
    sft_message_path = "dataset/tiny_sft_message.txt"
    tokenizer_path = "dataset/tokenizer/tokenizer.msg"
    pretrain_model_path = "model/pretrain_model.pt"
    sft_model_path = "model/sft_model.pt"

    parser = argparse.ArgumentParser(description="运行参数")
    # generate：不重新训练，直接用旧模型
    # sft：重新sft，用保存的预训练模型
    # pretrain：从预训练开始重新训练，只使用保存的tokenizer
    # tokenizer：tokenizer也重新训练
    parser.add_argument("--start_from", type=str, default="generate", help="重新训练开始阶段")
    args = parser.parse_args()
    start_from = args.start_from

    if start_from == "tokenizer" or not os.path.exists(tokenizer_path):
        tokenizer = BPETokenizer(corpus_path, sft_message_path, tokenizer_path)
    else:
        tokenizer = BPETokenizer.load(tokenizer_path)
    print("------tokenizer ready------")

    trainer = TinyTrainer(
        corpus_path, 
        sft_message_path,
        block_size,
        batch_size,
        num_epoch_pretrain,
        num_epoch_sft,
        iter_per_epoch,
        lr,
        fixed_prompt1,
        fixed_prompt2,
        tokenizer
    )

    load_model_path = ""

    if start_from == "pretrain" or start_from == "tokenizer" or not os.path.exists(pretrain_model_path):
        trainer.pretrain(draw=True, save_path=pretrain_model_path)
        print("------end of pretrain-----")
        trainer.sft_train(draw=True, save_path=sft_model_path)
        print("------end of sft---------")
    elif start_from == "sft" or not os.path.exists(sft_model_path):
        trainer.load_checkpoint(pretrain_model_path)
        print("------end of pretrain-----")
        trainer.sft_train(draw=True, save_path=sft_model_path)
        print("------end of sft---------")
    else:
        trainer.load_checkpoint(sft_model_path)
        print("------end of pretrain-----")
        print("------end of sft---------")

    while True:
        prompt = input("User: ")
        if prompt == "quit":
            break
        output = trainer.generate(prompt)
        print("Agent: ", output)
