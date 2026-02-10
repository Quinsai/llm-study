from dataset.tokenizer.bpe_tokenizer import BPETokenizer
from trainer.tiny_trainer import TinyTrainer
from dataset.pretrain.dataset import build_pretrain_bin
from dataset.sft.dataset import build_sft_bin
import os
import argparse

block_size=256
batch_size=32
num_epoch_pretrain=700
num_epoch_sft=600
iter_per_epoch=10
lr_pretrain=3e-4
lr_sft=1e-4

fixed_prompt1 = "###User:\n"
fixed_prompt_len1 = len(fixed_prompt1)
fixed_prompt2 = "\n###Agent:\n"
fixed_prompt_len2 = len(fixed_prompt2)
fixed_prompt_len = fixed_prompt_len1 + fixed_prompt_len2

if __name__ == "__main__":
    corpus_path = "dataset/tiny_corpus.txt"
    sft_message_path = "dataset/tiny_sft_message.txt"
    tokenizer_path = "dataset/tokenizer/tokenizer.msg"
    pretrain_bin_path = "dataset/pretrain/encode.bin"
    sft_bin_prefix = "dataset/sft/encode"
    pretrain_model_path = "model/pretrain_model.pt"
    sft_model_path = "model/sft_model.pt"

    parser = argparse.ArgumentParser(description="运行参数")
    # generate：不重新训练，直接用旧模型
    # sft：重新sft，用保存的预训练模型
    # pretrain：从预训练开始重新训练，只使用保存的tokenizer及其编码出的encode
    # encode：使用保存的tokenizer重新对样本编码encode
    # tokenizer：tokenizer也重新训练
    parser.add_argument("--start_from", type=str, default="generate", help="重新训练开始阶段")
    args = parser.parse_args()
    start_from = args.start_from

    if start_from == "tokenizer" or not os.path.exists(tokenizer_path):
        tokenizer = BPETokenizer(corpus_path, sft_message_path, tokenizer_path)
    else:
        tokenizer = BPETokenizer.load(tokenizer_path)
    print("------tokenizer ready------")

    if start_from == "encode" or start_from =="tokenizer" or not os.path.exists(pretrain_bin_path) or not os.path.exists(sft_bin_prefix+"_x.bin") or not os.path.exists(sft_bin_prefix+"_y.bin"):
        build_pretrain_bin(corpus_path, tokenizer, pretrain_bin_path)
        build_sft_bin(sft_message_path, tokenizer, block_size, sft_bin_prefix)
    print("------encode ready------")

    trainer = TinyTrainer(
        pretrain_bin_path, 
        sft_bin_prefix,
        block_size,
        batch_size,
        num_epoch_pretrain,
        num_epoch_sft,
        iter_per_epoch,
        lr_pretrain,
        lr_sft,
        fixed_prompt1,
        fixed_prompt2,
        tokenizer
    )

    load_model_path = ""

    if start_from == "pretrain" or start_from == "tokenizer" or start_from == "encode" or not os.path.exists(pretrain_model_path):
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
