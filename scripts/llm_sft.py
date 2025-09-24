"""
torchrun --nnodes 1 --nproc_per_node 1 --master_addr 127.0.0.1 --master_port 29501 scripts/llm_sft.py
"""
import sys
sys.path.append("src")

import os
import torch
import yaml
from munch import munchify
from peft import LoraConfig, PeftModelForCausalLM, get_peft_model
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)


# read arguments
with open("scripts/llm-sft-config.yaml", "r") as stream:
    args = yaml.safe_load(stream)
    args = munchify(args)


def match_sequence(arr, seq):
    starts, ends = [], []
    for i in range(len(arr) - len(seq) + 1):
        if arr[i:i+len(seq)] == seq:
            starts.append(i)
            ends.append(i + len(seq) - 1)
    return starts, ends


class InstructDataset(Dataset):
    def __init__(self, data, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.data = [self.preprocess(item) for item in data]

    def tokenize(self, prompt, add_eos_token=True):
        result = self.tokenizer(
            prompt,
            truncation=self.config.truncation,
            max_length=self.config.max_length,
            padding=self.config.padding,
            return_tensors=None,
        )

        # Llama tokenizer不会在末尾自动添加eos，这里手动添加。如果长度小于max_length，则添加eos。
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.config.max_length
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        result.pop("token_type_ids", None)  # remove key token_type_ids
        return result

    def preprocess(self, data_point):
        full_prompt = data_point["input"] + data_point["output"]
        full_prompt = self.tokenize(full_prompt)
        input_prompt = data_point["input"]
        input_prompt = self.tokenize(input_prompt, add_eos_token=False)
        input_len = len(input_prompt["input_ids"])
        full_prompt["labels"] = [-100] * input_len + full_prompt["input_ids"][input_len:]

        # attention_mask_conflict记录哪些token在<conflict>和</conflict>之间。这些位置为1，其余为0。

        start_token_ids = self.tokenizer.encode(" <conflict>", add_special_tokens=False)
        end_token_ids = self.tokenizer.encode(" </conflict>\n", add_special_tokens=False)

        conflict_starts, _ = match_sequence(full_prompt["input_ids"], start_token_ids)
        _, conflict_ends = match_sequence(full_prompt["input_ids"], end_token_ids)
        conflict_intervals = list(zip(conflict_starts, conflict_ends))
        attention_mask_conflict = [0] * len(full_prompt["input_ids"])
        for start, end in conflict_intervals:
            attention_mask_conflict[start: end+1] = [1] * (end - start + 1)
        full_prompt["attention_mask_conflict"] = attention_mask_conflict
        return full_prompt

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)


def load_data(data_file):
    import json
    data = []
    with open(data_file, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

tokenizer = AutoTokenizer.from_pretrained(
    args.model_config.base_model_name_or_path,
    trust_remote_code=True,
    # use_fast=False,
    padding_side="left",
)
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})

data = load_data(args.data_config.train_data_file)
dataset = InstructDataset(data, tokenizer=tokenizer, config=args.tokenizer_config)
train_dataset, eval_dataset = random_split(dataset, [args.data_config.train_set_ratio, 1 - args.data_config.train_set_ratio])

# initialize distributed training config
local_rank = int(os.environ.get("LOCAL_RANK", -1))
world_size = int(os.environ.get("WORLD_SIZE", -1))
print(f"{local_rank=}, {world_size=}")

if local_rank == -1 and world_size == -1:
    device_map = "auto"
elif torch.cuda.is_available():
    torch.distributed.init_process_group("nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device_map = {"": local_rank}
else:
    torch.distributed.init_process_group("gloo", rank=local_rank, world_size=world_size)
    device_map = {"": torch.device("cpu")}

model = AutoModelForCausalLM.from_pretrained(
    args.model_config.base_model_name_or_path,
    load_in_8bit=False,
    device_map=device_map,
    attn_implementation="eager",
)

print('Load model successfully!')

if args.fp16 and torch.cuda.is_available():
    model.half()

if args.model_config.lora_checkpoint_path is not None:
    model = PeftModelForCausalLM.from_pretrained(model, args.model_config.lora_checkpoint_path, is_trainable=True)
else:
    config = LoraConfig(
        r=args.lora_config.r,
        lora_alpha=args.lora_config.alpha,
        target_modules=args.lora_config.target_modules,
        lora_dropout=args.lora_config.dropout,
        bias=args.lora_config.bias,
        task_type=args.lora_config.task_type,
    )
    model = get_peft_model(model, config)

training_args = TrainingArguments(
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    warmup_ratio=args.warmup_ratio,
    num_train_epochs=args.num_train_epochs,
    learning_rate=args.learning_rate,
    fp16=args.fp16,
    logging_steps=args.logging_steps,
    eval_strategy="no",
    save_strategy="steps",
    eval_steps=args.eval_steps,
    save_steps=args.save_steps,
    output_dir=args.output_dir,
    save_total_limit=3,
    load_best_model_at_end=False,
    group_by_length=args.group_by_length,
    debug="underflow_overflow",
    report_to="tensorboard"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)


class AttentionTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        计算每个样本attention_mask_conflict中为1的token的平均注意力分数,
        将这个平均注意力分数乘以一个lambda系数, 并从原始损失中加上这个值。
        """
        attention_mask_conflict = inputs.pop("attention_mask_conflict")
        outputs = model(**inputs, output_attentions=True)
        base_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        attention_matrix = outputs.attentions[-1]

        attention_to_conflict = torch.einsum("bhst, bt -> bhs", attention_matrix, attention_mask_conflict.float())
        attention_loss = ((attention_to_conflict > 0).float() - attention_to_conflict).sum()
        attention_loss /= (attention_to_conflict > 0).float().sum() + 1e-8  # Avoid division by zero
        
        lambda_ = 0.1
        total_loss = (1 - lambda_) * base_loss + lambda_ * attention_loss
        return (total_loss, outputs) if return_outputs else total_loss


trainer = AttentionTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    data_collator=data_collator,
)

trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
torch.distributed.destroy_process_group()