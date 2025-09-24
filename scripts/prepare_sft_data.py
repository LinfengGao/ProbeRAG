import sys
sys.path.append("src")

import os
import json
from dataset_wrapper import ConFiQA, SquadDataset
from modules import ConflictDetector
from utils import load_model_and_tokenizer
from sklearn.model_selection import train_test_split


def data_split(data_path, output_path, test_size=0.25):
    for name in ["MC", "MR", "QA"]:
        data = json.load(open(os.path.join(data_path, f"ConFiQA-{name}.json"), "r", encoding="utf-8"))
        train, test = train_test_split(data, test_size=test_size, random_state=42)

        json.dump(
            train,
            open(os.path.join(output_path, f"{name.lower()}-train.json"), "w", encoding="utf-8"),
            indent=4, 
            ensure_ascii=False
        )
        json.dump(
            test, 
            open(os.path.join(output_path, f"{name.lower()}-test.json"), "w", encoding="utf-8"), 
            indent=4, 
            ensure_ascii=False
        )


def build_sft_data(model_path, classifier_path, output_path):
    dataset = [
        SquadDataset(split="train", samples=1000),
        ConFiQA(subset="MC", split="train"),
        ConFiQA(subset="MR", split="train"),
        ConFiQA(subset="QA", split="train"),
    ]

    accelerator, model, tokenizer = load_model_and_tokenizer(model_path)
    conflict_detector = ConflictDetector(accelerator, model, tokenizer, classifier_path)

    total_train_data = []
    for item in dataset:
        prompts, answers = item.get_logic_prompts_and_answers(conflict_detector=conflict_detector)
        for prompt, answer in zip(prompts, answers):
            total_train_data.append({
                "input": tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False, 
                    add_generation_prompt=True,
                    continue_final_message=False,
                    enable_thinking=False,
                ),
                "output": answer[0] + tokenizer.eos_token,
            })
            # total_train_data.append({
            #     "input": prompt[1]["content"],
            #     "output": answer[0]
            # })

    with open(output_path, "w") as f:
        for line in total_train_data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raw_data_path = "Context-DPO/ConFiQA"
    processed_data_path = "data/processed/ConFiQA"

    model_path = "/home/glf/data/models/Qwen3-8B"
    classifier_path = "checkpoints/classifier/qwen3-8b-0623.pt"
    sft_data_path = "data/sft/sft_train_qwen3_8b+1000squad.jsonl"

    # data_split(data_path=raw_data_path, output_path=processed_data_path)
    build_sft_data(model_path, classifier_path, sft_data_path)