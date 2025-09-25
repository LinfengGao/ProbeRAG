import sys
sys.path.append("src")

import os
import json
import torch
from classifier import train_model
from modules import ConflictDetector
from utils import load_model_and_tokenizer
from vector_extractor import HiddenStateExtractor


def process_mquake_claims(data_path, output_dir):
    with open(data_path, "r") as f:
        data = json.load(f)
    
    knowledge_claims = []
    for case in data:
        for triplet in case["requested_rewrite"]:
            real_knowledge = triplet["prompt"].format(triplet["subject"]) + " " + triplet["target_true"]["str"]
            contradict_knowledge = triplet["prompt"].format(triplet["subject"]) + " " + triplet["target_new"]["str"]
            knowledge_claims.append({"knowledge": real_knowledge, "conflict": False})
            knowledge_claims.append({"knowledge": contradict_knowledge, "conflict": True})
    
    os.makedirs(output_dir, exist_ok=True)
    # split train and test sets
    train_size = int(len(knowledge_claims) * 0.8)  # Use only the first 80% for training

    train_output_path = output_dir + "/mquake-train.jsonl"
    train_knowledge_claims = knowledge_claims[:train_size]
    with open(train_output_path, "w") as f:
        for claim in train_knowledge_claims:
            f.write(json.dumps(claim, ensure_ascii=False) + "\n")
    
    test_output_path = output_dir + "/mquake-test.jsonl"
    test_knowledge_claims = knowledge_claims[train_size:]
    with open(test_output_path, "w") as f:
        for claim in test_knowledge_claims:
            f.write(json.dumps(claim, ensure_ascii=False) + "\n")


def training(model_path, data_path, save_path, log_dir):
    accelerator, model, tokenizer = load_model_and_tokenizer(model_path)
    extractor = HiddenStateExtractor(accelerator, model, tokenizer)
    consistent_knowledges, conflict_knowledges = [], []
    with open(data_path, "r") as f:
        for line in f.readlines():
            data = json.loads(line)
            if data["conflict"] == True:
                conflict_knowledges.append(data["knowledge"])
            else:
                consistent_knowledges.append(data["knowledge"])
    
    consistent_vectors = extractor.get_hidden_states(consistent_knowledges)
    conflicting_vectors = extractor.get_hidden_states(conflict_knowledges)

    X = torch.cat([consistent_vectors, conflicting_vectors], dim=0).to(torch.float32)
    y = torch.cat([
        torch.zeros(consistent_vectors.shape[0]),
        torch.ones(conflicting_vectors.shape[0])
    ], dim=0).unsqueeze(-1).to(torch.float32)  # 一致标签为0，冲突标签为1

    X, y = X.to(torch.float32), y.to(torch.float32)
    trained_model = train_model(
        X, y,
        input_size=X.shape[1],
        hidden_size1=512,
        hidden_size2=256,
        output_size=1,
        num_epochs=100, 
        batch_size=32, 
        lr=0.001,
        report_to="tensorboard",
        log_dir=log_dir,
        device="cuda"
    )
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(trained_model.state_dict(), save_path)
    print(f"Classifier model saved to {save_path}")


def evaluating(model_path, classifier_path, data_path):
    accelerator, model, tokenizer = load_model_and_tokenizer(model_path)
    detector = ConflictDetector(accelerator, model, tokenizer, classifier_path)

    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
        knowledge_claims = [item["knowledge"] for item in data]
        labels = [item["conflict"] for item in data]

    conflict_indices = detector.detect_conflicts(facts=knowledge_claims, relevant_fact_indices=range(len(knowledge_claims)))
    conflict_indices = set(conflict_indices)

    accuracy = 0.0
    for i, label in enumerate(labels):
        if label:
            accuracy += (i in conflict_indices)
        else:
            accuracy += (i not in conflict_indices)

    accuracy = accuracy / len(knowledge_claims)
    print(f"Evaluation accuracy: {accuracy:.4f}")


if __name__ == "__main__":

    model_path = "models/Qwen3-8B"
    processed_data_path = "data/processed/classifier"
    classifier_path = "checkpoints/classifier/qwen3-8b-0623.pt"
    log_dir = "logs/classifier_training_logs/0623/qwen3-8b"

    # process_mquake_claims("data/raw/MQuAKE/datasets/MQuAKE-CF-3k-v2.json", processed_data_path)

    training(
        model_path=model_path, 
        data_path=processed_data_path + "/mquake-train.jsonl",
        save_path=classifier_path,
        log_dir=log_dir,
    )

    evaluating(
        model_path=model_path,
        classifier_path=classifier_path,
        data_path=processed_data_path + "/mquake-test.jsonl",
    )