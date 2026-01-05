import sys
sys.path.append("src")

import re
import os
import json
import torch
import string
from accelerate import Accelerator
from utils import load_model_and_tokenizer
from modules import ConflictDetector
from dataset_wrapper import RAGDataset, ConFiQA, FaithEvalCounterfactual, SquadDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines import pipeline
from transformers.models.qwen3 import Qwen3ForCausalLM


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str):
    from collections import Counter

    predict_tokens, ground_truth_tokens = prediction.split(), ground_truth.split()
    common = Counter(predict_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(predict_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


negation_words = [
    "no", "not", "never", "none", "cannot", "nobody", "nothing", "nowhere", 
    "neither", "nor", "without", "hardly"
]


def exact_match_score(prediction, ground_truth, is_cf):
    contains_negation = any(word in prediction.split() for word in negation_words)
    return (not contains_negation if is_cf else True) and (normalize_answer(prediction) == normalize_answer(ground_truth))


def recall_score(prediction, ground_truth, is_cf):
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    
    contains_negation = any(word in prediction.split() for word in negation_words)
    
    return (ground_truth in prediction) and (not contains_negation if is_cf else True)


def get_score(preds, golds, origs):
    em, gold_recall, orig_recall = 0, 0, 0
    for pred, gold, orig in zip(preds, golds, origs):
        if isinstance(gold, list):
            _em, _recall = 0, 0
            for g in gold:
                _em = max(exact_match_score(pred, g, True), _em)
                _recall = max(recall_score(pred, g, True), _recall)
        else:
            _em = exact_match_score(pred, gold, True)
            _recall = recall_score(pred, gold, True)
        if isinstance(orig, list):
            _recall_orig = 0
            for o in orig:
                _recall_orig = max(recall_score(pred, o, False), _recall_orig)
        else:
            _recall_orig = recall_score(pred, orig, False)
        em += _em
        gold_recall += _recall and not _recall_orig
        orig_recall +=  _recall_orig
        
    em = em * 100 / (len(preds) + 1e-5)
    gold_recall = gold_recall * 100 / (len(preds) + 1e-5)
    orig_recall = orig_recall * 100 / (len(preds) + 1e-5)
    return em, gold_recall, orig_recall


class QAEvaluator:
    def __init__(self, accelerator: Accelerator, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, classifier_path: str):
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer

        self.conflict_detector = ConflictDetector(
            accelerator=self.accelerator,
            model=self.model,
            tokenizer=self.tokenizer,
            classifier_path=classifier_path,
        )

        self.pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )
    
    def evaluate(self, dataset:RAGDataset, output_file:str="output/qa_exp/file.json", method="ProbeRAG"):
        if method == "ProbeRAG":
            prompts, answers = dataset.get_logic_prompts_and_answers(
                conflict_detector=self.conflict_detector
            )
        elif method == "WO-Context":
            prompts, answers = dataset.get_base_prompts_and_answers(with_context=False)
        elif method == "KRE":
            prompts, answers = dataset.get_kre_prompts_and_answers()
        elif method == "OPIN":
            prompts, answers = dataset.get_opin_prompts_and_answers()
        else:
            prompts, answers = dataset.get_base_prompts_and_answers()

        if type(self.pipeline.model) == Qwen3ForCausalLM:
            prompts = [
                self.tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                    continue_final_message=False,
                    enable_thinking=False,
                ) for prompt in prompts
            ]
        
        responses = self.pipeline(
            prompts,
            max_new_tokens=200,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            return_full_text=False,
            num_workers=1,
            batch_size=4,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        predicts = [
            normalize_answer(response[0]["generated_text"])
            for response in responses
        ]
        
        tmp = []
        for answer in answers:
            if type(answer) == list:
                answer = list(set([normalize_answer(ans) for ans in answer] + answer))
            else:
                answer = list(set([normalize_answer(answer), answer]))
            tmp.append(answer)
        answers = tmp

        case_f1, case_em = [], []
        for i in range(len(predicts)):
            predict, answer_list = predicts[i], answers[i]
            f1_results = [f1_score(predict, answer) for answer in answer_list]
            em_results = [exact_match_score(predict, answer, is_cf=True) for answer in answer_list]

            case_f1.append(max(f1_results))
            case_em.append(max(em_results))

        f1 = round(sum(case_f1) / len(predicts), 4)
        em = round(sum(case_em) / len(predicts), 4)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        output_file = open(output_file, "w", encoding="utf-8")
        result = [{"total_f1": f1, "total_em": em}]
        for i in range(len(predicts)):
            result.append({
                "question": prompts[i],
                "answer": answers[i],
                "predict": predicts[i],
                "f1": case_f1[i],
                "em": case_em[i]
            })
        json.dump(result, output_file, ensure_ascii=False, indent=4)
        print(f"F1 Score: {f1}, Exact Match: {em}")


if __name__ == "__main__":
    date = "current_time"

    model_dict = {
        "llama3.1-8b": "Meta-Llama-3.1-8B-Instruct",
        "llama2-7b": "llama-2-7b-chat-hf",
        "qwen2.5-7b": "Qwen2.5-7B-Instruct",
        "qwen3-8b": "Qwen3-8B",
        "mistral-7b": "Mistral-7B-Instruct-v0.3",
    }
    dataset_dict = {
        "faitheval/counterfactual": FaithEvalCounterfactual(data_path="data/raw/FaithEval/counterfactual-v1.0"),
        "confiqa/mc": ConFiQA(data_path="data/processed/ConFiQA", subset="MC", split="test"),
        "confiqa/mr": ConFiQA(data_path="data/processed/ConFiQA", subset="MR", split="test"),
        "confiqa/qa": ConFiQA(data_path="data/processed/ConFiQA", subset="QA", split="test"),
        "squad": SquadDataset(data_path="data/raw/squad"),
    }
    for method in [
        # "WO-Context",
        # "Full-Context",
        # "KRE",
        # "OPIN"
        # "CAD",
        # "ContextDPO",
        # "CANOE",
        # "ParamMute"
        "ProbeRAG",
    ]:
        for model_short_cut, model_name in model_dict.items():
            for dataset_name, dataset in dataset_dict.items():
                output_file = f"experiments/qa_exp/{dataset_name}/{method}-{model_short_cut}-{date}.json"
                if os.path.exists(output_file) and "tmp" not in date:
                    print(f"Already evaluated {method} {model_short_cut} on {dataset_name}")
                    continue
                if method == "ContextDPO":
                    model_path = f"checkpoints/Context-DPO-adapter/Context-Faithful-{model_name}/final"
                elif method == "ProbeRAG":
                    model_path = f"checkpoints/conflict-aware-models/conflict-aware-{model_short_cut}/final"
                elif method == "ParamMute":
                    model_path = f"checkpoints/ParamMute/{model_short_cut}"
                elif method == "CANOE":
                    model_path = f"checkpoints/CANOE/CANOE-{model_name}"
                else:
                    model_path = f"/home/glf/data/models/{model_name}"
                
                print(model_path)
                accelerator, model, tokenizer = load_model_and_tokenizer(
                    model_name_or_path=model_path,
                )
                evaluator = QAEvaluator(accelerator, model, tokenizer, classifier_path=f"checkpoints/classifier/{model_short_cut}.pt")
                evaluator.evaluate(
                    dataset,
                    method=method,
                    output_file=output_file,
                )
                del accelerator, model, tokenizer
                torch.cuda.empty_cache()
                print(f"Finished evaluating {method} {model_short_cut} on {dataset_name}")
