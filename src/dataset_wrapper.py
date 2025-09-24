import os
import json
import random
from typing import Optional, Dict, Any
from tqdm import trange, tqdm
from prompt_template import (
    CONFIQA_BASE_TEMPLATE,
    CONFIQA_LOGICRAG_TEMPLATE,
    FAITHEVAL_COUNTERFACTUAL_BASE_TEMPLATE,
    FAITHEVAL_COUNTERFACTUAL_LOGICRAG_TEMPLATE,
    QA_SYSTEM_PTOMPT,
    SQUAD_TEMPLATE,
    SQUAD_LOGICRAG_TEMPLATE,
    KRE_PROMPT,
    OPIN_PTOMPT
)
from datasets import load_dataset
from modules import (
    AutomicFactsDecomposer, 
    QueryStepDecomposer, 
    FactSearcher, 
    ConflictDetector,
)


class RAGDataset:
    def __init__(self):
        self.questions = []
        self.answers = []

        self.automatic_facts_decomposer = AutomicFactsDecomposer(model="gpt-4o-2024-11-20")
        self.query_step_decomposer = QueryStepDecomposer(model="gpt-4o-2024-11-20")
        self.facts_searcher = FactSearcher()

    def __len__(self):
        return len(self.questions)
    
    def get_base_prompts_and_answers(self):
        raise NotImplementedError

    def get_logic_prompts_and_answers(self, cache_dir="cache", conflict_detector: Optional[ConflictDetector]=None):
        raise NotImplementedError
    
    def get_kre_prompts_and_answers(self):
        raise NotImplementedError
    
    def get_opin_prompts_and_answers(self):
        raise NotImplementedError

    def process_context(self, question, context):
        facts = self.automatic_facts_decomposer.generate_automic_facts(context)
        steps = self.query_step_decomposer.generate_query_steps(question)
        relevant_fact_indices = []
        for step in steps:
            relevant_fact_indices += self.facts_searcher.search_facts(step, facts)
        relevant_fact_indices = list(set(relevant_fact_indices))
        return facts, steps, relevant_fact_indices


class ConFiQA(RAGDataset):
    def __init__(self, data_path: str="data/processed/ConFiQA", subset: str="MC", split="test"):
        super().__init__()

        self.subset = subset
        self.split = split
        file_path = os.path.join(data_path, f"{subset.lower()}-{split}.json")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            self.origin_answers = []
            self.cf_answers = []
            self.conflicting_context = []
            self.consistent_context = []
            for item in data:
                self.questions.append(item["question"])
                self.origin_answers.append([item["orig_answer"]] + item["orig_alias"])
                self.cf_answers.append([item["cf_answer"]] + item["cf_alias"])
                self.conflicting_context.append(item["cf_context"])
                self.consistent_context.append(item["orig_context"])

    def __getitem__(self, index):
        # Return a single data point
        return {
            "question": self.questions[index],
            "origin_answer": self.origin_answers[index],
            "cf_answer": self.cf_answers[index],
            "conflicting_context": self.conflicting_context[index],
            "consistent_context": self.consistent_context[index]
        }

    def get_base_prompts_and_answers(self, with_context=True):
        prompts, answers = [], []
        for i in range(len(self.questions)):
            prompt = CONFIQA_BASE_TEMPLATE.format(
                context=self.conflicting_context[i] if with_context else "",
                question=self.questions[i],
            )
            messages = [
                {"role": "system", "content": QA_SYSTEM_PTOMPT},
                {"role": "user", "content": prompt},
            ]
            prompts.append(messages)
            answers.append(self.cf_answers[i])
        return prompts, answers

    def get_logic_prompts_and_answers(self, cache_dir="data/processed/ConFiQA", conflict_detector: Optional[ConflictDetector]=None):
        cache_file = os.path.join(cache_dir, f"{self.subset.lower()}-context-processed-{self.split}.jsonl")
        if not os.path.exists(cache_file):
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as cache_file_fp:
                for i in trange(len(self.questions)):
                    try:
                        facts, steps, relevant_fact_indices = self.process_context(
                            self.questions[i],
                            self.conflicting_context[i]
                        )
                    except Exception as e:
                        print(f"Error processing question {i}: {e}")
                        continue
                    
                    cache_file_fp.write(json.dumps({
                        "question": self.questions[i],
                        "context": self.conflicting_context[i],
                        "facts": facts,
                        "relevant_fact_indices": relevant_fact_indices,
                        "steps": steps,
                        "answer": self.cf_answers[i]
                    }, ensure_ascii=False) + "\n")
                    cache_file_fp.flush()

        prompts, answers = [], []
        with open(cache_file, "r", encoding="utf-8") as cache_file_fp:
            for line in tqdm(cache_file_fp.readlines(), desc="Detecting conflicts"):
                data = json.loads(line)

                if conflict_detector is not None:
                    conflict_fact_indices = conflict_detector.detect_conflicts(data["facts"], data["relevant_fact_indices"])
                    for i in conflict_fact_indices:
                        data["facts"][i] = f"<conflict> {data["facts"][i]} </conflict>"
                
                prompt = CONFIQA_LOGICRAG_TEMPLATE.format(
                    context="\n- " + "\n- ".join(data["facts"]),
                    question=data["question"],
                )
                messages = [
                    {"role": "system", "content": QA_SYSTEM_PTOMPT},
                    {"role": "user", "content": prompt},
                ]
                prompts.append(messages)
                answers.append(data["answer"])
        return prompts, answers

    def get_kre_prompts_and_answers(self):
        prompts, answers = [], []
        for i in range(len(self.questions)):
            prompt = KRE_PROMPT.format(
                context=self.consistent_context[i],
                question=self.questions[i],
                options=""
            )
            messages = [
                {"role": "system", "content": QA_SYSTEM_PTOMPT},
                {"role": "user", "content": prompt},
            ]
            prompts.append(messages)
            answers.append(self.origin_answers[i])
        return prompts, answers
    
    def get_opin_prompts_and_answers(self):
        prompts, answers = [], []
        for i in range(len(self.questions)):
            prompt = OPIN_PTOMPT.format(
                context=self.consistent_context[i],
                question=self.questions[i],
                options=""
            )
            messages = [
                {"role": "system", "content": QA_SYSTEM_PTOMPT},
                {"role": "user", "content": prompt},
            ]
            prompts.append(messages)
            answers.append(self.origin_answers[i])
        return prompts, answers


class FaithEvalCounterfactual(RAGDataset):
    def __init__(self, data_path:str="data/raw/FaithEval/counterfactual-v1.0"):
        super().__init__()

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File {data_path} does not exist.")
        
        dataset = load_dataset(data_path, split="test")
        self.contexts, self.options, self.answerKeys = [], [], []
        for item in dataset:
            item_dict: Dict[str, Any] = dict(item)
            self.questions.append(item_dict["question"])
            self.answers.append(item_dict["answer"])
            self.answerKeys.append(item_dict["answerKey"])
            self.options.append([
                label + ": " + text 
                for label, text in zip(item_dict["choices"]["label"], item_dict["choices"]["text"])
            ])
            self.contexts.append(item_dict["context"])
        
    def __getitem__(self, index):
        return {
            "question": self.questions[index],
            "answer": self.answers[index],
            "answerKey": self.answerKeys[index],
            "options": "\n".join(self.options[index]),
            "context": self.contexts[index]
        }
    
    def get_base_prompts_and_answers(self, with_context=True):
        prompts, answers = [], []
        for i in range(len(self.questions)):
            prompt = FAITHEVAL_COUNTERFACTUAL_BASE_TEMPLATE.format(
                context=self.contexts[i] if with_context else "",
                question=self.questions[i],
                options="\n".join(self.options[i])
            )
            messages = [
                {"role": "system", "content": QA_SYSTEM_PTOMPT},
                {"role": "user", "content": prompt},
            ]
            prompts.append(messages)
            answers.append(self.answers[i])
        return prompts, answers

    def get_logic_prompts_and_answers(self, cache_dir="data/processed/FaithEval", conflict_detector: Optional[ConflictDetector]=None):
        cache_file = os.path.join(cache_dir, "counterfactual-tmp.jsonl")
        if not os.path.exists(cache_file):
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as cache_file_fp:
                for i in trange(len(self.questions)):
                    try:
                        facts, steps, relevant_fact_indices = self.process_context(
                            self.questions[i],
                            self.contexts[i]
                        )
                    except Exception as e:
                        print(f"Error processing question {i}: {e}")
                        continue

                    cache_file_fp.write(json.dumps({
                        "question": self.questions[i],
                        "context": self.contexts[i],
                        "facts": facts,
                        "relevant_fact_indices": relevant_fact_indices,
                        "steps": steps,
                        "options": self.options[i],
                        "answer": self.answers[i]
                    }, ensure_ascii=False) + "\n")
                    cache_file_fp.flush()
        
        prompts, answers = [], []
        with open(cache_file, "r", encoding="utf-8") as cache_file_fp:
            for line in tqdm(cache_file_fp.readlines(), desc="Detecting conflicts"):
                data = json.loads(line)

                if conflict_detector is not None:
                    conflict_fact_indices = conflict_detector.detect_conflicts(data["facts"], data["relevant_fact_indices"])
                    for i in conflict_fact_indices:
                        data["facts"][i] = f"<conflict> {data["facts"][i]} </conflict>"
            
                prompt = FAITHEVAL_COUNTERFACTUAL_LOGICRAG_TEMPLATE.format(
                    context="\n- " + "\n- ".join(data["facts"]),
                    question=data["question"],
                    options="\n".join(data["options"])
                )
                messages = [
                    {"role": "system", "content": QA_SYSTEM_PTOMPT},
                    {"role": "user", "content": prompt},
                ]
                prompts.append(messages)
                answers.append(data["answer"])
        return prompts, answers

    def get_kre_prompts_and_answers(self):
        prompts, answers = [], []
        for i in range(len(self.questions)):
            prompt = KRE_PROMPT.format(
                context=self.contexts[i],
                question=self.questions[i],
                options="\n".join(self.options[i])
            )
            messages = [
                {"role": "system", "content": QA_SYSTEM_PTOMPT},
                {"role": "user", "content": prompt},
            ]
            prompts.append(messages)
            answers.append(self.answers[i])
        return prompts, answers
    
    def get_opin_prompts_and_answers(self):
        prompts, answers = [], []
        for i in range(len(self.questions)):
            prompt = OPIN_PTOMPT.format(
                context=self.contexts[i],
                question=self.questions[i],
                options=""
            )
            messages = [
                {"role": "system", "content": QA_SYSTEM_PTOMPT},
                {"role": "user", "content": prompt},
            ]
            prompts.append(messages)
            answers.append(self.answers[i])
        return prompts, answers


class SquadDataset(RAGDataset):
    def __init__(self, data_path:str="data/raw/squad", split="test", samples=3000):
        super().__init__()

        self.split = split
        data_path = os.path.join(data_path, f"{split}.jsonl")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File {data_path} does not exist.")
            
        self.questions, self.answers, self.contexts = [], [], []
        with open(data_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            if split == "train":
                random.seed(42)
                lines = random.choices(lines, k=samples)
            for line in lines:
                item = json.loads(line)
                self.questions.append(item["question"])
                self.answers.append(item["answers"])
                self.contexts.append(item["context"])
    
    def get_base_prompts_and_answers(self, with_context=True):
        prompts, answers = [], []
        for i in range(len(self.questions)):
            prompt = SQUAD_TEMPLATE.format(
                context=self.contexts[i] if with_context else "",
                question=self.questions[i],
            )
            messages = [
                {"role": "system", "content": QA_SYSTEM_PTOMPT},
                {"role": "user", "content": prompt},
            ]
            prompts.append(messages)
            answers.append(self.answers[i])
        return prompts, answers

    def get_logic_prompts_and_answers(self, cache_dir="data/processed/squad", conflict_detector: Optional[ConflictDetector]=None):
        cache_file = os.path.join(cache_dir, f"{self.split}.jsonl")
        if not os.path.exists(cache_file):
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as cache_file_fp:
                for i in trange(len(self.questions)):
                    try:
                        facts, steps, relevant_fact_indices = self.process_context(
                            self.questions[i],
                            self.contexts[i]
                        )
                    except Exception as e:
                        print(f"Error processing question {i}: {e}")
                        continue

                    cache_file_fp.write(json.dumps({
                        "question": self.questions[i],
                        "context": self.contexts[i],
                        "facts": facts,
                        "relevant_fact_indices": relevant_fact_indices,
                        "steps": steps,
                        "answer": self.answers[i]
                    }, ensure_ascii=False) + "\n")
                    cache_file_fp.flush()
        
        prompts, answers = [], []
        with open(cache_file, "r", encoding="utf-8") as cache_file_fp:
            for line in tqdm(cache_file_fp.readlines(), desc="Detecting conflicts"):
                data = json.loads(line)

                if conflict_detector is not None:
                    conflict_fact_indices = conflict_detector.detect_conflicts(data["facts"], data["relevant_fact_indices"])
                    for i in conflict_fact_indices:
                        data["facts"][i] = f"<conflict> {data["facts"][i]} </conflict>"
                
                prompt = SQUAD_LOGICRAG_TEMPLATE.format(
                    context="\n- " + "\n- ".join(data["facts"]),
                    question=data["question"],
                )
                messages = [
                    {"role": "system", "content": QA_SYSTEM_PTOMPT},
                    {"role": "user", "content": prompt},
                ]
                prompts.append(messages)
                answers.append(data["answer"])
        return prompts, answers

    def get_kre_prompts_and_answers(self):
        prompts, answers = [], []
        for i in range(len(self.questions)):
            prompt = KRE_PROMPT.format(
                context=self.contexts[i],
                question=self.questions[i],
                options=""
            )
            messages = [
                {"role": "system", "content": QA_SYSTEM_PTOMPT},
                {"role": "user", "content": prompt},
            ]
            prompts.append(messages)
            answers.append(self.answers[i])
        return prompts, answers
    
    def get_opin_prompts_and_answers(self):
        prompts, answers = [], []
        for i in range(len(self.questions)):
            prompt = OPIN_PTOMPT.format(
                context=self.contexts[i],
                question=self.questions[i],
                options=""
            )
            messages = [
                {"role": "system", "content": QA_SYSTEM_PTOMPT},
                {"role": "user", "content": prompt},
            ]
            prompts.append(messages)
            answers.append(self.answers[i])
        return prompts, answers

if __name__ == "__main__":
    dataset = FaithEvalCounterfactual(data_path="data/raw/FaithEval/counterfactual-v1.0")
    # dataset = ConFiQA(data_path="data/processed/ConFiQA", subset="MC")
    # dataset = SquadDataset(data_path="data/raw/squad", split="train", samples=1000)

    print("Dataset length:", len(dataset))
    # prompts, answers = dataset.get_base_prompts_and_answers()
    prompts, answers = dataset.get_logic_prompts_and_answers()
    print("Finished")
