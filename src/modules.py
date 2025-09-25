import os
import re
import time
import torch
import dotenv
from typing import List
from openai import OpenAI
from accelerate import Accelerator
from classifier import Classifier
from vector_extractor import HiddenStateExtractor
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from prompt_template import (
    FACT_DECOMPOSE_PROMPT,
    QUERY_DECOMPOSE_PROMPT,
)


class AutomicFactsDecomposer:
    def __init__(self, model="gpt-3.5-turbo"):
        dotenv.load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        self.model = model

    def generate_automic_facts(self, context):
        """
        Generates atomic facts from the given context.
        """

        prompt = FACT_DECOMPOSE_PROMPT.format(text=context)
        max_retries = 10
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.9,
                )
                response = completion.choices[0].message.content.strip()
                time.sleep(1)  # To avoid hitting the rate limit
                break
            except Exception as e:
                max_retries -= 1
                if max_retries == 0:
                    raise Exception("Max retries exceeded")
                time.sleep(3)
        
        # Convert the response into a list of atomic facts
        facts = [fact.removeprefix("- ").strip() for fact in response.split('\n')]
        return facts


class QueryStepDecomposer:
    def __init__(self, model="gpt-3.5-turbo"):
        dotenv.load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        self.model = model

    def generate_query_steps(self, query):
        """
        Generates query steps.
        """
        # Here you would implement your logic to extract query steps
        prompt = QUERY_DECOMPOSE_PROMPT.format(query=query)
        max_retries = 10
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.9,
                )
                response = completion.choices[0].message.content.strip()
                time.sleep(1)
                break
            except:
                max_retries -= 1
                if max_retries == 0:
                    raise Exception("Max retries exceeded")
                time.sleep(3)
        
        # Remove the prefix "- " from each step
        steps = [step.removeprefix("- ").strip() for step in response.split('\n')]
        # Remove prefixes like "1.", "2.", etc.
        steps = [re.sub(r'^\d+\.\s*', '', step) for step in steps]
        # Remove any empty steps
        steps = [step for step in steps if step]
        
        return steps


class FactSearcher:
    def __init__(self, model="models/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)
        self.model.eval()
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

    def search_facts(self, query, facts, top_k=5):
        """
        Searches for the most relevant facts based on the query.
        """
        # Encode the query and facts
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        facts_embeddings = self.model.encode(facts, convert_to_tensor=True)

        # Calculate cosine similarity
        similarities = torch.nn.functional.cosine_similarity(query_embedding, facts_embeddings)

        # Get the top_k most similar facts
        most_similar_indices = torch.topk(similarities, min(similarities.size(0), top_k)).indices
        return most_similar_indices.tolist()


class ConflictDetector:
    def __init__(
            self,
            accelerator: Accelerator,
            model: AutoModelForCausalLM,
            tokenizer: AutoTokenizer, 
            classifier_path: str,
        ):
        self.hidden_state_extractor = HiddenStateExtractor(accelerator, model, tokenizer)
        self.classifier_model = Classifier(self.hidden_state_extractor.get_hidden_state_size())
        self.classifier_model.load_state_dict(torch.load(classifier_path))
        self.classifier_model.eval()
        self.classifier_model.to("cuda" if torch.cuda.is_available() else "cpu")

    def detect_conflicts(self, facts: List[str], relevant_fact_indices: List[int]):
        relevant_facts = [facts[i] for i in relevant_fact_indices]
        hidden_state = self.hidden_state_extractor.get_hidden_states(relevant_facts, show_progress=False)
        hidden_state = hidden_state.to("cuda" if torch.cuda.is_available() else "cpu").to(torch.float32)
        with torch.no_grad():
            probs = self.classifier_model(hidden_state)
            probs = probs.cpu().numpy()
        
        # Assuming a threshold of 0.5 for binary classification
        if probs.size == 1:
            conflict_indices = [relevant_fact_indices[0]] if probs > 0.5 else []
        else:
            conflict_indices = [relevant_fact_indices[i] for i, prob in enumerate(probs) if prob > 0.5]
        return conflict_indices


if __name__ == "__main__":
    facts_decomposer = AutomicFactsDecomposer()
    context = "Horizon Zero Dawn is a renowned video game development studio based in the Netherlands, known for creating immersive and visually stunning games. Founded in 2007, the studio has gained recognition for its unique blend of post-apocalyptic and sci-fi elements, combined with a deep narrative focus. With a talented team of developers and a passion for storytelling, Horizon Zero Dawn has established itself as a leader in the gaming industry. Netherlands, a constitutional monarchy located in Northwestern Europe, is ruled by Willem-Alexander of the Netherlands, the current head of state. With a population of approximately 17.5 million people, the country boasts a rich history, cultural heritage, and modern infrastructure. Known for its tolerant and inclusive society, the Netherlands is a popular destination for tourists and businesses alike. Willem-Alexander of the Netherlands, the King of the Netherlands, has been in power since 2013, and has worked to strengthen the country's economy and international relations. Under his leadership, the Netherlands has become a leader in the European Union and a model for democratic governance."
    facts = facts_decomposer.generate_automic_facts(context)
    print("Generated Atomic Facts:")
    for fact in facts:
        print(f"- {fact}")
    
    question = "Who is the head of state for the country of origin of Horizon Zero Dawn?"
    query_decomposer = QueryStepDecomposer()
    steps = query_decomposer.generate_query_steps(question)
    print("\nGenerated Query Steps:")
    for step in steps:
        print(f"- {step}")

    facts_searcher = FactSearcher(model="models/all-MiniLM-L6-v2")
    for step in steps:
        relevant_facts = facts_searcher.search_facts(step, facts, top_k=3)
        print(f"\nRelevant Facts for Step '{step}':")
        for fact in relevant_facts:
            print(f"- {fact}")

    conflict_detector = ConflictDetector(
        llm_model_path="models/Meta-Llama-3.1-8B-Instruct", 
        classifier_path="output/classifier/llama3-8b-classifier.pt"
    )
    facts_with_conflicts = conflict_detector.detect_conflicts(facts)