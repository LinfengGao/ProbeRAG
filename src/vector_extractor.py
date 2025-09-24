import torch
from tqdm import tqdm
from accelerate import Accelerator
from utils import load_model_and_tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer


class Extrator:
    def __init__(
            self, 
            accelerator: Accelerator, 
            model: AutoModelForCausalLM, 
            tokenizer: AutoTokenizer
        ):
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer


class HiddenStateExtractor(Extrator):
    def get_hidden_states(self, prompts, batch_size=8, show_progress=True):
        batched_prompts = [
            prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)
        ]
        batched_hidden_states = []
        for batched_prompt in tqdm(batched_prompts, desc="Extracting hidden states", disable=not show_progress):
            inputs = self.tokenizer(batched_prompt, return_tensors="pt", padding=True).to(self.accelerator.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )

            hidden_states = outputs.hidden_states[0][-1]  # 最后一层的hidden states: (batch_size, seq_len, hidden_size)
            last_token_hidden_states = hidden_states[:, -1, :]  # last_token_hidden_states: (batch_size, hidden_size)

            batched_hidden_states.append(last_token_hidden_states.cpu())
    
        return torch.cat(batched_hidden_states, dim=0)
    
    def get_hidden_state_size(self):
        return self.model.config.hidden_size


class LogitsExtractor(Extrator):
    def get_logits_softmax(self, prompts, batch_size=8, show_progress=True):
        batched_prompts = [
            prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)
        ]
        batched_logits_softmax = []
        for batched_prompt in tqdm(batched_prompts, desc="Extracting logits", disable=not show_progress):
            inputs = self.tokenizer(batched_prompt, return_tensors="pt", padding=True).to(self.accelerator.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,  # 只生成一个新token
                    output_scores=True,  # 输出scores (logits)
                    return_dict_in_generate=True
                )
        
            last_token_logits = outputs.scores[-1]  # (batch_size, vocab_size)
            last_token_logits_softmax = torch.softmax(last_token_logits, dim=-1)  # (batch_size, vocab_size)
            batched_logits_softmax.append(last_token_logits_softmax)
        
        return torch.cat(batched_logits_softmax, dim=0)


class AttentionWeightExtractor(Extrator):
    def get_last_token_attention_weights(self, prompts, batch_size=8, show_progress=True):
        batched_prompts = [
            prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)
        ]
        batched_attention_weights, batched_tokens = [], []
        for batched_prompt in tqdm(batched_prompts, desc="Extracting Attention Weights", disable=not show_progress):
            inputs = self.tokenizer(batched_prompt, return_tensors="pt", padding=True).to(self.accelerator.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    output_attentions=True,
                    return_dict_in_generate=True,
                )

            first_layer_attentions = outputs.attentions[0][0]  # 第一层的注意力权重
            last_token_attentions = first_layer_attentions[:, :, -1, :]  # 最后一个token的注意力权重 (batch_size, num_heads, seq_len)
            for i in range(last_token_attentions.shape[0]):
                batched_attention_weights.append(last_token_attentions[i].squeeze(0).cpu())  # (num_heads, seq_len)

            for token_ids in inputs["input_ids"]:
                tokens = self.tokenizer.convert_ids_to_tokens(token_ids.tolist())
                batched_tokens.append(tokens)

        return batched_attention_weights, batched_tokens
    
    def get_attention_weight_matrix(self, prompts, batch_size=8, show_progress=True):
        batched_prompts = [
            prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)
        ]
        batched_attention_weights, batched_tokens = [], []
        for batched_prompt in tqdm(batched_prompts, desc="Extracting Attention Weights", disable=not show_progress):
            inputs = self.tokenizer(batched_prompt, return_tensors="pt", padding=True).to(self.accelerator.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    output_attentions=True,
                    return_dict_in_generate=True,
                )

            last_layer_attentions = outputs.attentions[-1][0]  # 最后一层的注意力权重
            # attention_matrix = last_layer_attentions[:, head_idx, :, :]  # 获取第head_idx个头的注意力权重 (batch_size, seq_len, seq_len)
            attention_matrix = sum(
                last_layer_attentions[:, i, :, :] for i in range(last_layer_attentions.shape[1])
            )/ last_layer_attentions.shape[1]  # 平均所有头的注意力权重 (batch_size, seq_len, seq_len)
            for i in range(attention_matrix.shape[0]):
                batched_attention_weights.append(attention_matrix[i].squeeze(0).cpu())  # (seq_len, seq_len)

            for token_ids in inputs["input_ids"]:
                tokens = self.tokenizer.convert_ids_to_tokens(token_ids.tolist())
                batched_tokens.append(tokens)
        
        return batched_attention_weights, batched_tokens


class PerplexityExtractor(Extrator):
    def get_perplexity(self, prompts, show_progress=True):
        perplexities = []
        for prompt in tqdm(prompts, desc="Extracting Perplexity", disable=not show_progress):
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.accelerator.device)

            # 计算负对数似然损失
            with torch.no_grad():
                outputs = self.model(inputs["input_ids"], labels=inputs["input_ids"])
                loss = outputs.loss
            
            # 计算困惑度
            perplexity = torch.exp(loss)
            perplexities.append(perplexity.item())

        return perplexities


if __name__ == "__main__":
    model_path = "/home/glf/data/models/Mistral-7B-Instruct-v0.3"

    accelerator, model, tokenizer = load_model_and_tokenizer(model_path)

    prompts = [
        "The earth's satellite is the moon.",
        "The earth's satellite is Mars.",
        "The earth's satellite is the sun."
    ]

    # Hidden State Extractor Example
    # extractor = HiddenStateExtractor(accelerator, model, tokenizer)
    # print(extractor.get_hidden_state_size())  # 打印hidden size
    # last_token_hidden_states = extractor.get_hidden_states(prompts, batch_size=1)
    # print(last_token_hidden_states.shape)  # (batch_size, hidden_size)

    # Logits Extractor Example
    # extractor = LogitsExtractor(accelerator, model, tokenizer)
    # last_token_logits = extractor.get_logits_softmax(prompts, batch_size=1)
    # print(last_token_logits.shape)  # (batch_size, vocab_size)

    # Attention Weight Extractor Example
    # extrator = AttentionWeightExtractor(accelerator, model, tokenizer)
    # attention_weights, tokens = extrator.get_last_token_attention_weights(prompts, batch_size=1)
    # attention_matrix, tokens = extrator.get_attention_weight_matrix(prompts, head_idx=0, batch_size=3)
    # print(attention_matrix[0].shape)  # (seq_len, seq_len)
    # print(attention_matrix[1].shape)  # (seq_len, seq_len)
    # print(tokens[0])  # 打印第一个输入的tokens
    # print(tokens[1])  # 打印第二个输入的tokens

    # Perplexity Extractor Example
    extractor = PerplexityExtractor(accelerator, model, tokenizer)
    perplexities = extractor.get_perplexity(prompts)
    print(perplexities)  # 打印每个输入的困惑度