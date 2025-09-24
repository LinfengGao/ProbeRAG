QA_SYSTEM_PTOMPT = """\
You are an expert in retrieval question answering. Please respond with the exact answer only. Do not be verbose or provide extra information.
"""

QA_PTOMPT = """You are a helpful assistant. You will be given a context and a question. Your task is to answer the question based on the context provided.
Context: {context}
Question: {question}
Answer:
"""

QUESTION_STEPS_TEMPLATE = """\
Steps: {steps}
"""

PAY_ATTENTION_CONFLICT_TEMPLATE = """\
Please pay attention to the context wrapped by '<conflict>' and '</conflict>', which means that the information in this part is conflict with your internal knowledge.
You should strictly follow the information in this part and do not use your internal knowledge.
"""

FAITHEVAL_COUNTERFACTUAL_BASE_TEMPLATE = """\
Context: {context}
Question: {question}
Options:
{options}
Now based on the given context answer the question and select the best answer from the options.
Please provide the answer itself and do not provide the choice letter.
Please pay attention to the context "conflict" with your internal knowledge.
You should strictly be faithful to the context, follow the information in the context and do not use your internal knowledge.
Answer: 
"""

FAITHEVAL_COUNTERFACTUAL_LOGICRAG_TEMPLATE = """\
Now based on the given context answer the question and select the best answer from the options. Please provide the answer itself and do not provide the choice letter:
Please pay attention to the context wrapped by '<conflict>' and '</conflict>', which means that the information in this part is conflict with your internal knowledge.
You should strictly follow the information in this part and do not use your internal knowledge.
Question: {question}
Context: {context}
Options:
{options}
Answer: 
"""

FAITHEVAL_INCONSISTENT_BASE_TEMPLATE = """\
Now based on the given two contexts answer the question. If there is conflict information or multiple answers from the context, the answer should be 'conflict':
Context1: {context1}
Context2: {context2}
Question: {question}
Answer: 
"""

FAITHEVAL_INCONSISTENT_LOGICRAG_TEMPLATE = """\
Now based on the given two contexts answer the question. If there is conflict information or multiple answers from the context, the answer should be 'conflict':
Please pay attention to the context wrapped by '<conflict>' and '</conflict>', which means that the information in this part is conflict with your internal knowledge.
You should strictly follow the information in this part and do not use your internal knowledge.
Context1: {context1}
Context2: {context2}
Question: {question}
Again, if there is conflict information or multiple answers from the context, the answer should be 'conflict'. You do not need to provide the specific answer.
Answer: 
"""

FAITHEVAL_UNANSWERABLE_BASE_TEMPLATE = """\
Now based on the given context answer the question. If there is no information available from the context, the answer should be 'unknown':
Context: {context}
Question: {question}
Answer: 
"""

FAITHEVAL_UNANSWERABLE_LOGICRAG_TEMPLATE = """\
Now based on the given context answer the question. If there is no information available from the context, the answer should be 'unknown':
Please pay attention to the context wrapped by '<conflict>' and '</conflict>', which means that the information in this part is conflict with your internal knowledge.
You should strictly follow the information in this part and do not use your internal knowledge.
Context: {context}
Question: {question}
Again, if there is no information available from the context, the answer should be 'unknown'. You do not need to provide the specific answer.
Answer: 
"""

CONFLICTQA_POPQA_TEMPLATE = """\
Now based on the given context answer the question:
{optional_pay_attention}
Context: {context}
Question: {question}
{optional_steps}
Please provide the answer directly and do not be verbose. Only several words is engough.
Answer: 
"""

CONFLICTQA_STRATEGYQA_TEMPLATE = """\
Now based on the given context judge the question. Please answer "True" if the result is positive and "False" if the result is negative:
{optional_pay_attention}
Context: {context}
Question: {question}
{optional_steps}
Again, please answer "True" if the result is positive and "False" if the result is negative. Only True or False is engough. Do not provide any other information.
The answer is (True or False): \
"""

CONFIQA_BASE_TEMPLATE = """\
Now based on the given context answer the question.
Context: {context}
Question: {question}
Answer: 
"""

# CONFIQA_LOGICRAG_TEMPLATE = """\
# Now based on the given context answer the question.
# Please pay attention to the context wrapped by '<conflict>' and '</conflict>', which means that the information in this part is conflict with your internal knowledge.
# You should strictly follow the information in this part and do not use your internal knowledge.
# Context: {context}
# Question: {question}
# Answer: 
# """

CONFIQA_LOGICRAG_TEMPLATE = """\
Now based on the given context answer the question.
Context: {context}
Question: {question}
Answer: 
"""

FACT_DECOMPOSE_PROMPT = """\
Please breakdown the following text into independent automic facts. 
Each fact should be a complete sentence with the subject being a specific name instead of the word "the". 

For example:
Text: Christopher Nolan directed a 2006 film in which Ron Perkins' character plays the manager of a hotel.
Facts:
- Christopher Nolan directed a 2006 film.
- Ron Perkins' character plays the manager of a hotel. 

Now please breakdown the following text:
Text: {text}
Facts:
"""

QUERY_DECOMPOSE_PROMPT = """\
Please breakdown the following query into several steps. Do not be verbose.

For example:
Query: What is the character of Ron Perkins in the film directed by Christopher Nolan?
Steps:
- Identify the film directed by Christopher Nolan.
- Identify the character of Ron Perkins in that film.

Now please breakdown the following query:
Query: {query}
Steps:
"""

REALTIMEQA_TEMPLATE = """\
Context: {context}
Question: {question}
Options:
{options}
Now based on the given context answer the question and select the best answer from the options.
Please provide the answer itself and do not provide the choice letter.
Please pay attention to the context "conflict" with your internal knowledge.
You should strictly be faithful to the context, follow the information in the context and do not use your internal knowledge.
Answer: 
"""

REALTIMEQA_LOGICRAG_TEMPLATE = """\
Context: {context}
Question: {question}
Options:
{options}
Answer: 
"""

SQUAD_TEMPLATE = """\
Context: {context}
Question: {question}
Now based on the given context answer the question and select the best answer from the options.
Please provide the answer itself and do not provide the choice letter.
Please pay attention to the context "conflict" with your internal knowledge.
You should strictly be faithful to the context, follow the information in the context and do not use your internal knowledge.
Answer: 
"""

SQUAD_LOGICRAG_TEMPLATE = """\
Context: {context}
Question: {question}
Answer: 
"""

CANOE_SYSTEM_PROMPT = """\
A conversation between User and Assistant. The user gives an instruction that consists of two parts: a passage and the actual instruction, separated by two newline characters (\\n\\n).

The passage is provided within <context> and </context> tags. The Assistant need to refer to the given passage and complete the instruction. 

The Assistant solves the question by first thinking about the reasoning process internally according to the given passage and then providing the response.

The response must be structured and include the following three sections, clearly marked by the respective tags:

- **Reasoning Process**: Explain your thought process or logical steps to derive the answer. Enclose this within <think> and </think> tags.
- **Long Answer**: Provide a long response that consists of syntactically complete and semantically complete sentences to answer the question. Enclose this within <long_answer> and </long_answer> tags.
- **Short Answer**: Present a concise response that directly answer the question. Enclose this within <answer> and </answer> tags.

Format your response exactly as follows:
<think> reasoning process here.</think><long_answer> detailed answer here.</long_answer><answer> the concise answer here.</answer>.
"""

KRE_PROMPT= """\
You are a professional assistant and can answer most of the questions. I'll give you the question and the context and so on. Your goal is to answer my questions, provide explanations for your answers, and re-answer based on context if you are unsure or if your answers conflict with other information I have provided.If you can't provide an answer,please respond with 'None'.
{context}
Question: {question}
Options:
{options}
Answer:
"""

OPIN_PTOMPT="""\
Bob said \"{context}\"
Q: {question} in Bob\'s opinion?
options:
{options}
A:
"""