export CUDA_VISIBLE_DEVICES=2
nohup torchrun --nnodes 1 --nproc_per_node 1 --master_addr 127.0.0.1 --master_port 29501 scripts/llm_sft.py > logs/train_canoe_mistral_7b_0705.log 2>&1 &
# nohup python scripts/llm_sft.py > logs/train_sft_llama2-7b_0524.log 2>&1 &
# nohup accelerate launch cripts/llm_sft.py > logs/train_sft_llama2-7b_0524.log 2>&1 &