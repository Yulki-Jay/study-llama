import torch
model_name = "/home/jiangyunqi/study/LLM/study-llama/llama-finetune/llama-2-7b"
num_labels = 6
LORA_R = 8
LORA_ALPHA = 1
LORA_DROPOUT = 0.05
train_bsz = 16
val_bsz = 16
test_bsz = 16
load_in_8bit = True
torch_dtype = torch.float16