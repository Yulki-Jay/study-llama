from transformers import LlamaTokenizer
import llama_args  
def get_tokenizer():
    tokenizer = LlamaTokenizer.from_pretrained(llama_args.model_name)
    return tokenizer

print(llama_args.model_name)
text = "I am a student"
tokenizer = get_tokenizer()
token = tokenizer(text)
print(token)