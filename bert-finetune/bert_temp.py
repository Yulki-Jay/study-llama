from transformers import BertTokenizer,BertForSequenceClassification

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name).cuda()
print(model)

test = "I am a student"
token = tokenizer(test, truncation=True, padding="max_length",return_tensors="pt")
token = token.to('cuda')

print(token)
print(token.keys())
print(model(input_ids=token['input_ids'],attention_mask=token['attention_mask'],token_type_ids=token['token_type_ids']))