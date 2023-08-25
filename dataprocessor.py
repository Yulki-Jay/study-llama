from datasets import load_dataset
from tokenizer import get_tokenizer
import torch.utils.data
ds_name = 'dair-ai/emotion'

ds = load_dataset(ds_name)
def encode(examples):
    tokenizer = get_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer(examples["text"], truncation=True, padding=True)

ds = ds.map(encode, batched=True) # tokenize the dataset

train_ds = ds['train']
val_ds = ds['validation']
test_ds = ds['test']

# print(ds)
# label = ds['train'].features['label']

# print(label)
# print(ds['train'][0])
def get_train_dataloader():
    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=128,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )
    return train_dataloader

def get_val_dataloader():
    val_dataloader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=128,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )
    return val_dataloader

def get_test_dataloader():
    test_dataloader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=2,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )
    return test_dataloader

def main():
    train_dataloader = get_test_dataloader()
    # print(train_dataloader[0])
    batch =  next(iter(train_dataloader))
    inputs_ids = batch['input_ids']
    labels = batch['label']
    print(inputs_ids)
    print(labels)
    
if __name__ == '__main__':
    main()