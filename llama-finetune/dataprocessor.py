from datasets import load_dataset
from tokenizer import get_tokenizer
import torch.utils.data
from transformers import DataCollatorWithPadding # 这个是用来进行动态padding的


'''
直接封装到类里面，就不会出现数据不一致的问题了
'''

class DataProcessor():
    def __init__(self):
        self.ds_name = 'dair-ai/emotion'
        self.ds = load_dataset(self.ds_name)
        self.tokenizer = get_tokenizer()
        self.ds_maped = None
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.get_maped_dataset() # 获得map后的dataset,一定要在最下面执行，因为llama v2的tokenizer中没有pad_token_id，所以要在这里设置一下
    
    def encode(self,examples):
        return self.tokenizer(examples["text"], truncation=True, padding=True)
    
    def get_maped_dataset(self):
        self.ds_maped = self.ds.map(self.encode,batched=True)
        self.ds_maped.set_format(type='torch',columns=['label','input_ids','attention_mask'])
        print('已经map了啊')
    
    def get_train_dataset(self):
        return self.ds_maped['train']

    def get_val_dataset(self):
        return self.ds_maped['validation']

    def get_test_dataset(self):
        return self.ds_maped['test']

    def get_train_dataloader(self):
        train_ds = self.get_train_dataset()
        print(train_ds[500])
        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_ds,
            batch_size=128,
            num_workers=1,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            collate_fn=self.data_collator
        )
        # print('in dataloader'+str( next(iter(train_dataloader)).keys()))
        return train_dataloader

    def get_val_dataloader(self):
        val_ds = self.get_val_dataset()
        val_dataloader = torch.utils.data.DataLoader(
            dataset=val_ds,
            batch_size=128,
            num_workers=1,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            collate_fn=self.data_collator
        )
        return val_dataloader

    def get_test_dataloader(self):
        test_ds = self.get_val_dataset()
        test_dataloader = torch.utils.data.DataLoader(
            dataset=test_ds,
            batch_size=128,
            num_workers=1,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            collate_fn=self.data_collator
        )
        return test_dataloader



def main():
    # train_dataloader = get_train_dataloader()
    # # print(train_dataloader[0])
    # batch =  next(iter(train_dataloader))
    # print(batch.keys())
    # # inputs_ids = batch['input_ids']
    # # labels = batch['label']
    # # print(inputs_ids)
    # # print(labels)
    # # print(batch)
    dataprocessor = DataProcessor()
    train_dataloader = dataprocessor.get_train_dataloader()
    batch = next(iter(train_dataloader))
    print(batch.keys())
    print(type(batch['input_ids']))
    # print(dataprocessor.pad_token_id)
    # print(dataprocessor.ds.features['label'].names)
if __name__ == '__main__':
    main()
    
# 存在bug 当前文件执行 和 在不同文件执行结果不一样