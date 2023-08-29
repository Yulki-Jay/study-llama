from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import lightning.pytorch as pl
import torch.nn.functional as F
from transformers import LlamaForSequenceClassification, LlamaConfig, LlamaTokenizer
import llama_args
import  torch.nn as nn
from peft import get_peft_model, prepare_model_for_int8_training,LoraConfig
from torch.cuda.amp import autocast
# scaler = torch.cuda.amp.GradScaler()

'''
Llama 模型现在有问题，原版的llama并不支持seq classification 需要进行修改一下

'''

# 这个地方还没有进行修改
class MyLLamaModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # self.model.lm_head = nn.Linear(infeatures,outfeatures)
        # self.model.classifier = nn.Linear(infeatures,outfeatures)
    def training_step(self,batch, batch_idx):
        # batch = 'label','input_ids','attention_mask'
        #print('***'+str(batch.keys()))
        # 'input_ids','attention_mask','token_type_ids','labels'
        # print(batch.keys())
        labels,input_ids,attention_mask= batch['labels'],batch['input_ids'],batch['attention_mask']
        y = labels
        # y = torch.nn.functional.one_hot(y, num_classes=llama_args.num_labels)  # 将 y 转换为独热编码
        # with autocast():
        y_hat = self.model(input_ids=input_ids,attention_mask=attention_mask)
        # print(f"y={y.shape},y_hat={y_hat.logits.shape}")
        loss = F.cross_entropy(y_hat.logits, y)
        self.log('train_loss', loss,on_step=True,on_epoch=True,prog_bar=True)
        acc = self.compute_accuracy(y_hat.logits, y)
        self.log('train_acc', acc,on_step=True,on_epoch=True,prog_bar=True)
        return loss
    
    def validation_step(self,batch, batch_idx):
        labels,input_ids,attention_mask= batch['labels'],batch['input_ids'],batch['attention_mask']
        y = labels
        # y = torch.nn.functional.one_hot(y, num_classes=llama_args.num_labels)  # 将 y 转换为独热编码
        # with autocast():
        y_hat = self.model(input_ids=input_ids,attention_mask=attention_mask)
        # print(f"y={y.shape},y_hat={y_hat.logits.shape}")
        loss = F.cross_entropy(y_hat.logits, y)
        self.log('val_loss', loss,on_step=True,on_epoch=True,prog_bar=True)
        acc = self.compute_accuracy(y_hat.logits, y)
        self.log('val_acc', acc,on_step=True,on_epoch=True,prog_bar=True)
        return loss
    
    def test_step(self,batch, batch_idx):
        labels,input_ids,attention_mask= batch['labels'],batch['input_ids'],batch['attention_mask']
        y = labels
        # y = torch.nn.functional.one_hot(y, num_classes=llama_args.num_labels)  # 将 y 转换为独热编码
        # with autocast():
        y_hat = self.model(input_ids=input_ids,attention_mask=attention_mask)
        loss = F.cross_entropy(y_hat.logits, y)
        self.log('test_loss', loss,on_step=True,on_epoch=True,prog_bar=True)
        acc = self.compute_accuracy(y_hat.logits, y)
        self.log('test_acc', acc,on_step=True,on_epoch=True,prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
    def compute_accuracy(self, logits, labels):
        preds = torch.argmax(logits, dim=-1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def get_LLama_model(model_name): # 获得院士的llama模型，并且封装到pytorch lightning中
    model = LlamaForSequenceClassification.from_pretrained(model_name)
    infeatures = model.score.in_features
    outfeatures = llama_args.num_labels
    model.score = nn.Linear(infeatures,outfeatures)
    my_model = MyLLamaModel(model)
    return my_model

def get_peft_config(): # 获得peft的config 这里面用的是lora config
    config = LoraConfig(
        r = llama_args.LORA_R,
        lora_alpha=llama_args.LORA_ALPHA,
        lora_dropout=llama_args.LORA_DROPOUT,
        bias = "none",
        target_modules=["q_proj","v_proj"],
        task_type="SEQ_CLS",
    )
    return config


def get_peft_LLama_model(model_name): # 获得peft的模型
    model = LlamaForSequenceClassification.from_pretrained(model_name)
    infeatures = model.score.in_features
    outfeatures = llama_args.num_labels
    model.score = nn.Linear(infeatures,outfeatures)
    config = get_peft_config()
    model = get_peft_model(model,config)
    model.print_trainable_parameters()
    # print(model)
    my_model = MyLLamaModel(model)
    return my_model


def main():
    # model = get_LLama_model()
    # print(model)
    model2 = get_LLama_model(llama_args.model_name)
    print(model2)
    
if __name__ == '__main__':
    main()
    # 这个模块已经执行成功了，还没有测试是否可以训练起来
    # 在训练的时候 会出错 
    # TypeError: `model` must be a `LightningModule` or `torch._dynamo.OptimizedModule`, got `PeftModelForSequenceClassification`
    # 这个错误是因为没有继承LightningModule，