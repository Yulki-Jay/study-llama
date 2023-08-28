from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import lightning.pytorch as pl
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer,BertForSequenceClassification
import llama_args
import  torch.nn as nn
from peft import get_peft_model, prepare_model_for_int8_training,LoraConfig


# 这个地方还没有进行修改
class MyBERTModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # self.model.lm_head = nn.Linear(infeatures,outfeatures)
        # self.model.classifier = nn.Linear(infeatures,outfeatures)
        # self.save_hyperparameters() # 这个地方是为了保存超参数
    def training_step(self,batch, batch_idx):
        # batch = [text,label,input_ids,attention_mask]
        #print('***'+str(batch.keys()))
        # 'input_ids','attention_mask','token_type_ids','labels'
        # print(batch.keys())
        input_ids,attention_mask,token_type_ids,labels= batch['input_ids'],batch['attention_mask'],batch['token_type_ids'],batch['labels']
        y = labels
        y_hat = self.model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        # print(f"y={y.shape},y_hat={y_hat.logits.shape}")
        loss = F.cross_entropy(y_hat.logits, y)
        self.log('train_loss', loss,on_step=True,on_epoch=True,prog_bar=True)
        acc = self.compute_accuracy(y_hat.logits, y)
        self.log('train_acc', acc,on_step=True,on_epoch=True,prog_bar=True)
        return loss
    
    def validation_step(self,batch, batch_idx):
        input_ids,attention_mask,token_type_ids,labels = batch['input_ids'],batch['attention_mask'],batch['token_type_ids'],batch['labels']
        y = labels
        y_hat = self.model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        # print(f"y={y.shape},y_hat={y_hat.logits.shape}")
        loss = F.cross_entropy(y_hat.logits, y)
        self.log('val_loss', loss,on_step=True,on_epoch=True,prog_bar=True)
        acc = self.compute_accuracy(y_hat.logits, y)
        self.log('val_acc', acc,on_step=True,on_epoch=True,prog_bar=True)
        return loss
    
    def test_step(self,batch, batch_idx):
        input_ids,attention_mask,token_type_ids,labels = batch['input_ids'],batch['attention_mask'],batch['token_type_ids'],batch['labels']
        y = labels
        y_hat = self.model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
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

def get_bert_model(model_name): # 获得院士的llama模型，并且封装到pytorch lightning中
    model = BertForSequenceClassification.from_pretrained(model_name)
    infeatures = model.classifier.in_features
    outfeatures = llama_args.num_labels
    model.classifier = nn.Linear(infeatures,outfeatures)
    my_model = MyBERTModel(model)
    return my_model

def get_peft_config(): # 获得peft的config 这里面用的是lora config
    config = LoraConfig(
        r = llama_args.LORA_R,
        lora_alpha=llama_args.LORA_ALPHA,
        lora_dropout=llama_args.LORA_DROPOUT,
        bias = "none",
        target_modules=["query","value"],
        task_type="SEQ_CLS",
    )
    return config


def get_peft_bert_model(model_name): # 获得peft的模型
    model = BertForSequenceClassification.from_pretrained(model_name)
    infeatures = model.classifier.in_features
    outfeatures = llama_args.num_labels
    model.classifier = nn.Linear(infeatures,outfeatures)
    config = get_peft_config()
    # print(model)
    model = get_peft_model(model,config)
    model.print_trainable_parameters()
    print(model)
    my_model = MyBERTModel(model)
    return my_model


def main():
    # model = get_LLama_model()
    # print(model)
    model2 = get_peft_bert_model('bert-base-uncased')
    print(model2)
    
if __name__ == '__main__':
    main()
    # 这个模块已经执行成功了，还没有测试是否可以训练起来
    # 在训练的时候 会出错 
    # TypeError: `model` must be a `LightningModule` or `torch._dynamo.OptimizedModule`, got `PeftModelForSequenceClassification`
    # 这个错误是因为没有继承LightningModule，