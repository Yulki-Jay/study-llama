from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import lightning.pytorch as pl
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer
import llama_args
import  torch.nn as nn
from peft import get_peft_model, prepare_model_for_int8_training,LoraConfig


# 这个地方还没有进行修改
class MyLLamaModel(pl.LightningModule):
    def __init__(self, model,infeatures,outfeatures):
        super().__init__()
        self.model = model
        self.model.lm_head = nn.Linear(infeatures,outfeatures)
    
    def training_step(self,batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss
    def validation_step(self,batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        return loss
    def test_step(self,batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

def get_LLama_model():
    model = LlamaForCausalLM.from_pretrained(llama_args.model_name)
    infeatures = model.lm_head.in_features
    outfeatures = llama_args.num_labels
    my_model = MyLLamaModel(model,infeatures,outfeatures)
    return my_model

def get_peft_config():
    config = LoraConfig(
        r = llama_args.LORA_R,
        lora_alpha=llama_args.LORA_ALPHA,
        lora_dropout=llama_args.LORA_DROPOUT,
        bias = "none",
        target_modules=["q_proj","v_proj"],
        task_type="SEQ_CLS",
    )
    return config


def get_peft_LLama_model():
    model = get_LLama_model()
    config = get_peft_config()
    model = get_peft_model(model,config)
    model.print_trainable_parameters()
    return model


def main():
    # model = get_LLama_model()
    # print(model)
    model2 = get_peft_LLama_model()
    print(model2)
    
if __name__ == '__main__':
    main()
    # 这个模块已经执行成功了，还没有测试是否可以训练起来