import lightning.pytorch as pl
from bert_emotion_model import MyBERTModel,get_peft_bert_model,get_bert_model
from dataprocessor import DataProcessor
from transformers import BertTokenizer,BertForSequenceClassification
import llama_args
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 根据你的显卡数量和编号进行设置

'''
出现的bug是tokenize后的dataloader在跨文件的时候
传递的东西不一样
目前正在思考如何解决

解决思路：封装到类中，然后在需要的时候进行实例化
'''

def get_model(model_name,lora):
    if lora == False:
        model = get_bert_model(model_name)
    else:
        model = get_peft_bert_model(model_name)
    return model    
    


def main():
    print('开始加载模型')
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = get_model(model_name,lora=False)
    # 开始获取bert-lora模型
    print(model)
    print('模型获取成功')
    
    print('*'*50)
    print('数据开始加载')
    dataprocessor = DataProcessor()
    # dataloader = dataprocessor.get_train_dataloader()
    # data = next(iter(dataloader))
    # print(data.keys())
    print('数据加载成功')
    print('*'*50)
    print('模型开始训练')
    # trainer = pl.Trainer(devices=1,max_epochs=5,accelerator='gpu',default_root_dir='./checkpoints',limit_train_batches=5,limit_val_batches=1,limit_test_batches=5)
    trainer = pl.Trainer(devices=4,max_epochs=50,accelerator='gpu',strategy='ddp',num_nodes=1,default_root_dir='./checkpoints')
    
    trainer.fit(model, train_dataloaders=dataprocessor.get_train_dataloader(), val_dataloaders=dataprocessor.get_val_dataloader())
    print('模型训练成功')
    print('*'*50)
    print('开始进行test')
    trainer.test(model, dataloaders=dataprocessor.get_test_dataloader())
    
if __name__ == '__main__':
    main()