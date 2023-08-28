import lightning.pytorch as pl
from bert_emotion_model import MyBERTModel
from dataprocessor import DataProcessor
from transformers import BertTokenizer,BertForSequenceClassification
import llama_args
'''
出现的bug是tokenize后的dataloader在跨文件的时候
传递的东西不一样
目前正在思考如何解决

解决思路：封装到类中，然后在需要的时候进行实例化
'''


def main():
    print('开始加载模型')
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name).cuda()
    model = MyBERTModel(model,768,llama_args.num_labels)
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
    trainer = pl.Trainer(devices=1,max_epochs=5,accelerator='gpu',default_root_dir='./checkpoints')
    
    trainer.fit(model, train_dataloaders=dataprocessor.get_train_dataloader(), val_dataloaders=dataprocessor.get_val_dataloader())
    print('模型训练成功')
    print('*'*50)
    print('开始进行test')
    trainer.test(model, dataloaders=dataprocessor.get_test_dataloader())
    
if __name__ == '__main__':
    main()