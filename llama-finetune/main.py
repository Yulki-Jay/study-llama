import lightning.pytorch as pl
from llama_v2_emotion_model import get_peft_LLama_model,get_LLama_model
from llama_v2_emotion_model import MyLLamaModel
import llama_args
from dataprocessor import DataProcessor

def get_model(model_name,lora):
    if lora == False:
        model = get_LLama_model(model_name)
    else:
        model = get_peft_LLama_model(model_name)
    return model    


def main():
    print('开始加载模型')
    model_name = llama_args.model_name
    model = get_model(model_name,lora=True)
    # 开始获取llama-lora模型
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
    trainer = pl.Trainer(devices=1,max_epochs=5,accelerator='gpu',default_root_dir='./checkpoints',limit_train_batches=5,limit_val_batches=1,limit_test_batches=5)
    # trainer = pl.Trainer(devices=1,max_epochs=5,accelerator='gpu',num_nodes=1,default_root_dir='./checkpoints')
    
    trainer.fit(model, train_dataloaders=dataprocessor.get_train_dataloader(), val_dataloaders=dataprocessor.get_val_dataloader())
    print('模型训练成功')
    print('*'*50)
    print('开始进行test')
    # trainer.test(model, dataloaders=dataprocessor.get_test_dataloader())
   
if __name__ == '__main__':
    main()