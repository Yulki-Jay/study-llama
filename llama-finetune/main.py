import lightning.pytorch as pl
from llama_v2_emotion_model import get_peft_LLama_model,get_LLama_model
from llama_v2_emotion_model import MyLLamaModel
import llama_args
from dataprocessor import DataProcessor
import datetime
'''
出现的问题：没有办法进行量化，如果在trainer中进行量化的话，会cuda out of memory，目前还没有解决
'''

def get_model(model_name,lora):
    if lora == False:
        model = get_LLama_model(model_name)
    else:
        model = get_peft_LLama_model(model_name)
    return model    


def main():
    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    print('开始加载模型')
    model_name = llama_args.model_name
    model = get_model(model_name,lora=False)
    # 开始获取llama-lora模型
    # print(model)
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
    trainer = pl.Trainer(devices=4,max_epochs=5,accelerator='gpu',precision=16,default_root_dir='./checkpoints',limit_train_batches=5,limit_val_batches=1,limit_test_batches=5,strategy='ddp_find_unused_parameters_true') 
    # 因为有些参数没有用到，所以会出现报错的情况，因此如果采用ddp的话，就需要设置strategy='ddp_find_unused_parameters_true'
    # trainer = pl.Trainer(devices=4,max_epochs=5,accelerator='gpu',precision=16,num_nodes=1,default_root_dir='./checkpoints',strategy='ddp_find_unused_parameters_true')
    
    trainer.fit(model, train_dataloaders=dataprocessor.get_train_dataloader(), val_dataloaders=dataprocessor.get_val_dataloader())
    print('模型训练成功')
    print('*'*50)
    print('开始进行test')
    trainer.test(model, dataloaders=dataprocessor.get_test_dataloader())
    finish_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    print(f'开始时间：{current_time}\n结束时间：{finish_time}')
if __name__ == '__main__':
    main()