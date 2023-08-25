import lightning.pytorch as pl
from llama_v2_emotion_model import get_peft_LLama_model
from dataprocessor import get_train_dataloader, get_val_dataloader, get_test_dataloader

def main():
    
    model = get_peft_LLama_model()
    
    trainer = pl.Trainer(gpus=1, max_epochs=5,accelerator='gpu',default_root_dir='./checkpoints')
    trainer.fit(model, train_dataloaders=get_train_dataloader(), val_dataloaders=get_val_dataloader())
    
    # print('开始进行test')
    # trainer.test(model, test_dataloaders=get_test_dataloader())