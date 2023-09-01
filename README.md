# study-llama

这是一个在 [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) 微调的仓库
分别使用了bert以及lora进行微调
bert+lora
|  test metric   |   |
|  ----  | ----  |
| test_acc_epoch  | 0.9245 |
| test_loss_epoch  | 0.1709 |

bert
|  test metric   |   |
|  ----  | ----  |
| test_acc_epoch  | 0.4179 |
| test_loss_epoch  | 1.5432 |

llama v2+lora
|  test metric   |   |
|  ----  | ----  |
| test_acc_epoch  | 0.9456 |
| test_loss_epoch  | 0.1156 |

llama v2
|  test metric   |   |
|  ----  | ----  |
| test_acc_epoch  | --- |
| test_loss_epoch  | --- |
---
## 存在问题
1. lightning在使用量化的时候没有使用明白，使用了16比特反而out of memory<br>
    `trainer=pl.Trainer(devices=4,max_epochs=5,accelerator='gpu',precision=16,num_nodes=1,default_root_dir='./checkpoints',strategy='ddp_find_unused_parameters_true')`
2. 如果按照下述代码会存在不在一个cuda的问题<br>
   `model = LlamaForSequenceClassification.from_pretrained(model_name,load_in_8bit=llama_args.load_in_8bit, device_map='auto',torch_dtype=llama_args.torch_dtype)`
3. 有些对比学习的任务，用交叉熵的时候，标签内容是从0~bsz-1的，我不太理解。为什么可以利用序号信息区分正负样本？<br>
   `loss = self.criterion(logits, labels)`
4. tokenizer的过程应该放到那里，是放到构造dataloader的时候，还是放到模型forward的时候?