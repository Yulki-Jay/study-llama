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