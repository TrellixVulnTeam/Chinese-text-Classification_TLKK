# Chinese-text-Classification
本仓库包含如下两部分：
- 基于pytorch的微调训练（训练10个epoch，在测试集测试结果为**89%**）
- 基于自定义模型的微调训练

## 基于pytorch

### data
- train
训练数据，验证数据（通过拆分）
- testA
待预测数据，仓库提供label（testA_label）以供测试

### model
- 使用官方pytorch的**bert-base-chinese**预训练模型
- 注意：文本编码需要与**bert-base-chinese**对应

### distributed
- 本仓库提供多GPU分布式训练、单GPU训练两种模式
- 建议使用单GPU调试，多GPU训练
- 多GPU训练命令：`CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_GPU.py`

### predict
对**test**的预测脚本

## 基于custom

### data
需要使用脚本`make_dataset.py`对`csv`数据转换为`txt`格式

### pretrain
使用**huggingface**提供的预训练模型及词表
