# M3芯片Mac运行指南 - Week 2 (Hydra Config)

**注意：此指南专门为Apple M3芯片的Mac设计，使用MPS（Metal Performance Shaders）加速**

## 系统要求

- macOS 12.3 或更高版本
- Apple M3芯片
- Python 3.8 或更高版本

## 安装步骤

### 1. 创建虚拟环境

```bash
# 使用conda创建虚拟环境
conda create --name hydra-m3 python=3.9
conda activate hydra-m3
```

### 2. 安装依赖

```bash
# 安装M3芯片兼容的依赖
pip install -r requirements_m3.txt
```

### 3. 验证MPS支持

运行以下Python代码验证MPS是否可用：

```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"MPS可用: {torch.backends.mps.is_available()}")
print(f"MPS已构建: {torch.backends.mps.is_built()}")
```

### 4. 配置W&B（可选）

如果你还没有W&B账号，可以：
1. 访问 https://wandb.ai 注册账号
2. 运行 `wandb login` 登录

**注意**：代码默认使用当前登录的W&B用户。如果你想使用特定的entity，可以设置环境变量：
```bash
export WANDB_ENTITY=your_username
```

## 运行项目

### 训练模型

**方式1：使用W&B在线模式（推荐）**

首先登录W&B：
```bash
wandb login
```

然后运行训练（默认使用当前登录的用户）：
```bash
python train.py
```

如果需要指定特定的entity（团队或用户名），可以设置环境变量：
```bash
export WANDB_ENTITY=your_username
python train.py
```

**方式2：使用W&B离线模式**

如果不想登录W&B，可以使用离线模式：
```bash
WANDB_MODE=offline python train.py
```

训练脚本会自动检测并使用MPS加速（如果可用）。训练完成后，你会在日志末尾看到W&B的链接，点击链接可以查看训练指标和可视化。

### 使用Hydra配置

Week 2 使用 Hydra 进行配置管理。配置文件位于 `configs/` 目录：

- `configs/config.yaml` - 主配置文件
- `configs/model/default.yaml` - 模型配置
- `configs/processing/default.yaml` - 数据处理配置
- `configs/training/default.yaml` - 训练配置

你可以通过命令行覆盖配置参数：

```bash
# 修改训练轮数
python train.py training.max_epochs=5

# 修改批处理大小
python train.py processing.batch_size=32

# 修改模型名称
python train.py model.name=bert-base-uncased

# 组合多个参数
python train.py training.max_epochs=5 processing.batch_size=32
```

### 推理

训练完成后，更新 `inference.py` 中的模型检查点路径，然后运行：

```bash
python inference.py
```

### 运行Jupyter Notebook

```bash
# 安装Jupyter相关包
conda install ipykernel
python -m ipykernel install --user --name hydra-m3
pip install ipywidgets

# 启动Jupyter Lab
jupyter lab
```

## 代码更新说明

为了兼容PyTorch Lightning 2.0+ 和 M3芯片，代码已做以下更新：

1. **model.py**: 
   - 将 `validation_epoch_end` 更新为 `on_validation_epoch_end`
   - 更新 torchmetrics API（使用 `F1Score` 而不是 `F1`，添加 `task="binary"` 参数）
   - 添加了CPU转换以确保MPS张量可以正确转换为numpy数组

2. **train.py**:
   - 添加了 `accelerator="auto"` 以自动检测MPS/GPU/CPU
   - 更新了回调函数以支持MPS张量
   - 更新了W&B logger配置，使用当前登录用户

## 性能优化建议

1. **MPS加速**: 脚本会自动使用MPS加速，这比CPU训练快很多
2. **内存管理**: M3芯片有统一内存架构，训练时注意内存使用
3. **批处理大小**: 如果遇到内存不足，可以在配置文件中减小 `batch_size` 参数
4. **W&B监控**: 使用W&B可以实时监控训练过程，包括损失、准确率等指标
5. **Hydra配置**: 使用Hydra可以轻松管理不同的实验配置，无需修改代码

## 故障排除

### 如果MPS不可用
- 确保macOS版本 >= 12.3
- 确保PyTorch版本 >= 2.0.0
- 重启终端并重新激活虚拟环境

### 如果遇到内存问题
- 减小模型大小或批处理大小
- 使用梯度累积来模拟更大的批处理
- 在配置文件中设置 `training.limit_train_batches` 和 `training.limit_val_batches`

### 如果遇到依赖冲突
- 使用conda而不是pip安装PyTorch：
```bash
conda install pytorch torchvision torchaudio -c pytorch
```

### 如果W&B登录失败
- 检查网络连接
- 运行 `wandb login` 重新登录
- 或者设置环境变量 `WANDB_MODE=offline` 进行离线模式

### 如果遇到 "project not found" 错误
- 代码默认使用当前登录的W&B用户创建项目
- 如果项目不存在，W&B会自动创建
- 如果仍有问题，检查你的W&B账号是否有权限创建项目
- 或者使用离线模式：`WANDB_MODE=offline python train.py`

### 如果遇到tokenizers并行警告
设置环境变量：
```bash
export TOKENIZERS_PARALLELISM=false
```

### 如果遇到protobuf版本冲突
已通过 `requirements_m3.txt` 中的 `protobuf>=5.28.0,<6.0.0` 解决。如果仍有问题，可以手动安装：
```bash
pip install "protobuf>=5.28.0,<6.0.0"
```

## 性能对比

在M3芯片上使用MPS加速，训练速度通常比CPU快3-5倍，具体取决于模型大小和复杂度。

## W&B Dashboard

训练完成后，访问W&B dashboard可以查看：
- 训练和验证损失曲线
- 准确率、精确率、召回率、F1分数
- 混淆矩阵
- 错误预测的样本示例

## Hydra配置示例

查看和修改配置文件：

```bash
# 查看当前配置
python train.py --cfg job

# 查看特定配置组
cat configs/training/default.yaml

# 运行实验时覆盖配置
python train.py training.max_epochs=10 processing.batch_size=128
```

