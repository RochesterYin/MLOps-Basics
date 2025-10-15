# M3芯片Mac运行指南

**注意：此指南专门为Apple M3芯片的Mac设计，使用MPS（Metal Performance Shaders）加速**

## 系统要求

- macOS 12.3 或更高版本
- Apple M3芯片
- Python 3.8 或更高版本

## 安装步骤

### 1. 创建虚拟环境

```bash
# 使用conda创建虚拟环境
conda create --name project-setup-m3 python=3.9
conda activate project-setup-m3
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

## 运行项目

### 训练模型

```bash
python train.py
```

训练脚本会自动检测并使用MPS加速（如果可用）。

### 推理

```bash
python inference.py
```

### 运行Jupyter Notebook

```bash
# 安装Jupyter相关包
conda install ipykernel
python -m ipykernel install --user --name project-setup-m3
pip install ipywidgets

# 启动Jupyter Lab
jupyter lab
```

## 性能优化建议

1. **MPS加速**: 脚本会自动使用MPS加速，这比CPU训练快很多
2. **内存管理**: M3芯片有统一内存架构，训练时注意内存使用
3. **批处理大小**: 如果遇到内存不足，可以减小批处理大小

## 故障排除

### 如果MPS不可用
- 确保macOS版本 >= 12.3
- 确保PyTorch版本 >= 2.0.0
- 重启终端并重新激活虚拟环境

### 如果遇到内存问题
- 减小模型大小或批处理大小
- 使用梯度累积来模拟更大的批处理

### 如果遇到依赖冲突
- 使用conda而不是pip安装PyTorch：
```bash
conda install pytorch torchvision torchaudio -c pytorch
```

## 性能对比

在M3芯片上使用MPS加速，训练速度通常比CPU快3-5倍，具体取决于模型大小和复杂度。
