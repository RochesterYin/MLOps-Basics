import torch
from model import ColaModel
from data import DataModule


class ColaPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = ColaModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=0)
        self.lables = ["unacceptable", "acceptable"]
        
        # 设置设备（支持MPS）
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        self.model = self.model.to(self.device)
        print(f"推理使用设备: {self.device}")

    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)
        
        # 将输入张量移动到正确的设备
        input_ids = torch.tensor([processed["input_ids"]]).to(self.device)
        attention_mask = torch.tensor([processed["attention_mask"]]).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
        
        scores = self.softmax(logits[0]).tolist()
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": score})
        return predictions


def find_latest_model():
    """自动找到最新的模型文件"""
    import os
    import glob
    
    model_dir = "./models"
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型目录 {model_dir} 不存在")
    
    # 查找所有.ckpt文件
    ckpt_files = glob.glob(os.path.join(model_dir, "*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"在 {model_dir} 中未找到.ckpt文件")
    
    # 返回最新的文件（按修改时间排序）
    latest_model = max(ckpt_files, key=os.path.getmtime)
    print(f"使用模型文件: {latest_model}")
    return latest_model

if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    
    try:
        # 自动找到最新的模型文件
        model_path = find_latest_model()
        predictor = ColaPredictor(model_path)
        
        print(f"\n预测句子: '{sentence}'")
        result = predictor.predict(sentence)
        
        print("\n预测结果:")
        for pred in result:
            print(f"  {pred['label']}: {pred['score']:.4f}")
            
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先运行训练脚本生成模型文件: python train.py")
