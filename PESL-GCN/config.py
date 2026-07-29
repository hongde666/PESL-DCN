import os
import json
import argparse
import torch


class Config:
    """
    PESL-GCN 全局配置类（结合本地 pretrained_bert_models/config.json 参数）
    """

    def __init__(self):
        # ---------------- 1. 路径配置 ----------------
        self.data_dir = "./data/clue"
        self.exp_dir = "./experiments/clue"
        # 预训练模型本地路径（包含 config.json, pytorch_model.bin, vocab.txt）
        self.pretrained_model_path = "./pretrained_bert_models"

        # ---------------- 2. 加载本地 config.json 并映射参数 ----------------
        bert_config_file = os.path.join(self.pretrained_model_path, "config.json")
        if os.path.exists(bert_config_file):
            with open(bert_config_file, "r", encoding="utf-8") as f:
                bert_config_json = json.load(f)
            # 从 JSON 自动映射核心维度参数
            self.embedding_dim = bert_config_json.get("hidden_size", 768)
            self.max_position_embeddings = bert_config_json.get("max_position_embeddings", 512)
            self.vocab_size = bert_config_json.get("vocab_size", 21128)
        else:
            # 若本地 json 不存在，使用默认配置
            self.embedding_dim = 768
            self.max_position_embeddings = 512
            self.vocab_size = 21128

        # ---------------- 3. 序列长度校验 ----------------
        self.max_seq_len = 128
        assert self.max_seq_len <= self.max_position_embeddings, \
            f"max_seq_len ({self.max_seq_len}) 不能超过 max_position_embeddings ({self.max_position_embeddings})"

        # ---------------- 4. 标签映射配置 (MLEE Dataset) ----------------
        self.label_list = [
            "O",
            "B-Anatomical", "I-Anatomical",
            "B-Molecular", "I-Molecular",
            "B-General", "I-General",
            "B-Planned", "I-Planned",
            "B-Phosphorylation", "I-Phosphorylation",
            "B-Gene_expression", "I-Gene_expression",
            "B-Binding", "I-Binding",
            "B-Localization", "I-Localization",
            "B-Regulation", "I-Regulation",
            "B-Transcription", "I-Transcription"
        ]
        self.num_classes = len(self.label_list)
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for i, label in enumerate(self.label_list)}

        # ---------------- 5. 双流网络与超参数设置 ----------------
        self.gcn_hidden_dim = self.embedding_dim  # GCN 隐藏层维度与 BERT hidden_size (768) 保持一致
        self.gcn_num_layers = 2  # GCN 卷积层数
        self.dropout = 0.5  # 分类器 Dropout
        self.learning_rate = 3e-5  # AdamW 学习率
        self.weight_decay = 0.01  # 权重衰减
        self.batch_size = 16  # Batch Size
        self.epochs = 30  # 训练轮次
        self.seed = 42

        # ---------------- 6. 损失加权与自适应阈值 ----------------
        self.w_pos = 2.0  # 事件触发词权重
        self.w_neg = 0.2  # 背景类 'O' 权重
        self.adaptive_threshold = 0.25  # 推理置信度阈值 tau

        # ---------------- 7. 设备绑定 ----------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.exp_dir, exist_ok=True)

    def parse_args(self):
        parser = argparse.ArgumentParser(description="PESL-GCN Parameters")
        parser.add_argument("--batch_size", type=int, default=self.batch_size)
        parser.add_argument("--lr", type=float, default=self.learning_rate)
        parser.add_argument("--epochs", type=int, default=self.epochs)
        parser.add_argument("--pretrained_path", type=str, default=self.pretrained_model_path)
        args = parser.parse_args()

        self.batch_size = args.batch_size
        self.learning_rate = args.lr
        self.epochs = args.epochs
        self.pretrained_model_path = args.pretrained_path
        return self