import argparse
import json
import os
import torch


class ModelArguments:
    """
    统一参数管理解析器
    支持从 test_config.json 自动读取，并允许通过 command-line 参数进行覆盖
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="PESL-GCN Framework Master Arguments")

        # 1. 基础配置文件路径
        self.parser.add_argument("--config_file", type=str, default="test_config.json", help="Path to config JSON file")

        # 2. 训练与评估覆盖参数
        self.parser.add_argument("--batch_size", type=int, default=None, help="Batch size for training/testing")
        self.parser.add_argument("--lr", type=float, default=None, help="Learning rate")
        self.parser.add_argument("--epochs", type=int, default=None, help="Total number of training epochs")
        self.parser.add_argument("--device", type=str, default=None, help="Computing device: cuda or cpu")
        self.parser.add_argument("--pretrained_path", type=str, default=None,
                                 help="Path to pretrained SciBERT directory")
        self.parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to evaluation checkpoint .pt")

    def parse(self):
        args = self.parser.parse_args()

        # 如果配置文件存在，优先读取 JSON 内的字段
        if os.path.exists(args.config_file):
            print(f"[Arguments] Loading configuration file from: {args.config_file}")
            with open(args.config_file, "r", encoding="utf-8") as f:
                json_cfg = json.load(f)

            # 解析 JSON 中的嵌套结构
            for group_name, group_values in json_cfg.items():
                if isinstance(group_values, dict):
                    for k, v in group_values.items():
                        if not hasattr(args, k) or getattr(args, k) is None:
                            setattr(args, k, v)
                else:
                    if not hasattr(args, group_name) or getattr(args, group_name) is None:
                        setattr(args, group_name, group_values)

        # 补全默认缺失参数的保底值
        args.batch_size = args.batch_size if args.batch_size is not None else 16
        args.learning_rate = args.lr if args.lr is not None else 2e-5
        args.epochs = args.epochs if args.epochs is not None else 30
        args.device = torch.device(
            "cuda" if (args.device == "cuda" or args.device is None) and torch.cuda.is_available() else "cpu")
        args.pretrained_model_path = getattr(args, "pretrained_model_path", "./pretrained_bert_models")
        args.exp_dir = getattr(args, "exp_dir", "./experiments/clue")
        args.data_dir = getattr(args, "data_dir", "./data/clue")
        args.gcn_num_layers = getattr(args, "gcn_num_layers", 2)
        args.dropout = getattr(args, "dropout", 0.5)
        args.layer_norm_eps = float(getattr(args, "layer_norm_eps", 1e-12))
        args.num_classes = getattr(args, "num_classes", 21)

        # 建立全局标签 ID 映射字典
        if hasattr(args, "labels") and isinstance(args.labels, list):
            args.id2label = {i: label for i, label in enumerate(args.labels)}
            args.label2id = {label: i for i, label in enumerate(args.labels)}
        else:
            args.labels = [f"LABEL_{i}" for i in range(args.num_classes)]
            args.id2label = {i: label for i, label in enumerate(args.labels)}
            args.label2id = {label: i for i, label in enumerate(args.labels)}

        return args