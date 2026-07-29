import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class EventDataset(Dataset):
    """
    PESL-GCN 专用生物医学事件提取数据集加载器
    负责解析预处理生成的 .npz 格式张量数组
    """

    def __init__(self, data_path: str):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"[Error] Specified dataset path does not exist: {data_path}")

        print(f"[Dataset] Loading processed numpy binary file from: {data_path}")
        data = np.load(data_path, allow_pickle=True)

        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.adj_matrix = data["adj_matrix"]
        self.labels = data["labels"]

        # 如果 npz 包含文档 ID 则读取，否则自动生成虚构 ID
        if "doc_ids" in data:
            self.doc_ids = data["doc_ids"]
        else:
            self.doc_ids = [f"DOC_{i:05d}" for i in range(len(self.input_ids))]

        self._validate_shapes()

    def _validate_shapes(self):
        """严格校验数据集内部各矩阵张量的样本数对齐情况"""
        n_samples = len(self.input_ids)
        assert len(self.attention_mask) == n_samples, "Attention mask length mismatch!"
        assert len(self.adj_matrix) == n_samples, "Adjacency matrix length mismatch!"
        assert len(self.labels) == n_samples, "Labels length mismatch!"
        print(f"[Dataset] Successfully validated {n_samples} samples.")

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict:
        """
        提取单条样本并转换为 PyTorch 强类型 Tensor
        """
        return {
            "doc_id": str(self.doc_ids[idx]),
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "adj_matrix": torch.tensor(self.adj_matrix[idx], dtype=torch.float32),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def build_dataloader(
        data_path: str,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 2
) -> DataLoader:
    """
    构建全功能 PyTorch DataLoader 实例
    """
    dataset = EventDataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    return dataloader