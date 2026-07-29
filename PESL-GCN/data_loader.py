import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MLEEDataset(Dataset):
    """
    MLEE 生物医学事件提取数据集封装类[cite: 1]
    """
    def __init__(self, npz_path):
        loaded = np.load(npz_path, allow_pickle=True)
        self.data = loaded['data']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "adj_matrix": torch.tensor(item["adj_matrix"], dtype=torch.float32),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
            "doc_id": item["doc_id"]
        }

def collate_fn(batch):
    """
    批次组合函数，将单一 Batch 的数据转换为 Tensor 组合
    """
    input_ids = torch.stack([x["input_ids"] for x in batch])
    attention_mask = torch.stack([x["attention_mask"] for x in batch])
    adj_matrix = torch.stack([x["adj_matrix"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    doc_ids = [x["doc_id"] for x in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "adj_matrix": adj_matrix,
        "labels": labels,
        "doc_ids": doc_ids
    }

def get_dataloader(npz_path, batch_size, shuffle=True):
    dataset = MLEEDataset(npz_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )