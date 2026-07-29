import os
import json
import numpy as np
import torch
import spacy
from transformers import AutoTokenizer


class DataProcessor:
    """
    负责提取句法依存树矩阵（Adjacency Matrix）并将其与 SciBERT Subword Tokenizer 对齐
    """

    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_path)
        # 加载 SpaCy 生物医学或通用英文句法解析器
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Warning: spacy 'en_core_web_sm' model not found. Downloading...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def build_dependency_matrix(self, text, max_len):
        """
        基于依存树构建对角线有自环 (Self-loop) 的归一化邻接矩阵 A = A + I (Eq. 6)[cite: 1]
        """
        doc = self.nlp(text)
        seq_len = min(len(doc), max_len)
        adj = np.eye(max_len, dtype=np.float32)  # 初始化自带自环的单位矩阵 I

        for token in doc:
            if token.i >= seq_len:
                continue
            # 建立头节点与依赖节点之间的双向/单向依存弧
            target_i = token.head.i
            if target_i < seq_len:
                adj[token.i, target_i] = 1.0
                adj[target_i, token.i] = 1.0  # 无向图构建，增强信息互通

        # 计算对称归一化邻接矩阵: D^(-1/2) * A * D^(-1/2) (Eq. 6)[cite: 1]
        deg = np.sum(adj, axis=-1)
        deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
        deg_inv_sqrt[deg == 0] = 0.0
        deg_mat = np.diag(deg_inv_sqrt)
        norm_adj = np.dot(np.dot(deg_mat, adj), deg_mat)

        return norm_adj

    def process_single_sample(self, tokens, labels, doc_id=0):
        """
        处理单个样本：词片段对齐、标签填充与句法邻接矩阵提取
        """
        text = " ".join(tokens)
        adj_matrix = self.build_dependency_matrix(text, self.config.max_seq_len)

        input_ids = [self.tokenizer.cls_token_id]
        label_ids = [-100]  # [CLS] 对应 mask

        for token, label in zip(tokens, labels):
            subwords = self.tokenizer.tokenize(token)
            if not subwords:
                subwords = [self.tokenizer.unk_token]

            sub_ids = self.tokenizer.convert_tokens_to_ids(subwords)
            input_ids.extend(sub_ids)

            # BIO 标记策略：仅首个 subword 保留原标签，其余标记为 -100 (Ignore)
            label_id = self.config.label2id.get(label, 0)
            label_ids.append(label_id)
            label_ids.extend([-100] * (len(sub_ids) - 1))

            if len(input_ids) >= self.config.max_seq_len - 1:
                break

        input_ids.append(self.tokenizer.sep_token_id)
        label_ids.append(-100)  # [SEP] 对应 mask

        seq_len = len(input_ids)
        attention_mask = [1] * seq_len

        # 补齐 Padding
        pad_len = self.config.max_seq_len - seq_len
        input_ids.extend([self.tokenizer.pad_token_id] * pad_len)
        attention_mask.extend([0] * pad_len)
        label_ids.extend([-100] * pad_len)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "adj_matrix": adj_matrix.tolist(),
            "labels": label_ids,
            "doc_id": doc_id
        }

    def convert_dataset_to_npz(self, json_file, output_npz):
        """读取标准 JSON 数据并保存为 npz 压缩缓存"""
        print(f"Processing {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        processed_data = []
        for item in raw_data:
            sample = self.process_single_sample(
                tokens=item["tokens"],
                labels=item["labels"],
                doc_id=item.get("doc_id", 0)
            )
            processed_data.append(sample)

        np.savez_compressed(output_npz, data=np.array(processed_data, dtype=object))
        print(f"Saved processed cache to {output_npz}")