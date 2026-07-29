import os
import json

def create_sample_datasets():
    data_dir = "./data/clue"
    os.makedirs(data_dir, exist_ok=True)

    # 模拟符合 BIO 标注格式的生物医学文本样本
    sample_train_data = [
        {
            "doc_id": 1,
            "tokens": ["Protein", "A", "binds", "to", "its", "promoter", "in", "cell", "death", "process"],
            "labels": ["O", "O", "B-Binding", "O", "O", "O", "O", "B-Anatomical", "I-Anatomical", "O"]
        },
        {
            "doc_id": 1,
            "tokens": ["Phosphorylation", "of", "STAT3", "inhibits", "cell", "proliferation"],
            "labels": ["B-Phosphorylation", "O", "O", "O", "B-Anatomical", "I-Anatomical"]
        }
    ]

    sample_test_data = [
        {
            "doc_id": 2,
            "tokens": ["Gene", "expression", "was", "upregulated", "during", "apoptosis"],
            "labels": ["B-Gene_expression", "I-Gene_expression", "O", "O", "O", "B-Anatomical"]
        }
    ]

    train_path = os.path.join(data_dir, "train.json")
    test_path = os.path.join(data_dir, "test.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(sample_train_data, f, indent=2, ensure_ascii=False)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(sample_test_data, f, indent=2, ensure_ascii=False)

    print(f"Sample train dataset generated at: {train_path}")
    print(f"Sample test dataset generated at: {test_path}")

if __name__ == "__main__":
    create_sample_datasets()