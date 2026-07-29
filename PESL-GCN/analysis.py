import json
import torch
from config import Config
from model import PESL_GCN_Model
from data_loader import get_dataloader


def analyze_errors(model_path, test_npz_path, output_report="error_analysis_report.md"):
    config = Config()
    test_loader = get_dataloader(test_npz_path, batch_size=1, shuffle=False)

    model = PESL_GCN_Model(config).to(config.device)
    checkpoint = torch.load(model_path, map_location=config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    bad_cases = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            adj_matrix = batch["adj_matrix"].to(config.device)
            labels = batch["labels"].to(config.device)

            logits = model(input_ids, attention_mask, adj_matrix)
            preds = torch.argmax(logits, dim=-1)

            for p_seq, l_seq in zip(preds, labels):
                for idx, (p, l) in enumerate(zip(p_seq, l_seq)):
                    if l.item() != -100 and p.item() != l.item():
                        bad_cases.append({
                            "token_index": idx,
                            "predicted_label": config.id2label.get(p.item(), "O"),
                            "ground_truth_label": config.id2label.get(l.item(), "O")
                        })

    with open(output_report, "w", encoding="utf-8") as f:
        f.write("# PESL-GCN Error Analysis Report\n\n")
        f.write(f"Total Prediction Errors Found: {len(bad_cases)}\n\n")
        f.write("| Token Index | Predicted Event Label | Ground Truth Event Label |\n")
        f.write("| :---: | :---: | :---: |\n")
        for case in bad_cases[:50]:  # 列出前 50 个典型案例
            f.write(f"| {case['token_index']} | `{case['predicted_label']}` | `{case['ground_truth_label']}` |\n")

    print(f"Error analysis report generated at {output_report}")


if __name__ == "__main__":
    analyze_errors("./experiments/clue/best_pesl_gcn_mlee.pt", "./data/clue/test.npz")