import os
import json
import torch
from tqdm import tqdm

from config import Config
from utils import get_logger, set_seed
from data_loader import get_dataloader
from model import PESL_GCN_Model
from metrics import MetricsEvaluator


def run_test_evaluation(model, test_loader, config, evaluator, logger):
    """
    在测试集上运行完整的评估，并生成逐 Token 的错误分析报告
    """
    model.eval()
    all_preds = []
    all_labels = []
    error_cases = []

    logger.info("Starting Evaluation on Test Dataset...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing Progress")):
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            adj_matrix = batch["adj_matrix"].to(config.device)
            labels = batch["labels"].to(config.device)
            doc_ids = batch.get("doc_ids", [batch_idx] * len(input_ids))

            # 前向传播
            logits = model(input_ids, attention_mask, adj_matrix)
            probs = torch.softmax(logits, dim=-1)

            # 提取最大概率值与预测类别 ID
            max_probs, preds = torch.max(probs, dim=-1)

            # 应用自适应置信度阈值过滤 (tau = 0.25)
            preds = torch.where(
                (max_probs < config.adaptive_threshold) & (preds != 0),
                torch.tensor(0, device=config.device),
                preds
            )

            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_preds.extend(preds_np.tolist())
            all_labels.extend(labels_np.tolist())

            # 收集预测错误的 Case 用于论文 Case Study 分析
            for b_i in range(len(input_ids)):
                seq_len = len(preds_np[b_i])
                for t_i in range(seq_len):
                    true_idx = labels_np[b_i][t_i]
                    pred_idx = preds_np[b_i][t_i]

                    # 忽略 -100 padding/subword 标记
                    if true_idx == -100:
                        continue

                    if true_idx != pred_idx:
                        error_cases.append({
                            "doc_id": doc_ids[b_i],
                            "token_pos": t_i,
                            "gold_label": config.id2label.get(true_idx, "O"),
                            "pred_label": config.id2label.get(pred_idx, "O"),
                            "confidence": round(float(max_probs[b_i][t_i].cpu()), 4)
                        })

    # 计算整体 Metrics 指标
    metrics = evaluator.compute_metrics(all_preds, all_labels)

    # 打印评估报告
    logger.info("=" * 60)
    logger.info("                PESL-GCN TEST RESULTS                ")
    logger.info("=" * 60)
    logger.info(f"  * Precision : {metrics['precision']:.2f}%")
    logger.info(f"  * Recall    : {metrics['recall']:.2f}%")
    logger.info(f"  * F1-Score  : {metrics['f1']:.2f}%")
    logger.info("-" * 60)
    logger.info(f"  * True Positives  (TP) : {metrics.get('tp', 0)}")
    logger.info(f"  * False Positives (FP) : {metrics.get('fp', 0)}")
    logger.info(f"  * False Negatives (FN) : {metrics.get('fn', 0)}")
    logger.info("=" * 60)

    # 保存错误日志分析
    error_analysis_file = os.path.join(config.exp_dir, "test_error_analysis.json")
    with open(error_analysis_file, "w", encoding="utf-8") as f:
        json.dump(error_cases, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(error_cases)} error cases to: {error_analysis_file}")

    return metrics


def main():
    # 1. 读取全局配置
    config = Config().parse_args()
    set_seed(config.seed)

    # 2. 日志初始化
    log_file = os.path.join(config.exp_dir, "test_evaluation.log")
    logger = get_logger(log_file)
    logger.info("================ Running Test Pipeline ================")

    # 3. 校验并定位测试数据缓存
    test_npz = os.path.join(config.data_dir, "test.npz")
    if not os.path.exists(test_npz):
        logger.error(f"Test data cache not found at: {test_npz}. Please run data_process.py first.")
        return

    test_loader = get_dataloader(test_npz, batch_size=config.batch_size, shuffle=False)

    # 4. 构建并初始化 PESL-GCN 模型
    logger.info("Initializing PESL-GCN Model Architecture...")
    model = PESL_GCN_Model(config).to(config.device)

    # 5. 加载预训练权重 Checkpoint
    checkpoint_path = os.path.join(config.exp_dir, "best_pesl_gcn_mlee.pt")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found at {checkpoint_path}. Please complete training first.")
        return

    logger.info(f"Loading Checkpoint Weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.device)

    # 兼容完整模型 Save 与 State_Dict Save
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        best_f1_in_ckpt = checkpoint.get("f1_score", "N/A")
        logger.info(f"Loaded Model Checkpoint (Saved Training F1: {best_f1_in_ckpt}%)")
    else:
        model.load_state_dict(checkpoint)

    # 6. 初始化评估计算引擎并执行测试
    evaluator = MetricsEvaluator(config.id2label)
    run_test_evaluation(model, test_loader, config, evaluator, logger)


if __name__ == "__main__":
    main()