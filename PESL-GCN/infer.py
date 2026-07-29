import os
import json
import torch
from tqdm import tqdm

from arguments import ModelArguments
from utils import set_seed, get_logger
from input_engineering.dataset_builder import build_dataloader
from model.pesl_gcn_builder import PESL_GCN_Network
from evaluation.metrics import EventMetricsEvaluator
from infer_module.pipeline import PESLInferencePipeline


def run_test_and_inference():
    # 1. 读取并解析命令行与 JSON 配置
    args = ModelArguments().parse()
    set_seed(getattr(args, "seed", 42))

    # 2. 初始化日志记录器
    log_file = os.path.join(args.exp_dir, "test_evaluation.log")
    logger = get_logger(log_file)
    logger.info("==========================================================")
    logger.info("             PESL-GCN INFERENCE & EVALUATION              ")
    logger.info("==========================================================")

    # 3. 加载测试数据集 DataLoader
    test_data_path = getattr(args, "test_data_path", os.path.join(args.data_dir, "test.npz"))
    if not os.path.exists(test_data_path):
        logger.error(f"Test dataset npz file not found at: {test_data_path}")
        return

    test_loader = build_dataloader(test_data_path, batch_size=args.batch_size, shuffle=False)

    # 4. 构建模型并加载训练好的检查点权重
    logger.info("Building PESL-GCN Network Architecture...")
    model = PESL_GCN_Network(args)

    checkpoint_path = getattr(args, "checkpoint_path", os.path.join(args.exp_dir, "best_pesl_gcn_mlee.pt"))
    if not os.path.exists(checkpoint_path):
        logger.error(f"Model Checkpoint file missing: {checkpoint_path}")
        return

    logger.info(f"Loading Checkpoint Weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Successfully loaded checkpoint (Saved Epoch F1: {checkpoint.get('f1_score', 'N/A')}%)")
    else:
        model.load_state_dict(checkpoint)

    model.to(args.device)

    # 5. 构建推理 Pipeline 与评估计算器
    evaluator = EventMetricsEvaluator(args.id2label)
    pipeline = PESLInferencePipeline(model, args, threshold=getattr(args, "adaptive_threshold", 0.25))

    # 6. 执行全量测试集推理
    logger.info("Executing Batch Inference Pipeline...")
    all_preds = []
    all_labels = []
    detailed_error_logs = []

    for batch in tqdm(test_loader, desc="Testing Progress"):
        preds_np = pipeline.predict_batch(batch)
        labels_np = batch["labels"].numpy()

        all_preds.extend(preds_np.tolist())
        all_labels.extend(labels_np.tolist())

        # 提取错例收集
        for b_i in range(len(preds_np)):
            for t_i in range(len(preds_np[b_i])):
                g_idx = labels_np[b_i][t_i]
                p_idx = preds_np[b_i][t_i]
                if g_idx != -100 and g_idx != p_idx:
                    detailed_error_logs.append({
                        "doc_id": batch["doc_id"][b_i],
                        "token_pos": t_i,
                        "gold_label": args.id2label.get(g_idx, "O"),
                        "pred_label": args.id2label.get(p_idx, "O")
                    })

    # 7. 评估性能与输出报告
    metrics = evaluator.compute_metrics(all_preds, all_labels)

    logger.info("=" * 60)
    logger.info("                   FINAL TEST METRICS                     ")
    logger.info("=" * 60)
    logger.info(f"  * Overall Precision : {metrics['precision']:.2f}%")
    logger.info(f"  * Overall Recall    : {metrics['recall']:.2f}%")
    logger.info(f"  * Overall F1-Score  : {metrics['f1']:.2f}%")
    logger.info("-" * 60)
    logger.info(f"  * TP: {metrics['tp']} | FP: {metrics['fp']} | FN: {metrics['fn']}")
    logger.info("=" * 60)

    # 8. 保存错例 JSON 报告供论文写 Case Study
    error_out_path = os.path.join(args.exp_dir, "test_error_cases.json")
    with open(error_out_path, "w", encoding="utf-8") as f:
        json.dump(detailed_error_logs, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(detailed_error_logs)} error cases for error analysis to: {error_out_path}")


if __name__ == "__main__":
    run_test_and_inference()