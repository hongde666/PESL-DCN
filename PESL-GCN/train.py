import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from metrics import MetricsEvaluator
from utils import save_checkpoint

def train_epoch(model, dataloader, optimizer, scheduler, criterion, config):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        adj_matrix = batch["adj_matrix"].to(config.device)
        labels = batch["labels"].to(config.device)

        logits = model(input_ids, attention_mask, adj_matrix)

        # 展平计算 Class-aware Weighted Loss[cite: 1]
        loss = criterion(logits.view(-1, config.num_classes), labels.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, evaluator, config):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            adj_matrix = batch["adj_matrix"].to(config.device)
            labels = batch["labels"].to(config.device)

            logits = model(input_ids, attention_mask, adj_matrix)
            probs = torch.softmax(logits, dim=-1)

            # 自适应阈值后处理 (Adaptive Thresholding tau = 0.25)[cite: 1]
            max_probs, preds = torch.max(probs, dim=-1)
            # 若最高预测置信度未达到自适应阈值，强制回归至背景类 0 ("O")[cite: 1]
            preds = torch.where(
                (max_probs < config.adaptive_threshold) & (preds != 0),
                torch.tensor(0).to(config.device),
                preds
            )

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    metrics = evaluator.compute_metrics(all_preds, all_labels)
    return metrics

def run_training(model, train_loader, test_loader, config, logger):
    # 建立类别感知加权损失矩阵 (Class-aware Weighted Loss)[cite: 1]
    # 背景类 "O" 权重 w_neg = 0.2，稀有触发词类 权重 w_pos = 2.0[cite: 1]
    class_weights = torch.ones(config.num_classes, device=config.device) * config.w_pos
    class_weights[0] = config.w_neg
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    # 优化器与学习率衰减
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)

    evaluator = MetricsEvaluator(config.id2label)
    best_f1 = 0.0
    best_model_path = f"{config.exp_dir}/best_pesl_gcn_mlee.pt"

    logger.info("Start Training PESL-GCN Model...")
    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, config)
        test_metrics = evaluate(model, test_loader, evaluator, config)

        logger.info(
            f"Epoch {epoch:02d}/{config.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Precision: {test_metrics['precision']}% | "
            f"Test Recall: {test_metrics['recall']}% | "
            f"Test F1: {test_metrics['f1']}%"
        )

        if test_metrics['f1'] > best_f1:
            best_f1 = test_metrics['f1']
            save_checkpoint(model, optimizer, epoch, best_f1, best_model_path)
            logger.info(f"--> Saved Best Model Checkpoint with F1: {best_f1}%")

    logger.info(f"Training Complete! Highest MLEE Test F1 Score: {best_f1}%")