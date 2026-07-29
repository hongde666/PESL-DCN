import os
import random
import logging
import numpy as np
import torch

def set_seed(seed=42):
    """锁定随机种子保证实验完全可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_logger(log_file):
    """构建控制台与文件双流日志系统"""
    logger = logging.getLogger("PESL-GCN")
    logger.setLevel(logging.INFO)

    # 清除旧的 Handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # File Handler
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def save_checkpoint(model, optimizer, epoch, f1, save_path):
    """保存最佳模型权重矩阵"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'f1_score': f1
    }, save_path)