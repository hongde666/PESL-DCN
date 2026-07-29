import os
from config import Config
from utils import set_seed, get_logger
from data_process import DataProcessor
from data_loader import get_dataloader
from model import PESL_GCN_Model
from train import run_training

def main():
    # 1. 加载配置并解析命令行参数
    config = Config().parse_args()
    set_seed(config.seed)

    # 2. 初始化日志
    log_file = os.path.join(config.exp_dir, "train.log")
    logger = get_logger(log_file)
    logger.info("Initializing PESL-GCN Execution Pipeline...")

    # 3. 检查数据与预处理
    train_npz = os.path.join(config.data_dir, "train.npz")
    test_npz = os.path.join(config.data_dir, "test.npz")

    if not (os.path.exists(train_npz) and os.path.exists(test_npz)):
        logger.info("Processed NPZ cache not found. Running DataProcessor to generate dependencies...")
        processor = DataProcessor(config)
        processor.convert_dataset_to_npz(
            json_file=os.path.join(config.data_dir, "train.json"),
            output_npz=train_npz
        )
        processor.convert_dataset_to_npz(
            json_file=os.path.join(config.data_dir, "test.json"),
            output_npz=test_npz
        )

    # 4. 构建 DataLoader
    train_loader = get_dataloader(train_npz, batch_size=config.batch_size, shuffle=True)
    test_loader = get_dataloader(test_npz, batch_size=config.batch_size, shuffle=False)

    # 5. 初始化模型
    logger.info(f"Loading SciBERT Backbone from: {config.pretrained_model_path}")
    model = PESL_GCN_Model(config).to(config.device)

    # 6. 开启训练流程
    run_training(model, train_loader, test_loader, config, logger)

if __name__ == "__main__":
    main()