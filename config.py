"""
配置文件 - 双塔推荐系统 Late-Interaction方案
包含DCN v2编码器、可学习温度系数、Cross-Attention重排
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """模型配置"""
    # 嵌入维度
    embedding_dim: int = 64
    # 隐藏层维度
    hidden_dim: int = 256
    # DCN v2 交叉层数
    num_cross_layers: int = 3
    # DCN v2 深度层数
    num_deep_layers: int = 3
    # 注意力头数
    num_attention_heads: int = 4
    # Dropout率
    dropout: float = 0.2
    # 温度系数初始值
    temperature_init: float = 0.07
    # 温度系数最小值
    temperature_min: float = 0.01
    # 温度系数最大值
    temperature_max: float = 1.0
    # Late-Interaction Top-K
    late_interaction_topk: int = 200
    # Cross-Attention层数
    num_cross_attention_layers: int = 2


@dataclass
class DataConfig:
    """数据配置"""
    # 数据路径
    data_dir: str = "data/ml-25m"
    # 最小用户交互数（提高以过滤低活跃用户，保证数据质量）
    min_user_interactions: int = 50
    # 最小物品交互数（提高以过滤冷门物品，保证数据质量）
    min_item_interactions: int = 50
    # 训练集比例
    train_ratio: float = 0.8
    # 时间序列分割（按时间排序）
    time_series_split: bool = True
    # 负采样数量
    num_negatives: int = 4
    # 批次大小
    batch_size: int = 512
    # 数据加载线程数
    num_workers: int = 0
    # 数据采样比例（使用10%数据，约250万条，足够训练且内存可控）
    sample_ratio: float = 0.1


@dataclass
class TrainingConfig:
    """训练配置"""
    # 学习率
    learning_rate: float = 1e-3
    # 权重衰减
    weight_decay: float = 1e-5
    # 训练轮数
    num_epochs: int = 20
    # 早停耐心值
    early_stopping_patience: int = 3
    # 早停最小改进
    early_stopping_min_delta: float = 1e-4
    # 梯度裁剪
    max_grad_norm: float = 1.0
    # 学习率调度器
    lr_scheduler: str = "cosine"
    # 预热轮数
    warmup_epochs: int = 1
    # 混合精度训练
    use_amp: bool = False
    # 梯度累积步数
    gradient_accumulation_steps: int = 1
    # 日志间隔（步）
    log_interval: int = 100
    # 评估间隔（轮）
    eval_interval: int = 1
    # 保存间隔（轮）
    save_interval: int = 1


@dataclass
class EvalConfig:
    """评估配置"""
    # Recall@K
    recall_k: List[int] = field(default_factory=lambda: [10, 50, 100, 200])
    # NDCG@K
    ndcg_k: List[int] = field(default_factory=lambda: [10, 50, 100])
    # MRR@K
    mrr_k: List[int] = field(default_factory=lambda: [10, 50, 100])
    # 评估批次大小
    eval_batch_size: int = 256
    # Faiss索引类型
    faiss_index_type: str = "IVFFlat"
    # Faiss nlist
    faiss_nlist: int = 100
    # Faiss nprobe
    faiss_nprobe: int = 10


@dataclass
class Config:
    """总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    
    # 项目信息
    project_name: str = "late-interaction-recsys"
    seed: int = 42
    device: str = "cpu"
    
    # 路径配置
    output_dir: str = "outputs"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    
    def __post_init__(self):
        """初始化后处理"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)


def get_config():
    """获取配置"""
    return Config()
