"""
训练模块 - 双塔推荐系统
包含: 训练循环、早停机制、学习率调度、日志记录、模型保存
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.cuda.amp import GradScaler, autocast

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from model import TwoTowerModel, build_model
from data_processor import MovieLensDataProcessor, create_dataloaders


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 3, min_delta: float = 1e-4, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """检查是否应该早停"""
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: str, log_name: str = "training"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 日志文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{log_name}_{timestamp}.log")
        self.metrics_file = os.path.join(log_dir, f"metrics_{timestamp}.json")
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 指标历史
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'recall@10': [],
            'recall@50': [],
            'recall@100': [],
            'recall@200': [],
            'ndcg@10': [],
            'ndcg@50': [],
            'ndcg@100': [],
            'temperature': [],
            'learning_rate': [],
            'epoch_time': []
        }
    
    def log(self, message: str):
        """记录日志"""
        self.logger.info(message)
    
    def log_metrics(self, epoch: int, metrics: Dict):
        """记录指标"""
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        # 保存指标历史
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def log_config(self, config: Config):
        """记录配置"""
        config_dict = {
            'model': vars(config.model),
            'data': vars(config.data),
            'training': vars(config.training),
            'eval': vars(config.eval)
        }
        config_file = os.path.join(self.log_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)


class Trainer:
    """训练器"""
    
    def __init__(self, config: Config, model: TwoTowerModel, 
                 train_loader, test_loader, processor: MovieLensDataProcessor):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.processor = processor
        
        # 设备
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 日志记录器
        self.logger = TrainingLogger(config.log_dir)
        self.logger.log(f"Using device: {self.device}")
        self.logger.log_config(config)
        
        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # 学习率调度器
        num_training_steps = len(train_loader) * config.training.num_epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.training.learning_rate,
            total_steps=num_training_steps,
            pct_start=config.training.warmup_epochs / config.training.num_epochs
        )
        
        # 混合精度
        self.scaler = GradScaler() if config.training.use_amp else None
        
        # 早停
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            min_delta=config.training.early_stopping_min_delta,
            mode='max'
        )
        
        # 最佳模型
        self.best_score = 0.0
        self.best_epoch = 0
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 前向传播
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(batch, self.device)
                    loss = outputs['loss']
                
                # 反向传播
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch, self.device)
                loss = outputs['loss']
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.max_grad_norm
                )
                self.optimizer.step()
            
            self.scheduler.step()
            
            # 统计
            batch_size = batch['user_ids'].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # 日志
            if (batch_idx + 1) % self.config.training.log_interval == 0:
                avg_loss = total_loss / total_samples
                lr = self.scheduler.get_last_lr()[0]
                temp = self.model.get_temperature_value()
                self.logger.log(
                    f"Epoch {epoch} | Batch {batch_idx+1}/{len(self.train_loader)} | "
                    f"Loss: {avg_loss:.4f} | LR: {lr:.6f} | Temp: {temp:.4f}"
                )
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / total_samples
        
        return {
            'loss': avg_loss,
            'time': epoch_time,
            'lr': self.scheduler.get_last_lr()[0],
            'temperature': self.model.get_temperature_value()
        }
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        # 收集所有嵌入用于Recall计算
        all_user_embs = []
        all_item_embs = []
        all_user_ids = []
        all_item_ids = []
        
        for batch in self.test_loader:
            outputs = self.model(batch, self.device)
            loss = outputs['loss']
            
            batch_size = batch['user_ids'].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # 收集嵌入
            all_user_embs.append(outputs['user_emb'].cpu())
            all_item_embs.append(outputs['pos_item_emb'].cpu())
            all_user_ids.append(batch['user_ids'])
            all_item_ids.append(batch['item_ids'])
        
        avg_loss = total_loss / total_samples
        
        # 计算Recall@K和NDCG@K
        user_embs = torch.cat(all_user_embs, dim=0).numpy()
        item_embs = torch.cat(all_item_embs, dim=0).numpy()
        
        # 简化版评估：计算相似度矩阵
        # 由于数据量大，使用分批计算
        recall_metrics = {}
        ndcg_metrics = {}
        
        k_values = self.config.eval.recall_k
        batch_size_eval = 1000
        
        for k in k_values:
            recall_metrics[f'recall@{k}'] = 0.0
            ndcg_metrics[f'ndcg@{k}'] = 0.0
        
        num_users_eval = min(len(user_embs), 10000)  # 限制评估用户数
        
        for i in range(0, num_users_eval, batch_size_eval):
            batch_user_embs = user_embs[i:i+batch_size_eval]
            batch_item_embs = item_embs[i:i+batch_size_eval]
            
            # 计算相似度
            sim_matrix = np.dot(batch_user_embs, item_embs.T)
            
            for j in range(len(batch_user_embs)):
                user_idx = i + j
                scores = sim_matrix[j]
                
                # 排序
                ranked_indices = np.argsort(scores)[::-1]
                
                # 正样本位置
                pos_item_idx = user_idx  # 简化：假设对角线是正样本
                
                for k in k_values:
                    top_k_indices = ranked_indices[:k]
                    if pos_item_idx in top_k_indices:
                        recall_metrics[f'recall@{k}'] += 1.0
                    
                    # NDCG计算
                    pos_rank = np.where(ranked_indices == pos_item_idx)[0][0]
                    if pos_rank < k:
                        ndcg_metrics[f'ndcg@{k}'] += 1.0 / np.log2(pos_rank + 2)
        
        # 归一化
        for k in k_values:
            recall_metrics[f'recall@{k}'] /= num_users_eval
            ndcg_metrics[f'ndcg@{k}'] /= num_users_eval
        
        metrics = {
            'val_loss': avg_loss,
            **recall_metrics,
            **ndcg_metrics
        }
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': {
                'model': vars(self.config.model),
                'training': vars(self.config.training)
            }
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.config.checkpoint_dir, 'latest.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best.pt')
            torch.save(checkpoint, best_path)
            self.logger.log(f"Saved best model at epoch {epoch}")
        
        # 定期保存
        if epoch % self.config.training.save_interval == 0:
            epoch_path = os.path.join(self.config.checkpoint_dir, f'epoch_{epoch}.pt')
            torch.save(checkpoint, epoch_path)
    
    def train(self):
        """完整训练流程"""
        self.logger.log("=" * 60)
        self.logger.log("Starting Training")
        self.logger.log("=" * 60)
        self.logger.log(f"Dataset: {self.processor.stats}")
        self.logger.log(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.log(f"Device: {self.device}")
        self.logger.log("=" * 60)
        
        for epoch in range(1, self.config.training.num_epochs + 1):
            self.logger.log(f"\n{'='*60}")
            self.logger.log(f"Epoch {epoch}/{self.config.training.num_epochs}")
            self.logger.log("=" * 60)
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            self.logger.log(
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Time: {train_metrics['time']:.2f}s | "
                f"LR: {train_metrics['lr']:.6f} | "
                f"Temp: {train_metrics['temperature']:.4f}"
            )
            
            # 评估
            if epoch % self.config.training.eval_interval == 0:
                eval_metrics = self.evaluate()
                self.logger.log(
                    f"Val Loss: {eval_metrics['val_loss']:.4f} | "
                    f"Recall@10: {eval_metrics['recall@10']:.4f} | "
                    f"Recall@50: {eval_metrics['recall@50']:.4f} | "
                    f"Recall@100: {eval_metrics['recall@100']:.4f} | "
                    f"Recall@200: {eval_metrics['recall@200']:.4f}"
                )
                self.logger.log(
                    f"NDCG@10: {eval_metrics['ndcg@10']:.4f} | "
                    f"NDCG@50: {eval_metrics['ndcg@50']:.4f} | "
                    f"NDCG@100: {eval_metrics['ndcg@100']:.4f}"
                )
                
                # 记录指标
                self.logger.log_metrics(epoch, {
                    'train_loss': train_metrics['loss'],
                    'val_loss': eval_metrics['val_loss'],
                    'recall@10': eval_metrics['recall@10'],
                    'recall@50': eval_metrics['recall@50'],
                    'recall@100': eval_metrics['recall@100'],
                    'recall@200': eval_metrics['recall@200'],
                    'ndcg@10': eval_metrics['ndcg@10'],
                    'ndcg@50': eval_metrics['ndcg@50'],
                    'ndcg@100': eval_metrics['ndcg@100'],
                    'temperature': train_metrics['temperature'],
                    'learning_rate': train_metrics['lr'],
                    'epoch_time': train_metrics['time']
                })
                
                # 检查是否是最佳模型
                current_score = eval_metrics['recall@200']
                is_best = current_score > self.best_score
                if is_best:
                    self.best_score = current_score
                    self.best_epoch = epoch
                
                # 保存检查点
                self.save_checkpoint(epoch, eval_metrics, is_best)
                
                # 早停检查
                if self.early_stopping(current_score):
                    self.logger.log(f"Early stopping triggered at epoch {epoch}")
                    break
            else:
                # 只保存检查点
                self.save_checkpoint(epoch, {'train_loss': train_metrics['loss']})
        
        self.logger.log("\n" + "=" * 60)
        self.logger.log("Training Completed!")
        self.logger.log(f"Best Recall@200: {self.best_score:.4f} at epoch {self.best_epoch}")
        self.logger.log("=" * 60)
        
        return self.best_score, self.best_epoch


def main():
    """主函数"""
    # 配置
    config = Config()
    
    # 数据处理
    processor = MovieLensDataProcessor(config)
    
    # 检查是否有处理好的数据
    processed_dir = os.path.join(config.data.data_dir, 'processed')
    
    if os.path.exists(processed_dir):
        print("Loading processed data...")
        ratings_df, movies_df, train_df, test_df = processor.load_processed_data(processed_dir)
    else:
        print("Processing raw data...")
        ratings_df, movies_df = processor.load_data()
        ratings_df, movies_df = processor.preprocess(ratings_df, movies_df)
        train_df, test_df = processor.split_data(ratings_df)
        processor.save_processed_data(ratings_df, movies_df, train_df, test_df, processed_dir)
    
    # 创建数据加载器
    train_loader, test_loader = create_dataloaders(config, processor, train_df, test_df, movies_df)
    
    # 构建模型
    model = build_model(
        config,
        num_users=processor.num_users,
        num_items=processor.num_items,
        num_genres=processor.num_genres
    )
    
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练
    trainer = Trainer(config, model, train_loader, test_loader, processor)
    best_score, best_epoch = trainer.train()
    
    print(f"\nTraining finished! Best Recall@200: {best_score:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
