"""
评估模块 - 双塔推荐系统
包含: Recall@K, NDCG@K, MRR@K, Faiss索引评估
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import faiss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from model import TwoTowerModel, build_model
from data_processor import MovieLensDataProcessor, create_dataloaders


class FaissIndex:
    """Faiss向量索引"""
    
    def __init__(self, embed_dim: int, index_type: str = "IVFFlat", 
                 nlist: int = 100, nprobe: int = 10):
        self.embed_dim = embed_dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.index = None
    
    def build(self, embeddings: np.ndarray):
        """构建索引"""
        if self.index_type == "IVFFlat":
            quantizer = faiss.IndexFlatIP(self.embed_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embed_dim, self.nlist)
            self.index.train(embeddings.astype(np.float32))
            self.index.add(embeddings.astype(np.float32))
            self.index.nprobe = self.nprobe
        elif self.index_type == "Flat":
            self.index = faiss.IndexFlatIP(self.embed_dim)
            self.index.add(embeddings.astype(np.float32))
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """搜索"""
        return self.index.search(query.astype(np.float32), k)


class Evaluator:
    """模型评估器"""
    
    def __init__(self, config: Config, model: TwoTowerModel, 
                 processor: MovieLensDataProcessor, test_loader):
        self.config = config
        self.model = model
        self.processor = processor
        self.test_loader = test_loader
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def compute_recall_at_k(self, ranked_indices: np.ndarray, 
                             ground_truth: np.ndarray, k: int) -> float:
        """计算Recall@K"""
        hits = 0
        for i, gt in enumerate(ground_truth):
            if gt in ranked_indices[i, :k]:
                hits += 1
        return hits / len(ground_truth)
    
    def compute_ndcg_at_k(self, ranked_indices: np.ndarray, 
                          ground_truth: np.ndarray, k: int) -> float:
        """计算NDCG@K"""
        ndcg = 0.0
        for i, gt in enumerate(ground_truth):
            if gt in ranked_indices[i, :k]:
                rank = np.where(ranked_indices[i] == gt)[0][0]
                ndcg += 1.0 / np.log2(rank + 2)
        return ndcg / len(ground_truth)
    
    def compute_mrr_at_k(self, ranked_indices: np.ndarray, 
                         ground_truth: np.ndarray, k: int) -> float:
        """计算MRR@K"""
        mrr = 0.0
        for i, gt in enumerate(ground_truth):
            if gt in ranked_indices[i, :k]:
                rank = np.where(ranked_indices[i] == gt)[0][0]
                mrr += 1.0 / (rank + 1)
        return mrr / len(ground_truth)
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """完整评估"""
        self.model.eval()
        
        print("Extracting embeddings...")
        
        # 提取所有物品嵌入
        all_item_embs = []
        for item_id in range(self.processor.num_items):
            item_emb = self.model.item_tower.item_embedding.weight[item_id]
            all_item_embs.append(item_emb.cpu().numpy())
        all_item_embs = np.array(all_item_embs)
        
        # 构建Faiss索引
        print("Building Faiss index...")
        faiss_index = FaissIndex(
            embed_dim=self.config.model.embedding_dim,
            index_type=self.config.eval.faiss_index_type,
            nlist=self.config.eval.faiss_nlist,
            nprobe=self.config.eval.faiss_nprobe
        )
        faiss_index.build(all_item_embs)
        
        # 评估
        print("Evaluating...")
        all_user_embs = []
        all_ground_truth = []
        
        for batch in self.test_loader:
            user_ids = batch['user_ids'].to(self.device)
            user_dense = batch['user_dense_features'].to(self.device)
            item_ids = batch['item_ids']
            
            user_emb = self.model.get_user_embedding(user_ids, user_dense)
            all_user_embs.append(user_emb.cpu().numpy())
            all_ground_truth.extend(item_ids.numpy().tolist())
        
        user_embs = np.vstack(all_user_embs)
        ground_truth = np.array(all_ground_truth)
        
        # Faiss搜索
        max_k = max(self.config.eval.recall_k)
        print(f"Searching top-{max_k} items for {len(user_embs)} users...")
        
        start_time = time.time()
        distances, indices = faiss_index.search(user_embs, max_k)
        search_time = time.time() - start_time
        
        print(f"Search time: {search_time:.3f}s for {len(user_embs)} queries")
        print(f"Avg search time per query: {search_time/len(user_embs)*1000:.3f}ms")
        
        # 计算指标
        metrics = {}
        
        for k in self.config.eval.recall_k:
            recall = self.compute_recall_at_k(indices, ground_truth, k)
            metrics[f'recall@{k}'] = recall
        
        for k in self.config.eval.ndcg_k:
            ndcg = self.compute_ndcg_at_k(indices, ground_truth, k)
            metrics[f'ndcg@{k}'] = ndcg
        
        for k in self.config.eval.mrr_k:
            mrr = self.compute_mrr_at_k(indices, ground_truth, k)
            metrics[f'mrr@{k}'] = mrr
        
        metrics['search_time'] = search_time
        metrics['avg_search_time_ms'] = search_time / len(user_embs) * 1000
        
        return metrics
    
    @torch.no_grad()
    def evaluate_late_interaction(self, top_k: int = 200) -> Dict[str, float]:
        """评估Late-Interaction重排效果"""
        self.model.eval()
        
        print("Extracting embeddings...")
        
        # 提取所有物品嵌入
        all_item_embs = self.model.item_tower.item_embedding.weight.cpu().numpy()
        all_item_embs = all_item_embs / (np.linalg.norm(all_item_embs, axis=1, keepdims=True) + 1e-8)
        
        # 构建Faiss索引
        print("Building Faiss index...")
        faiss_index = FaissIndex(
            embed_dim=self.config.model.embedding_dim,
            index_type="Flat"
        )
        faiss_index.build(all_item_embs)
        
        # 评估
        print("Evaluating Late-Interaction...")
        
        all_metrics = {
            'recall@10': [], 'recall@50': [], 'recall@100': [], 'recall@200': [],
            'ndcg@10': [], 'ndcg@50': [], 'ndcg@100': [],
            'mips_time': [], 'rerank_time': []
        }
        
        for batch in self.test_loader:
            user_ids = batch['user_ids'].to(self.device)
            user_dense = batch['user_dense_features'].to(self.device)
            item_ids = batch['item_ids'].numpy()
            
            # 用户嵌入
            user_emb = self.model.get_user_embedding(user_ids, user_dense)
            user_emb_np = user_emb.cpu().numpy()
            
            # MIPS召回
            start_time = time.time()
            distances, indices = faiss_index.search(user_emb_np, top_k)
            mips_time = time.time() - start_time
            
            # 获取候选物品嵌入
            batch_size = user_ids.size(0)
            candidate_embs = []
            for i in range(batch_size):
                cand_ids = indices[i]
                cand_emb = torch.from_numpy(all_item_embs[cand_ids]).to(self.device)
                candidate_embs.append(cand_emb)
            candidate_embs = torch.stack(candidate_embs, dim=0)
            
            # Late-Interaction重排
            start_time = time.time()
            rerank_scores = self.model.late_interaction_rerank(user_emb, candidate_embs)
            rerank_time = time.time() - start_time
            
            # 重新排序
            reranked_indices = torch.argsort(rerank_scores, dim=1, descending=True)
            final_indices = torch.gather(
                torch.from_numpy(indices).to(self.device), 
                1, 
                reranked_indices
            ).cpu().numpy()
            
            # 计算指标
            for i, gt in enumerate(item_ids):
                ranked = final_indices[i]
                
                for k in [10, 50, 100, 200]:
                    if gt in ranked[:k]:
                        all_metrics[f'recall@{k}'].append(1.0)
                    else:
                        all_metrics[f'recall@{k}'].append(0.0)
                
                for k in [10, 50, 100]:
                    if gt in ranked[:k]:
                        rank = np.where(ranked == gt)[0][0]
                        all_metrics[f'ndcg@{k}'].append(1.0 / np.log2(rank + 2))
                    else:
                        all_metrics[f'ndcg@{k}'].append(0.0)
            
            all_metrics['mips_time'].append(mips_time)
            all_metrics['rerank_time'].append(rerank_time)
        
        # 汇总
        metrics = {}
        for key, values in all_metrics.items():
            if key in ['mips_time', 'rerank_time']:
                metrics[key] = np.mean(values)
                metrics[f'{key}_per_query_ms'] = np.mean(values) / batch_size * 1000
            else:
                metrics[key] = np.mean(values)
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Two-Tower Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to config')
    parser.add_argument('--late-interaction', action='store_true', help='Evaluate Late-Interaction')
    args = parser.parse_args()
    
    # 加载配置
    config = Config()
    
    # 加载数据
    processor = MovieLensDataProcessor(config)
    processed_dir = os.path.join(config.data.data_dir, 'processed')
    
    if os.path.exists(processed_dir):
        ratings_df, movies_df, train_df, test_df = processor.load_processed_data(processed_dir)
    else:
        ratings_df, movies_df = processor.load_data()
        ratings_df, movies_df = processor.preprocess(ratings_df, movies_df)
        train_df, test_df = processor.split_data(ratings_df)
        processor.save_processed_data(ratings_df, movies_df, train_df, test_df, processed_dir)
    
    # 创建测试数据加载器
    from data_processor import RecSysDataset, collate_fn
    from torch.utils.data import DataLoader
    
    user_history = processor.build_user_history(train_df)
    test_dataset = RecSysDataset(
        test_df, movies_df, user_history, processor.num_items,
        num_negatives=0, is_training=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 加载模型
    model = build_model(
        config,
        num_users=processor.num_users,
        num_items=processor.num_items,
        num_genres=processor.num_genres
    )
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Checkpoint metrics: {checkpoint.get('metrics', {})}")
    
    # 评估
    evaluator = Evaluator(config, model, processor, test_loader)
    
    if args.late_interaction:
        print("\nEvaluating with Late-Interaction...")
        metrics = evaluator.evaluate_late_interaction()
    else:
        print("\nEvaluating...")
        metrics = evaluator.evaluate()
    
    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
    
    # 保存结果
    results_path = os.path.join(config.output_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
