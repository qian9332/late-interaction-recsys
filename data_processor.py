"""
数据处理模块 - MovieLens 25M数据集处理
包含数据加载、预处理、特征工程、数据集划分
"""

import os
import pickle
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class MovieLensDataProcessor:
    """MovieLens数据处理器"""
    
    def __init__(self, config):
        self.config = config.data
        self.data_dir = config.data.data_dir
        
        # 编码器
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.genre_encoder = LabelEncoder()
        
        # 特征统计
        self.num_users = 0
        self.num_items = 0
        self.num_genres = 0
        
        # 数据统计
        self.stats = {}
        
        # 数据采样比例（使用全部数据）
        self.sample_ratio = getattr(config.data, 'sample_ratio', 1.0)
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载原始数据"""
        print("Loading MovieLens 25M dataset...")
        
        # 加载评分数据 - 使用分块读取减少内存
        ratings_path = os.path.join(self.data_dir, "ratings.csv")
        
        # 读取全部数据
        ratings_df = pd.read_csv(ratings_path)
        print(f"Ratings shape: {ratings_df.shape}")
        
        # 加载电影数据
        movies_path = os.path.join(self.data_dir, "movies.csv")
        movies_df = pd.read_csv(movies_path)
        print(f"Movies shape: {movies_df.shape}")
        
        return ratings_df, movies_df
    
    def preprocess(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """数据预处理"""
        print("Preprocessing data...")
        start_time = time.time()
        
        # 创建副本避免警告
        ratings_df = ratings_df.copy()
        movies_df = movies_df.copy()
        
        # 1. 过滤低频用户和物品
        print("Filtering low-frequency users and items...")
        user_counts = ratings_df['userId'].value_counts()
        item_counts = ratings_df['movieId'].value_counts()
        
        valid_users = user_counts[user_counts >= self.config.min_user_interactions].index
        valid_items = item_counts[item_counts >= self.config.min_item_interactions].index
        
        ratings_df = ratings_df[
            ratings_df['userId'].isin(valid_users) & 
            ratings_df['movieId'].isin(valid_items)
        ].copy()
        print(f"After filtering: {ratings_df.shape[0]} ratings")
        
        # 1.5 数据采样（如果配置了采样比例）
        if hasattr(self.config, 'sample_ratio') and self.config.sample_ratio < 1.0:
            print(f"Sampling {self.config.sample_ratio*100:.1f}% of data...")
            # 按用户分层采样
            sample_size = int(len(ratings_df) * self.config.sample_ratio)
            ratings_df = ratings_df.sample(n=sample_size, random_state=42).copy()
            print(f"After sampling: {ratings_df.shape[0]} ratings")
        
        # 2. 编码用户和物品ID
        print("Encoding user and item IDs...")
        ratings_df['user_id'] = self.user_encoder.fit_transform(ratings_df['userId'])
        ratings_df['item_id'] = self.item_encoder.fit_transform(ratings_df['movieId'])
        
        # 只保留在ratings中出现的电影
        valid_movie_ids = ratings_df['movieId'].unique()
        movies_df = movies_df[movies_df['movieId'].isin(valid_movie_ids)].copy()
        movies_df['item_id'] = self.item_encoder.transform(movies_df['movieId'])
        
        # 3. 处理电影特征
        print("Processing movie features...")
        
        # 提取年份
        movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)', expand=False)
        movies_df['year'] = movies_df['year'].fillna(0).astype(int)
        movies_df['year_norm'] = (movies_df['year'] - movies_df['year'].min()) / (
            movies_df['year'].max() - movies_df['year'].min() + 1e-8
        )
        
        # 处理类型
        all_genres = set()
        for genres in movies_df['genres']:
            all_genres.update(genres.split('|'))
        all_genres = sorted(list(all_genres))
        self.genre_encoder.fit(all_genres)
        self.num_genres = len(all_genres)
        
        # 创建类型多热编码
        def encode_genres(genres_str):
            genres = genres_str.split('|')
            genre_ids = [self.genre_encoder.transform([g])[0] for g in genres if g in self.genre_encoder.classes_]
            return genre_ids
        
        movies_df['genre_ids'] = movies_df['genres'].apply(encode_genres)
        
        # 4. 处理评分数据
        # 评分归一化
        ratings_df['rating_norm'] = ratings_df['rating'] / 5.0
        
        # 时间戳转换 - 优化版本
        print("Processing timestamps...")
        timestamps = ratings_df['timestamp'].values
        hours = (timestamps // 3600) % 24  # 直接计算小时
        days = ((timestamps // (24 * 3600)) + 4) % 7  # 1970-01-01是周四
        ratings_df['hour'] = hours.astype(int)
        ratings_df['day_of_week'] = days.astype(int)
        
        # 时间特征归一化
        ratings_df['hour_norm'] = ratings_df['hour'] / 23.0
        ratings_df['day_norm'] = ratings_df['day_of_week'] / 6.0
        
        # 5. 计算用户统计特征 - 优化版本
        print("Computing user statistics...")
        # 使用更高效的聚合方式
        user_stats = ratings_df.groupby('user_id', sort=False).agg(
            user_avg_rating=('rating', 'mean'),
            user_std_rating=('rating', 'std'),
            user_count=('rating', 'count'),
            user_avg_hour=('hour', 'mean')
        ).reset_index()
        user_stats['user_std_rating'] = user_stats['user_std_rating'].fillna(0)
        
        # 归一化
        scaler = MinMaxScaler()
        user_stats[['user_avg_rating_norm', 'user_std_rating_norm', 'user_count_norm', 'user_avg_hour_norm']] = \
            scaler.fit_transform(user_stats[['user_avg_rating', 'user_std_rating', 'user_count', 'user_avg_hour']])
        
        # 使用map代替merge加速
        user_avg_map = dict(zip(user_stats['user_id'], user_stats['user_avg_rating_norm']))
        user_std_map = dict(zip(user_stats['user_id'], user_stats['user_std_rating_norm']))
        user_count_map = dict(zip(user_stats['user_id'], user_stats['user_count_norm']))
        user_hour_map = dict(zip(user_stats['user_id'], user_stats['user_avg_hour_norm']))
        
        ratings_df['user_avg_rating_norm'] = ratings_df['user_id'].map(user_avg_map)
        ratings_df['user_std_rating_norm'] = ratings_df['user_id'].map(user_std_map)
        ratings_df['user_count_norm'] = ratings_df['user_id'].map(user_count_map)
        ratings_df['user_avg_hour_norm'] = ratings_df['user_id'].map(user_hour_map)
        
        # 6. 计算物品统计特征 - 优化版本
        print("Computing item statistics...")
        item_stats = ratings_df.groupby('item_id', sort=False).agg(
            item_avg_rating=('rating', 'mean'),
            item_std_rating=('rating', 'std'),
            item_count=('rating', 'count')
        ).reset_index()
        item_stats['item_std_rating'] = item_stats['item_std_rating'].fillna(0)
        
        item_stats[['item_avg_rating_norm', 'item_std_rating_norm', 'item_count_norm']] = \
            scaler.fit_transform(item_stats[['item_avg_rating', 'item_std_rating', 'item_count']])
        
        # 使用map代替merge加速
        item_avg_map = dict(zip(item_stats['item_id'], item_stats['item_avg_rating_norm']))
        item_std_map = dict(zip(item_stats['item_id'], item_stats['item_std_rating_norm']))
        item_count_map = dict(zip(item_stats['item_id'], item_stats['item_count_norm']))
        
        ratings_df['item_avg_rating_norm'] = ratings_df['item_id'].map(item_avg_map)
        ratings_df['item_std_rating_norm'] = ratings_df['item_id'].map(item_std_map)
        ratings_df['item_count_norm'] = ratings_df['item_id'].map(item_count_map)
        
        movies_df['item_avg_rating_norm'] = movies_df['item_id'].map(item_avg_map)
        movies_df['item_std_rating_norm'] = movies_df['item_id'].map(item_std_map)
        movies_df['item_count_norm'] = movies_df['item_id'].map(item_count_map)
        
        # 更新统计信息
        self.num_users = ratings_df['user_id'].nunique()
        self.num_items = ratings_df['item_id'].nunique()
        
        # 保存统计信息
        self.stats = {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'num_genres': self.num_genres,
            'num_ratings': len(ratings_df),
            'avg_user_interactions': len(ratings_df) / self.num_users,
            'avg_item_interactions': len(ratings_df) / self.num_items,
            'rating_distribution': ratings_df['rating'].value_counts().to_dict(),
            'processing_time': time.time() - start_time
        }
        
        print(f"Preprocessing completed in {self.stats['processing_time']:.2f}s")
        print(f"Users: {self.num_users}, Items: {self.num_items}, Ratings: {self.stats['num_ratings']}")
        
        return ratings_df, movies_df
    
    def split_data(self, ratings_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """划分训练集和测试集"""
        print("Splitting data...")
        
        if self.config.time_series_split:
            # 按时间序列划分
            ratings_df = ratings_df.sort_values('timestamp')
            split_point = int(len(ratings_df) * self.config.train_ratio)
            train_df = ratings_df.iloc[:split_point]
            test_df = ratings_df.iloc[split_point:]
        else:
            # 随机划分
            train_df = ratings_df.sample(frac=self.config.train_ratio, random_state=42)
            test_df = ratings_df.drop(train_df.index)
        
        print(f"Train: {len(train_df)}, Test: {len(test_df)}")
        return train_df, test_df
    
    def build_user_history(self, train_df: pd.DataFrame) -> Dict[int, List[int]]:
        """构建用户历史交互"""
        user_history = train_df.groupby('user_id')['item_id'].apply(list).to_dict()
        return user_history
    
    def save_processed_data(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, 
                           train_df: pd.DataFrame, test_df: pd.DataFrame, save_dir: str):
        """保存处理后的数据"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存数据
        ratings_df.to_pickle(os.path.join(save_dir, 'ratings_processed.pkl'))
        movies_df.to_pickle(os.path.join(save_dir, 'movies_processed.pkl'))
        train_df.to_pickle(os.path.join(save_dir, 'train.pkl'))
        test_df.to_pickle(os.path.join(save_dir, 'test.pkl'))
        
        # 保存编码器和统计信息
        with open(os.path.join(save_dir, 'encoders.pkl'), 'wb') as f:
            pickle.dump({
                'user_encoder': self.user_encoder,
                'item_encoder': self.item_encoder,
                'genre_encoder': self.genre_encoder,
                'num_users': self.num_users,
                'num_items': self.num_items,
                'num_genres': self.num_genres
            }, f)
        
        with open(os.path.join(save_dir, 'stats.pkl'), 'wb') as f:
            pickle.dump(self.stats, f)
        
        print(f"Processed data saved to {save_dir}")
    
    def load_processed_data(self, save_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """加载处理后的数据"""
        ratings_df = pd.read_pickle(os.path.join(save_dir, 'ratings_processed.pkl'))
        movies_df = pd.read_pickle(os.path.join(save_dir, 'movies_processed.pkl'))
        train_df = pd.read_pickle(os.path.join(save_dir, 'train.pkl'))
        test_df = pd.read_pickle(os.path.join(save_dir, 'test.pkl'))
        
        with open(os.path.join(save_dir, 'encoders.pkl'), 'rb') as f:
            encoders = pickle.load(f)
            self.user_encoder = encoders['user_encoder']
            self.item_encoder = encoders['item_encoder']
            self.genre_encoder = encoders['genre_encoder']
            self.num_users = encoders['num_users']
            self.num_items = encoders['num_items']
            self.num_genres = encoders['num_genres']
        
        with open(os.path.join(save_dir, 'stats.pkl'), 'rb') as f:
            self.stats = pickle.load(f)
        
        return ratings_df, movies_df, train_df, test_df


class RecSysDataset(Dataset):
    """推荐系统数据集"""
    
    def __init__(self, df: pd.DataFrame, movies_df: pd.DataFrame, 
                 user_history: Dict[int, List[int]], num_items: int,
                 num_negatives: int = 4, is_training: bool = True):
        self.df = df
        self.movies_df = movies_df.set_index('item_id')
        self.user_history = user_history
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.is_training = is_training
        
        # 预计算物品特征
        self.item_features = self._build_item_features()
        
    def _build_item_features(self) -> Dict:
        """构建物品特征字典"""
        features = {}
        for item_id, row in self.movies_df.iterrows():
            features[item_id] = {
                'year_norm': row.get('year_norm', 0),
                'genre_ids': row.get('genre_ids', []),
                'avg_rating_norm': row.get('item_avg_rating_norm', 0.5),
                'count_norm': row.get('item_count_norm', 0.5)
            }
        return features
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        user_id = int(row['user_id'])
        item_id = int(row['item_id'])
        
        # 用户特征
        user_features = {
            'user_id': user_id,
            'avg_rating_norm': row.get('user_avg_rating_norm', 0.5),
            'std_rating_norm': row.get('user_std_rating_norm', 0),
            'count_norm': row.get('user_count_norm', 0.5),
            'avg_hour_norm': row.get('user_avg_hour_norm', 0.5)
        }
        
        # 物品特征
        item_feat = self.item_features.get(item_id, {
            'year_norm': 0,
            'genre_ids': [],
            'avg_rating_norm': 0.5,
            'count_norm': 0.5
        })
        
        item_features = {
            'item_id': item_id,
            'year_norm': item_feat['year_norm'],
            'genre_ids': item_feat['genre_ids'],
            'avg_rating_norm': item_feat['avg_rating_norm'],
            'count_norm': item_feat['count_norm']
        }
        
        # 上下文特征
        context_features = {
            'hour_norm': row.get('hour_norm', 0.5),
            'day_norm': row.get('day_norm', 0.5),
            'rating_norm': row.get('rating_norm', 0.5)
        }
        
        # 负采样
        if self.is_training:
            neg_items = self._sample_negatives(user_id, self.num_negatives)
        else:
            neg_items = []
        
        return {
            'user_features': user_features,
            'item_features': item_features,
            'context_features': context_features,
            'neg_items': neg_items,
            'label': 1.0
        }
    
    def _sample_negatives(self, user_id: int, num_neg: int) -> List[int]:
        """负采样"""
        history = set(self.user_history.get(user_id, []))
        negatives = []
        while len(negatives) < num_neg:
            neg = np.random.randint(0, self.num_items)
            if neg not in history and neg not in negatives:
                negatives.append(neg)
        return negatives


def collate_fn(batch):
    """自定义批处理函数"""
    user_ids = []
    user_dense_features = []
    
    item_ids = []
    item_dense_features = []
    item_genre_ids = []
    
    context_features = []
    neg_item_ids = []
    labels = []
    
    for sample in batch:
        uf = sample['user_features']
        user_ids.append(uf['user_id'])
        user_dense_features.append([
            uf['avg_rating_norm'],
            uf['std_rating_norm'],
            uf['count_norm'],
            uf['avg_hour_norm']
        ])
        
        itf = sample['item_features']
        item_ids.append(itf['item_id'])
        item_dense_features.append([
            itf['year_norm'],
            itf['avg_rating_norm'],
            itf['count_norm']
        ])
        item_genre_ids.append(itf['genre_ids'])
        
        cf = sample['context_features']
        context_features.append([
            cf['hour_norm'],
            cf['day_norm']
        ])
        
        neg_item_ids.append(sample['neg_items'])
        labels.append(sample['label'])
    
    return {
        'user_ids': torch.LongTensor(user_ids),
        'user_dense_features': torch.FloatTensor(user_dense_features),
        'item_ids': torch.LongTensor(item_ids),
        'item_dense_features': torch.FloatTensor(item_dense_features),
        'item_genre_ids': item_genre_ids,
        'context_features': torch.FloatTensor(context_features),
        'neg_item_ids': torch.LongTensor(neg_item_ids),
        'labels': torch.FloatTensor(labels)
    }


def create_dataloaders(config, processor: MovieLensDataProcessor, 
                       train_df: pd.DataFrame, test_df: pd.DataFrame,
                       movies_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
    """创建数据加载器"""
    user_history = processor.build_user_history(train_df)
    
    train_dataset = RecSysDataset(
        train_df, movies_df, user_history, processor.num_items,
        num_negatives=config.data.num_negatives, is_training=True
    )
    
    test_dataset = RecSysDataset(
        test_df, movies_df, user_history, processor.num_items,
        num_negatives=0, is_training=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval.eval_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, test_loader
