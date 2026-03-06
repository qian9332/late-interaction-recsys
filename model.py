"""
模型定义 - 双塔推荐系统
包含: DCN v2编码器、双塔模型、Late-Interaction重排、可学习温度系数
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableTemperature(nn.Module):
    """
    可学习的InfoNCE温度系数
    使用sigmoid将参数约束在[min_temp, max_temp]范围内
    """
    
    def __init__(self, init_temp: float = 0.07, min_temp: float = 0.01, max_temp: float = 1.0):
        super().__init__()
        self.min_temp = min_temp
        self.max_temp = max_temp
        
        # 计算初始值对应的原始参数
        # temp = min_temp + (max_temp - min_temp) * sigmoid(raw_param)
        normalized_init = (init_temp - min_temp) / (max_temp - min_temp)
        normalized_init = max(0.01, min(0.99, normalized_init))  # 避免边界值
        raw_init = math.log(normalized_init / (1 - normalized_init))
        
        self.raw_param = nn.Parameter(torch.tensor(raw_init))
    
    def forward(self) -> torch.Tensor:
        """返回当前温度值"""
        normalized = torch.sigmoid(self.raw_param)
        temp = self.min_temp + (self.max_temp - self.min_temp) * normalized
        return temp
    
    def get_temperature(self) -> float:
        """获取当前温度值（Python float）"""
        return self.forward().item()


class CrossLayer(nn.Module):
    """
    DCN v2 交叉层
    显式建模特征交叉: x_{l+1} = x_0 ⊙ (W_l x_l + b_l) + x_l
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.weight = nn.Linear(input_dim, input_dim, bias=True)
    
    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return x0 * self.weight(x) + x


class CrossNetwork(nn.Module):
    """
    DCN v2 交叉网络
    堆叠多个交叉层，显式建模有界阶特征交叉
    """
    
    def __init__(self, input_dim: int, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        self.cross_layers = nn.ModuleList([
            CrossLayer(input_dim) for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x0 = x
        for cross_layer in self.cross_layers:
            x = cross_layer(x0, x)
        return x


class DeepNetwork(nn.Module):
    """
    DCN v2 深度网络
    标准的MLP结构，隐式建模高阶特征交叉
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(x)


class DCNv2Encoder(nn.Module):
    """
    DCN v2 编码器
    结合交叉网络和深度网络，显式+隐式建模特征交叉
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 num_cross_layers: int = 3, num_deep_layers: int = 3,
                 output_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        
        # 交叉网络
        self.cross_network = CrossNetwork(input_dim, num_cross_layers)
        
        # 深度网络
        deep_hidden_dims = [hidden_dim] * num_deep_layers
        self.deep_network = DeepNetwork(input_dim, deep_hidden_dims, dropout)
        
        # 组合层
        combined_dim = input_dim + hidden_dim
        self.combination_layer = nn.Sequential(
            nn.Linear(combined_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 交叉网络输出
        cross_output = self.cross_network(x)
        
        # 深度网络输出
        deep_output = self.deep_network(x)
        
        # 组合
        combined = torch.cat([cross_output, deep_output], dim=-1)
        output = self.combination_layer(combined)
        
        return output


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        batch_size = query.size(0)
        
        # 线性投影
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 注意力输出
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        return self.out_proj(attn_output)


class CrossAttentionLayer(nn.Module):
    """
    Cross-Attention层
    用于Late-Interaction重排，用户表示作为Query，候选物品作为Key和Value
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        # 自注意力 + 残差
        attn_output = self.attention(query, key, value, mask)
        query = self.norm1(query + attn_output)
        
        # FFN + 残差
        ffn_output = self.ffn(query)
        output = self.norm2(query + ffn_output)
        
        return output


class LateInteractionReranker(nn.Module):
    """
    Late-Interaction重排器
    在MIPS召回Top-K后，使用Cross-Attention进行精细重排
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 重排评分层
        self.score_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
    
    def forward(self, user_embedding: torch.Tensor, 
                candidate_embeddings: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            user_embedding: [batch_size, embed_dim] 用户嵌入
            candidate_embeddings: [batch_size, top_k, embed_dim] 候选物品嵌入
        Returns:
            scores: [batch_size, top_k] 重排分数
        """
        # 扩展用户嵌入作为Query
        batch_size, top_k, embed_dim = candidate_embeddings.shape
        query = user_embedding.unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        # Cross-Attention重排
        for cross_attn_layer in self.cross_attention_layers:
            query = cross_attn_layer(query, candidate_embeddings, candidate_embeddings)
        
        # 计算重排分数
        # 使用注意力后的用户表示与候选物品计算相似度
        rerank_scores = torch.matmul(query, candidate_embeddings.transpose(-2, -1)).squeeze(1)
        
        return rerank_scores


class UserTower(nn.Module):
    """
    用户塔
    使用DCN v2编码器处理用户特征
    """
    
    def __init__(self, num_users: int, num_dense_features: int,
                 embed_dim: int = 64, hidden_dim: int = 256,
                 num_cross_layers: int = 3, num_deep_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        
        # 用户ID嵌入
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        
        # 输入维度
        input_dim = embed_dim + num_dense_features
        
        # DCN v2编码器
        self.encoder = DCNv2Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_cross_layers=num_cross_layers,
            num_deep_layers=num_deep_layers,
            output_dim=embed_dim,
            dropout=dropout
        )
    
    def forward(self, user_ids: torch.Tensor, dense_features: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 用户ID嵌入
        user_emb = self.user_embedding(user_ids)
        
        # 拼接特征
        x = torch.cat([user_emb, dense_features], dim=-1)
        
        # DCN v2编码
        output = self.encoder(x)
        
        return F.normalize(output, p=2, dim=-1)


class ItemTower(nn.Module):
    """
    物品塔
    使用DCN v2编码器处理物品特征
    """
    
    def __init__(self, num_items: int, num_genres: int, num_dense_features: int,
                 embed_dim: int = 64, hidden_dim: int = 256,
                 num_cross_layers: int = 3, num_deep_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        
        # 物品ID嵌入
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        
        # 类型嵌入
        self.genre_embedding = nn.Embedding(num_genres + 1, embed_dim // 2, padding_idx=0)
        
        # 输入维度
        input_dim = embed_dim + embed_dim // 2 + num_dense_features
        
        # DCN v2编码器
        self.encoder = DCNv2Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_cross_layers=num_cross_layers,
            num_deep_layers=num_deep_layers,
            output_dim=embed_dim,
            dropout=dropout
        )
    
    def forward(self, item_ids: torch.Tensor, dense_features: torch.Tensor,
                genre_ids: Optional[List[List[int]]] = None) -> torch.Tensor:
        """前向传播"""
        # 物品ID嵌入
        item_emb = self.item_embedding(item_ids)
        
        # 类型嵌入（多热编码平均）
        if genre_ids is not None:
            batch_size = item_ids.size(0)
            genre_emb_list = []
            for i, gids in enumerate(genre_ids):
                if len(gids) > 0:
                    gids_tensor = torch.LongTensor(gids).to(item_ids.device)
                    g_emb = self.genre_embedding(gids_tensor).mean(dim=0)
                else:
                    g_emb = torch.zeros(self.genre_embedding.embedding_dim, device=item_ids.device)
                genre_emb_list.append(g_emb)
            genre_emb = torch.stack(genre_emb_list, dim=0)
        else:
            genre_emb = torch.zeros(item_ids.size(0), self.genre_embedding.embedding_dim, device=item_ids.device)
        
        # 拼接特征
        x = torch.cat([item_emb, genre_emb, dense_features], dim=-1)
        
        # DCN v2编码
        output = self.encoder(x)
        
        return F.normalize(output, p=2, dim=-1)
    
    def get_all_embeddings(self, device: torch.device) -> torch.Tensor:
        """获取所有物品嵌入（用于Faiss索引）"""
        with torch.no_grad():
            all_item_ids = torch.arange(self.item_embedding.num_embeddings, device=device)
            all_embeddings = self.item_embedding(all_item_ids)
            return F.normalize(all_embeddings, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """
    双塔推荐模型
    包含用户塔、物品塔、可学习温度系数、Late-Interaction重排
    """
    
    def __init__(self, config, num_users: int, num_items: int, num_genres: int):
        super().__init__()
        
        self.config = config.model
        
        # 用户塔
        self.user_tower = UserTower(
            num_users=num_users,
            num_dense_features=4,  # avg_rating, std_rating, count, avg_hour
            embed_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            num_cross_layers=self.config.num_cross_layers,
            num_deep_layers=self.config.num_deep_layers,
            dropout=self.config.dropout
        )
        
        # 物品塔
        self.item_tower = ItemTower(
            num_items=num_items,
            num_genres=num_genres,
            num_dense_features=3,  # year, avg_rating, count
            embed_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            num_cross_layers=self.config.num_cross_layers,
            num_deep_layers=self.config.num_deep_layers,
            dropout=self.config.dropout
        )
        
        # 可学习温度系数
        self.temperature = LearnableTemperature(
            init_temp=self.config.temperature_init,
            min_temp=self.config.temperature_min,
            max_temp=self.config.temperature_max
        )
        
        # Late-Interaction重排器
        self.late_interaction_reranker = LateInteractionReranker(
            embed_dim=self.config.embedding_dim,
            num_heads=self.config.num_attention_heads,
            num_layers=self.config.num_cross_attention_layers,
            dropout=self.config.dropout
        )
    
    def forward(self, batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 移动数据到设备
        user_ids = batch['user_ids'].to(device)
        user_dense = batch['user_dense_features'].to(device)
        item_ids = batch['item_ids'].to(device)
        item_dense = batch['item_dense_features'].to(device)
        context = batch['context_features'].to(device)
        neg_item_ids = batch['neg_item_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # 用户塔
        user_emb = self.user_tower(user_ids, user_dense)
        
        # 正样本物品塔
        pos_item_emb = self.item_tower(item_ids, item_dense, batch['item_genre_ids'])
        
        # 负样本物品塔
        batch_size = user_ids.size(0)
        neg_item_emb_list = []
        for i in range(neg_item_ids.size(1)):
            neg_emb = self.item_tower(neg_item_ids[:, i], item_dense, batch['item_genre_ids'])
            neg_item_emb_list.append(neg_emb)
        neg_item_emb = torch.stack(neg_item_emb_list, dim=1)  # [batch, num_neg, embed_dim]
        
        # 计算正样本分数
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=-1)  # [batch]
        
        # 计算负样本分数
        neg_scores = torch.bmm(neg_item_emb, user_emb.unsqueeze(-1)).squeeze(-1)  # [batch, num_neg]
        
        # InfoNCE损失
        temperature = self.temperature()
        logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1) / temperature
        labels_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, labels_idx)
        
        return {
            'loss': loss,
            'user_emb': user_emb,
            'pos_item_emb': pos_item_emb,
            'neg_item_emb': neg_item_emb,
            'pos_scores': pos_scores,
            'neg_scores': neg_scores,
            'temperature': temperature
        }
    
    def get_user_embedding(self, user_ids: torch.Tensor, user_dense: torch.Tensor) -> torch.Tensor:
        """获取用户嵌入"""
        return self.user_tower(user_ids, user_dense)
    
    def get_item_embedding(self, item_ids: torch.Tensor, item_dense: torch.Tensor,
                          genre_ids: Optional[List[List[int]]] = None) -> torch.Tensor:
        """获取物品嵌入"""
        return self.item_tower(item_ids, item_dense, genre_ids)
    
    def late_interaction_rerank(self, user_emb: torch.Tensor, 
                                candidate_embs: torch.Tensor) -> torch.Tensor:
        """Late-Interaction重排"""
        return self.late_interaction_reranker(user_emb, candidate_embs)
    
    def get_temperature_value(self) -> float:
        """获取当前温度值"""
        return self.temperature.get_temperature()


def build_model(config, num_users: int, num_items: int, num_genres: int) -> TwoTowerModel:
    """构建模型"""
    model = TwoTowerModel(config, num_users, num_items, num_genres)
    return model
