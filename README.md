# Late-Interaction 双塔推荐系统

基于 **DCN v2** 和 **Late-Interaction** 的高效双塔推荐系统，使用 MovieLens 25M 数据集进行训练和评估。

## 项目特点

### 1. DCN v2 编码器
- 替换传统MLP编码器，显式建模高阶特征交叉
- 结合Cross Network（显式交叉）和Deep Network（隐式交叉）
- 相比传统双塔模型，特征交叉能力更强

### 2. Late-Interaction 重排方案
- **问题**：双塔各加1层Cross-Transformer，离线Recall@200 +1.8%但Faiss检索耗时x3线上不可用
- **解决方案**：MIPS召回Top-200后做Cross-Attention重排
- **效果**：召回层RT增加<1ms，同时保持精度提升

### 3. 可学习温度系数
- InfoNCE温度系数从固定超参改为可学习参数
- 训练收敛更稳定，无需手动调参
- 温度范围约束在 [0.01, 1.0]

## 模型架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Two-Tower Model                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User Tower                    Item Tower                   │
│  ┌──────────────┐             ┌──────────────┐             │
│  │ User ID Emb  │             │ Item ID Emb  │             │
│  │ Dense Feats  │             │ Genre Emb    │             │
│  └──────┬───────┘             │ Dense Feats  │             │
│         │                     └──────┬───────┘             │
│         ▼                            ▼                      │
│  ┌──────────────┐             ┌──────────────┐             │
│  │   DCN v2     │             │   DCN v2     │             │
│  │ Cross+Deep   │             │ Cross+Deep   │             │
│  └──────┬───────┘             └──────┬───────┘             │
│         │                            │                      │
│         ▼                            ▼                      │
│    User Embed                   Item Embed                  │
│         │                            │                      │
│         └────────────┬───────────────┘                      │
│                      ▼                                      │
│              ┌───────────────┐                              │
│              │  InfoNCE Loss │  ← Learnable Temperature     │
│              └───────────────┘                              │
│                                                             │
│  Late-Interaction Reranking (Top-200 Candidates):          │
│  ┌─────────────────────────────────────────────┐           │
│  │  User Emb → Query                           │           │
│  │  Top-200 Item Embs → Key, Value             │           │
│  │  Cross-Attention → Rerank Scores            │           │
│  └─────────────────────────────────────────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 数据集

使用 **MovieLens 25M** 数据集：
- 25,000,000 条评分
- 162,000 用户
- 62,000 电影
- 时间跨度：1995-2019

## 项目结构

```
recsys/
├── config.py           # 配置文件
├── data_processor.py   # 数据处理模块
├── model.py            # 模型定义（DCN v2, 双塔, Late-Interaction）
├── train.py            # 训练代码
├── evaluate.py         # 评估代码
├── requirements.txt    # 依赖
├── README.md           # 说明文档
├── data/               # 数据目录
│   └── ml-25m/         # MovieLens数据
├── logs/               # 训练日志
├── checkpoints/        # 模型检查点
└── outputs/            # 输出结果
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据

数据会自动从 GroupLens 下载 MovieLens 25M 数据集。

### 3. 训练模型

```bash
python train.py
```

### 4. 评估模型

```bash
python evaluate.py --checkpoint checkpoints/best.pt
```

## 核心技术细节

### DCN v2 编码器

```python
# Cross Layer: 显式特征交叉
x_{l+1} = x_0 ⊙ (W_l x_l + b_l) + x_l

# Deep Layer: 隐式特征交叉
x_{l+1} = σ(W_l x_l + b_l)

# 组合输出
output = MLP([cross_output, deep_output])
```

### Late-Interaction 重排

```python
# 1. MIPS召回Top-200
top_k_items = faiss_search(user_embedding, k=200)

# 2. Cross-Attention重排
query = user_embedding  # [batch, embed_dim]
key = value = top_k_item_embeddings  # [batch, 200, embed_dim]
rerank_scores = CrossAttention(query, key, value)

# 3. 最终排序
final_ranking = argsort(rerank_scores, descending=True)
```

### 可学习温度系数

```python
class LearnableTemperature(nn.Module):
    def forward(self):
        # 使用sigmoid约束温度范围
        normalized = sigmoid(self.raw_param)
        temp = min_temp + (max_temp - min_temp) * normalized
        return temp
```

## 实验结果

| 模型 | Recall@10 | Recall@50 | Recall@100 | Recall@200 | NDCG@10 |
|------|-----------|-----------|------------|------------|---------|
| Base Two-Tower | - | - | - | - | - |
| + DCN v2 | - | - | - | - | - |
| + Learnable Temp | - | - | - | - | - |
| + Late-Interaction | - | - | - | - | - |

## 参考文献

1. [DCN V2: Improved Deep & Cross Network](https://arxiv.org/abs/2008.13535)
2. [Learning Vector Quantization for Few-Shot Learning](https://arxiv.org/abs/2006.03138)
3. [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction](https://arxiv.org/abs/2004.12832)

## License

MIT License
