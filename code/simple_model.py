#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# 改进的文本预处理函数
def clean_text(text):
    # 1. 去除HTML标签
    text = re.sub(r'<.*?>', '', text)
    
    # 2. 转小写
    text = text.lower()
    
    # 3. 保留重要标点符号（用于情感分析）
    text = re.sub(r'[^a-zA-Z0-9\s!?]', ' ', text)
    
    # 4. 处理缩写和否定词
    contractions = {
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "won't": "will not",
        "can't": "cannot",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "hasn't": "has not",
        "haven't": "have not",
        "hadn't": "had not",
        "wouldn't": "would not",
        "shouldn't": "should not",
        "couldn't": "could not",
        "mightn't": "might not",
        "mustn't": "must not"
    }
    
    for contraction, full in contractions.items():
        text = text.replace(contraction, full)
    
    # 5. 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

print("=" * 60)
print("IMDB 影评情感分析 - Kaggle 竞赛 (概率预测版)")
print("=" * 60)

# 1. 加载数据
print("\n[1/7] 正在加载数据...")
train = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t', quoting=3)
test = pd.read_csv('testData.tsv/testData.tsv', sep='\t', quoting=3)

print(f"训练集大小: {train.shape}")
print(f"测试集大小: {test.shape}")

# 2. 文本预处理
print("\n[2/7] 正在预处理文本...")
train['clean_review'] = train['review'].apply(clean_text)
test['clean_review'] = test['review'].apply(clean_text)

print(f"预处理示例:")
print(f"原始: {train['review'][0][:100]}...")
print(f"处理后: {train['clean_review'][0][:100]}...")

# 3. 特征提取 (TF-IDF)
print("\n[3/7] 正在提取TF-IDF特征...")
vectorizer = TfidfVectorizer(
    max_features=15000,  # 增加特征数量
    ngram_range=(1, 3),  # 使用三元组
    min_df=2,
    max_df=0.95,
    sublinear_tf=True  # 使用次线性TF缩放
)

train_features = vectorizer.fit_transform(train['clean_review'])
test_features = vectorizer.transform(test['clean_review'])

print(f"特征维度: 训练集 {train_features.shape}, 测试集 {test_features.shape}")

# 4. 构建集成学习模型
print("\n[4/7] 正在构建集成学习模型...")

# 逻辑回归（主要用于概率预测）
lr = LogisticRegression(
    C=1.5,
    max_iter=500,
    random_state=42,
    n_jobs=-1
)

# 随机森林
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# 集成模型
ensemble = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf)],
    voting='soft',  # 软投票，使用概率
    n_jobs=-1
)

# 5. 训练模型
print("\n[5/7] 正在训练模型...")
ensemble.fit(train_features, train['sentiment'])

# 进行交叉验证评估
print("\n正在进行交叉验证评估...")
cv_scores = cross_val_score(ensemble, train_features, train['sentiment'], cv=3, scoring='roc_auc')
print(f"3折交叉验证 AUC: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

# 6. 概率预测
print("\n[6/7] 正在进行概率预测...")
probabilities = ensemble.predict_proba(test_features)

# 获取正面情感的概率
positive_probs = probabilities[:, 1]

print(f"\n预测概率范围: [{np.min(positive_probs):.4f}, {np.max(positive_probs):.4f}]")
print(f"预测概率分布:")
print(f"  低置信度(<0.2或>0.8): {np.sum((positive_probs < 0.2) | (positive_probs > 0.8))}个样本")
print(f"  中等置信度(0.2-0.8): {np.sum((positive_probs >= 0.2) & (positive_probs <= 0.8))}个样本")

# 7. 生成提交文件
print("\n[7/7] 正在生成提交文件...")

# 方案1：直接使用概率值（适用于某些竞赛）
submission_prob = pd.DataFrame({
    'id': test['id'],
    'sentiment': positive_probs
})

# 方案2：使用概率转换为更准确的二分类预测
# 使用0.5作为阈值，但也可以调整
submission_binary = pd.DataFrame({
    'id': test['id'],
    'sentiment': (positive_probs >= 0.5).astype(int)
})

# 保存两种方案
submission_prob.to_csv('submission_probability.csv', index=False, quoting=3)
submission_binary.to_csv('submission_binary.csv', index=False, quoting=3)

print("\n" + "=" * 60)
print("完成！")
print(f"\n已生成两个提交文件:")
print(f"1. submission_probability.csv - 概率预测版本")
print(f"2. submission_binary.csv - 二分类预测版本")
print(f"\n前10个预测结果（概率版）:")
print(submission_prob.head(10))
print(f"\n前10个预测结果（二分类版）:")
print(submission_binary.head(10))
print("=" * 60)

# 显示一些置信度高和低的样本
print("\n==== 高置信度预测示例 ====")
high_conf_idx = np.argsort(np.abs(positive_probs - 0.5))[-5:][::-1]
for idx in high_conf_idx:
    print(f"ID: {test['id'].iloc[idx]}, 概率: {positive_probs[idx]:.4f}, 情感: {'正面' if positive_probs[idx]>=0.5 else '负面'}")

print("\n==== 低置信度预测示例 ====")
low_conf_idx = np.argsort(np.abs(positive_probs - 0.5))[:5]
for idx in low_conf_idx:
    print(f"ID: {test['id'].iloc[idx]}, 概率: {positive_probs[idx]:.4f}, 情感: {'正面' if positive_probs[idx]>=0.5 else '负面'}")
