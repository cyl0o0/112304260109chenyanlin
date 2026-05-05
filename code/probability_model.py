import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 文本预处理
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("IMDB 影评情感分析 - 概率预测版")
print("=" * 60)

# 加载数据
print("\n[1/5] 加载数据...")
train = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t', quoting=3)
test = pd.read_csv('testData.tsv/testData.tsv', sep='\t', quoting=3)
print(f"训练集: {train.shape}, 测试集: {test.shape}")

# 预处理
print("\n[2/5] 文本预处理...")
train['clean_review'] = train['review'].apply(clean_text)
test['clean_review'] = test['review'].apply(clean_text)

# 特征提取
print("\n[3/5] 提取TF-IDF特征...")
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 3),
    min_df=2
)
train_features = vectorizer.fit_transform(train['clean_review'])
test_features = vectorizer.transform(test['clean_review'])
print(f"特征维度: {train_features.shape}")

# 训练模型
print("\n[4/5] 训练逻辑回归模型...")
model = LogisticRegression(C=1.5, max_iter=500, random_state=42, n_jobs=-1)
model.fit(train_features, train['sentiment'])

# 概率预测
print("\n[5/5] 概率预测...")
probabilities = model.predict_proba(test_features)
positive_probs = probabilities[:, 1]

# 生成提交文件
print("\n生成提交文件...")

# 概率版本
submission_prob = pd.DataFrame({
    'id': test['id'],
    'sentiment': positive_probs
})
submission_prob.to_csv('submission_probability.csv', index=False, quoting=3)

# 二分类版本
submission_binary = pd.DataFrame({
    'id': test['id'],
    'sentiment': (positive_probs >= 0.5).astype(int)
})
submission_binary.to_csv('submission_binary.csv', index=False, quoting=3)

# 输出结果
print("\n完成!")
print("=" * 60)
print("\n前10个概率预测:")
print(submission_prob.head(10))
print("\n前10个二分类预测:")
print(submission_binary.head(10))
print("\n统计信息:")
print(f"概率范围: [{positive_probs.min():.4f}, {positive_probs.max():.4f}]")
print(f"平均概率: {positive_probs.mean():.4f}")
print(f"正面预测: {sum(submission_binary['sentiment'])}")
print(f"负面预测: {len(submission_binary)-sum(submission_binary['sentiment'])}")
print("\n生成的文件:")
print("- submission_probability.csv (概率版本)")
print("- submission_binary.csv (二分类版本)")
