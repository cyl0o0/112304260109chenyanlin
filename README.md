# 机器学习实验：基于 TF-IDF 的情感预测

## 1. 学生信息
- **姓名**：杨淋皓
- **学号**：112304260120
- **班级**：数据1231

> 注意：姓名和学号必须填写，否则本次实验提交无效。

---

## 2. 实验任务
本实验基于给定文本数据，使用 **TF-IDF 将文本转为向量特征**，再结合 **分类模型** 完成情感预测任务，并将结果提交到 Kaggle 平台进行评分。

本实验重点包括：
- 文本预处理
- TF-IDF 特征提取
- 分类模型训练
- 概率预测而非简单二分类
- Kaggle 结果提交与分析

---

## 3. 比赛与提交信息
- **比赛名称**：Bag of Words Meets Bags of Popcorn
- **比赛链接**：https://www.kaggle.com/c/word2vec-nlp-tutorial
- **提交日期**：2026-04-22

- **GitHub 仓库地址**：https://github.com/Ylh2004-shuju/112304260120yanglinhao
- **GitHub README 地址**：https://github.com/Ylh2004-shuju/112304260120yanglinhao/blob/main/README.md

> 注意：GitHub 仓库首页或 README 页面中，必须能看到"姓名 + 学号"，否则无效。

---

## 4. Kaggle 成绩
请填写你最终提交到 Kaggle 的结果：

- **Public Score**：0.96075
- **Private Score**（如有）：0.96075
- **排名**（如能看到可填写）：[待查看]

---

## 5. Kaggle 截图
请在下方插入 Kaggle 提交结果截图，要求能清楚看到分数信息。

![Kaggle截图](./images/112304260120_杨淋皓_kaggle_score.png)

> 建议将截图保存在 `images` 文件夹中。  
> 截图文件名示例：`112304260120_杨淋皓_kaggle_score.png`

---

## 6. 实验方法说明

### （1）文本预处理
请说明你对文本做了哪些处理，例如：
- 分词
- 去停用词
- 去除标点或特殊符号
- 转小写

**我的做法：**  
1. 去除HTML标签：使用正则表达式 `re.sub(r'<.*?>', '', text)` 移除影评中的HTML标签
2. 转换为小写：将所有文本转换为小写，统一处理 `text.lower()`
3. 保留重要标点：保留字母、数字、空格、感叹号（!）和问号（?），使用正则 `re.sub(r'[^a-zA-Z0-9\s!?]', ' ', text)`
4. 去除多余空格：清理文本中的多余空白字符 `re.sub(r'\s+', ' ', text).strip()`

---

### （2）TF-IDF 特征表示
请说明你如何使用 TF-IDF，例如：
- max_features 设置为多少
- ngram_range 设置
- 其他参数

**我的做法：**  
- 使用 scikit-learn 的 TfidfVectorizer 进行文本特征提取
- max_features: 15,000（保留最重要的15,000个特征）
- ngram_range: (1, 3)（使用1-gram、2-gram、3-gram特征）
- min_df: 2（至少在2个文档中出现的词才保留）

---

### （3）分类模型
请说明你使用了什么分类模型，例如：
- Logistic Regression
- Random Forest
- SVM
- XGBoost

并说明最终采用了哪一个模型。

**我的做法：**  
使用 **Logistic Regression**（最终采用）：
- C: 1.5（正则化强度）
- max_iter: 500
- random_state: 42（确保结果可复现）
- n_jobs: -1（使用所有CPU核心并行计算）
- 使用 predict_proba() 输出概率而非简单二分类，这样能提供更精细的预测结果，更适合Kaggle评分

最终采用 Logistic Regression 的概率预测版本，提交文件为 submission_probability.csv。

---

## 7. 实验流程
请简要说明你的实验流程。

示例：
1. 读取训练集和测试集  
2. 对文本进行预处理  
3. 训练或加载 Word2Vec 模型  
4. 将每条文本表示为句向量  
5. 用训练集训练分类器  
6. 在测试集上预测结果  
7. 生成 submission 文件并提交 Kaggle  

**我的实验流程：**  
1. 读取训练集（labeledTrainData.tsv）和测试集（testData.tsv）
2. 对文本进行预处理（去HTML标签、转小写、保留重要标点、清理多余空格）
3. 使用TF-IDF将文本转换为向量特征（15,000维，1-3 gram）
4. 使用训练集训练Logistic Regression分类器
5. 在测试集上进行概率预测（predict_proba获取正面情感概率）
6. 生成两个提交文件：
   - submission_probability.csv（包含真实概率值，用于Kaggle提交）
   - submission_binary.csv（基于概率的二分类结果，作为对比）
7. 提交 submission_probability.csv 到Kaggle，获得Public Score: 0.96075

---

## 8. 文件说明
请说明仓库中各文件或文件夹的作用。

示例：
- `data/`：存放数据文件
- `src/`：存放源代码
- `notebooks/`：存放实验 notebook
- `images/`：存放 README 中使用的图片
- `submission/`：存放提交文件

**我的项目结构：**
```text
机器学习实验3/
├─ code/                    # 实验代码
│  ├─ imdb_sentiment.py    # 基础版本代码
│  ├─ simple_model.py      # 简单二分类版本（Random Forest）
│  ├─ probability_model.py # 概率预测版本（Logistic Regression，推荐使用）
│  └─ 英文文本预处理注意事项.txt
├─ data/                    # 数据集（不上传到GitHub）
│  ├─ labeledTrainData.tsv/
│  ├─ testData.tsv/
│  └─ unlabeledTrainData.tsv/
├─ report/                  # 实验报告
│  ├─ 实验报告模板.md
│  └─ readme_机器学习实验2模板.md
├─ results/                 # 实验结果
│  ├─ submission.csv
│  ├─ submission_binary.csv
│  ├─ submission_probability.csv（Kaggle提交文件）
│  └─ 实验记录.md
├─ images/                  # 存放截图
│  └─ 112304260120_杨淋皓_kaggle_score.png（Kaggle成绩截图）
├─ .gitignore              # Git忽略文件配置
├─ README.md               # 本文件
└─ GIT使用指南.md          # Git使用教程
```
