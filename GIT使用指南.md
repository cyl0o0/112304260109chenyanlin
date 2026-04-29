# Git & GitHub 使用指南

## 一、前期准备

### 1. 注册GitHub账号
1. 访问 https://github.com
2. 点击 "Sign up" 注册账号
3. 完成邮箱验证

### 2. 安装Git
1. 访问 https://git-scm.com/downloads
2. 下载Windows版本并安装
3. 安装完成后，在命令行输入 `git --version` 验证

## 二、本地Git配置

### 1. 配置用户名和邮箱
```bash
git config --global user.name "你的用户名"
git config --global user.email "你的邮箱"
```

### 2. 配置查看
```bash
git config --list
```

## 三、在GitHub创建仓库

1. 登录GitHub，点击右上角的 "+" → "New repository"
2. 仓库名称建议：`machine-learning-experiments` 或 `ml-lab3`
3. 设置为 Public 或 Private（建议Private）
4. 不要初始化 README（我们已经有了）
5. 点击 "Create repository"

## 四、连接本地项目到GitHub

### 方法一：第一次推送（推荐）

在 `d:\机器学习实验3` 目录下执行：

```bash
# 1. 初始化Git仓库
git init

# 2. 添加所有文件（除了data/目录，已在.gitignore中排除）
git add .

# 3. 第一次提交
git commit -m "实验3初始版本：IMDB情感分析，完成二分类和概率预测"

# 4. 连接远程仓库
git remote add origin https://github.com/你的用户名/你的仓库名.git

# 5. 推送到GitHub
git branch -M main
git push -u origin main
```

### 如果遇到认证问题
GitHub现在需要使用Personal Access Token（PAT）
1. 在GitHub点击头像 → Settings → Developer settings → Personal access tokens
2. 生成新token，权限至少需要 repo
3. 推送时使用token代替密码

## 五、实验后更新仓库（常用命令）

### 每次实验完成后，按以下步骤：

```bash
# 1. 查看当前状态
git status

# 2. 添加修改的文件
git add .

# 3. 提交（提交说明要写清楚！）
git commit -m "实验3-v2：改进模型，提高预测精度，修改了XXX"

# 4. 推送到GitHub
git push
```

### 提交说明的写法建议
✅ 好的提交说明：
- "实验3-v2：使用逻辑回归，增加特征到15000，添加概率预测"
- "修复文本预处理中的否定词处理bug"
- "添加实验报告草稿"

❌ 不好的提交说明：
- "update"
- "test"
- "123"

## 六、查看历史记录和回退版本

### 查看历史提交
```bash
git log
git log --oneline  # 简洁版
```

### 查看文件变更
```bash
git diff  # 查看未暂存的修改
git diff --staged  # 查看已暂存的修改
```

### 回退到之前版本（如果实验效果变差）

```bash
# 查看提交历史，找到要回退的commit ID
git log --oneline

# 假设要回退到a1b2c3d这个版本
# 方法一：撤销但保留修改
git reset a1b2c3d

# 方法二：彻底回退到该版本（慎用！）
git reset --hard a1b2c3d

# 回退后记得推送到GitHub
git push -f origin main
```

## 七、从GitHub克隆项目（如果换电脑）

```bash
git clone https://github.com/你的用户名/你的仓库名.git
```

## 八、当前项目结构说明

```
机器学习实验3/
├── code/                    # 代码文件
│   ├── imdb_sentiment.py   # 基础版本
│   ├── simple_model.py     # 简单版本
│   ├── probability_model.py # 概率预测版本（推荐）
│   └── 英文文本预处理注意事项.txt
├── data/                    # 数据集（不会上传GitHub）
│   ├── labeledTrainData.tsv/
│   ├── testData.tsv/
│   └── unlabeledTrainData.tsv/
├── report/                  # 实验报告
│   └── 实验报告模板.md
├── results/                 # 实验结果
│   ├── submission.csv
│   ├── submission_binary.csv
│   ├── submission_probability.csv
│   └── 实验记录.md
├── .gitignore               # Git忽略文件配置
├── README.md                # 项目说明
└── GIT使用指南.md          # 本文件
```

## 九、常见问题

### Q: 如何查看哪些文件被Git跟踪？
A: `git ls-files`

### Q: 如何取消暂存某个文件？
A: `git reset HEAD 文件名`

### Q: 如何撤销工作区的修改？
A: `git checkout -- 文件名`

### Q: 如何配置Git记住密码？
A: 使用SSH key或者 credential helper
```bash
git config --global credential.helper store
```

## 十、开始你的第一次提交吧！

按照第四步的操作，完成第一次Git提交，将你的实验上传到GitHub！🚀
