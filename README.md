# credit-card-fraud-detection
针对信用卡欺诈检测中存在的类别极度不平衡问题（欺诈样本仅占0.17%），提出基于集成学习之多模型投票的解决方案。通过 SMOTE 过采样技术处理数据不平衡，构建包含随机森林、逻辑回归和 XGBoost 的投票集成模型。结果表明，系统在测试集上达到召回率 97.9% 和 F1 分数 0.9451，较单一模型提升显著

相关结果详见result

## 数据加载与预处理 (train.py 和 test.py)

### 数据加载

使用 pandas 加载信用卡交易数据集 creditcard.csv。
数据集包含以下列：
Time：交易时间（从数据集中的第一笔交易开始的时间，单位为秒）。
Amount：交易金额。
Class：交易是否为欺诈（0 表示正常，1 表示欺诈）。
其他列是经过 PCA（主成分分析）处理的特征，匿名化以保护隐私。

删除无关列：

删除 Time 列，因为它对模型训练没有实际意义。
保留 Amount 列和 Class 列。

特征与标签分离：

特征集 X：所有列中去掉 Class 列。
标签集 y：Class 列。

### 数据划分与处理

划分训练集和测试集：
使用 train_test_split 将数据集划分为训练集和测试集，比例为 80%（训练集）和 20%（测试集）。
参数 stratify=y 确保训练集和测试集中欺诈与非欺诈样本的比例一致。

数据标准化 (train.py 和 test.py)：
标准化 Amount 列：
使用 StandardScaler 对 Amount 列进行标准化，使其均值为 0，标准差为 1。
训练集用于拟合标准化器（scaler.fit_transform），测试集仅进行转换（scaler.transform），以避免数据泄露。

处理类别不平衡 (train.py)：
过采样：使用 SMOTE（合成少数类过采样技术）对训练集中的欺诈样本进行过采样，平衡欺诈与非欺诈样本的数量。过采样后的训练集用于模型训练。

## 模型训练 (train.py)

单模型训练：
训练三种分类器：
随机森林分类器 (RandomForestClassifier)：
参数：n_estimators=100（100 棵树），random_state=42（随机种子）。
逻辑回归分类器 (LogisticRegression)：
参数：random_state=42。
XGBoost 分类器 (XGBClassifier)：
参数：random_state=42。
集成模型训练：
使用 VotingClassifier 将上述三种分类器组合为一个投票分类器。
参数：voting='soft'，表示基于预测概率的软投票。

## 模型评估（test.py）

### 评估指标：

使用以下指标评估模型性能：
分类报告 (classification_report)：精确率（Precision）、召回率（Recall）、F1 分数。
准确率 (accuracy_score)：整体预测准确率。
精确率 (precision_score)：预测为正类的样本中实际为正类的比例。
召回率 (recall_score)：实际为正类的样本中被正确预测为正类的比例。
F1 分数 (f1_score)：精确率和召回率的调和平均值。
ROC 曲线和 AUC：评估分类器的二分类性能。

### 交叉验证：

使用 StratifiedKFold 对测试集进行 5 折交叉验证，计算每折的精确率、召回率、F1 分数和 AUC。

### 可视化：

绘制 ROC 曲线并保存为 roc_curve.png。
绘制混淆矩阵并保存为 confusion_matrix.png

