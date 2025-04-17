import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.over_sampling import SMOTE

# 加载数据
df = pd.read_csv('creditcard.csv')
print("数据读取完成。")

# 删除 'Time' 列
df = df.drop(['Time'], axis=1)
print("删除 'Time' 列完成。")

# 划分特征和标签
X = df.drop(['Class'], axis=1)  # 去除标签列
y = df['Class'].values  # 保留标签数据

# 划分训练集和测试集（先划分再处理！）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)  # 划分训练集和测试集，80%用于训练，20%用于测试，并且训练集和测试集中欺诈与非欺诈样本比例一致

# 标准化（仅在训练集标准化拟合）
scaler = StandardScaler()
X_train[['Amount']] = scaler.fit_transform(X_train[['Amount']])
X_test[['Amount']] = scaler.transform(X_test[['Amount']])
X_train = X_train.values
X_test = X_test.values

# 过采样：使用SMOTE平衡训练集
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"过采样后训练集样本数：{len(y_train_res)}")

# 训练随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_res, y_train_res)
print("随机森林分类器训练完成。")

# 训练逻辑回归分类器
lr_classifier = LogisticRegression(random_state=42)
lr_classifier.fit(X_train_res, y_train_res)
print("逻辑回归分类器训练完成。")

# 训练XGBoost分类器
xgb_classifier = XGBClassifier(random_state=42)
xgb_classifier.fit(X_train_res, y_train_res)
print("XGBoost分类器训练完成。")

# 创建投票分类器
voting_classifier = VotingClassifier(
    estimators=[
        ('rf', rf_classifier),
        ('lr', lr_classifier),
        ('xgb', xgb_classifier)
    ],
    voting='soft'  # 使用软投票，基于预测概率
)
voting_classifier.fit(X_train_res, y_train_res)
print("投票分类器训练完成。")

# 保存模型和标准化器
joblib.dump(voting_classifier, 'voting_classifier.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("模型和标准化器保存完成。")

