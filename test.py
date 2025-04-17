import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import joblib

# 设置matplotlib显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

# 加载模型和标准化器
voting_classifier = joblib.load('voting_classifier.pkl')
scaler = joblib.load('scaler.pkl')
print("模型和标准化器加载完成。")

# 加载测试集并预处理
df_test = pd.read_csv('creditcard.csv')
df_test = df_test.drop(['Time'], axis=1)  # 删除 'Time' 列
X_test = df_test.drop(['Class'], axis=1)  # 特征数据
y_test = df_test['Class'].values  # 标签数据

# 标准化测试集
X_test['Amount'] = scaler.transform(X_test[['Amount']])  # 使用训练集的scaler
X_test = X_test.values

# 预测
y_pred = voting_classifier.predict(X_test)

predictions_df = pd.DataFrame({'Predicted_Class': y_pred})
predictions_df.to_csv('predictions.csv', index=False)
print("预测结果已保存至 predictions.csv")

# 交叉验证
def cross_validation(model, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        # 训练模型
        model.fit(X_train_fold, y_train_fold)

        # 验证模型
        y_val_pred = model.predict(X_val_fold)
        y_val_proba = model.predict_proba(X_val_fold)[:, 1]

        # 计算评估指标
        precision = precision_score(y_val_fold, y_val_pred)
        recall = recall_score(y_val_fold, y_val_pred)
        f1 = f1_score(y_val_fold, y_val_pred)
        auc_score = auc(*roc_curve(y_val_fold, y_val_proba)[:2])

        cv_scores.append({
            'fold': fold + 1,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score
        })
    return cv_scores

# 交叉验证结果
cv_results = cross_validation(voting_classifier, X_test, y_test)
cv_df = pd.DataFrame(cv_results)
print("交叉验证结果：")
print(cv_df)

# 评估指标
print("========== 评估结果 ==========")
print(classification_report(y_test, y_pred, target_names=['正常', '欺诈']))
print(f"准确率：{accuracy_score(y_test, y_pred):.4f}")
print(f"精确率：{precision_score(y_test, y_pred):.4f}")
print(f"召回率：{recall_score(y_test, y_pred):.4f}")
print(f"F1分数：{f1_score(y_test, y_pred):.4f}")

# 计算ROC曲线和AUC
y_scores = voting_classifier.predict_proba(X_test)[:, 1]  # 获取预测概率
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)
print(f"ROC AUC：{roc_auc:.4f}")

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率')
plt.ylabel('真正率')
plt.title('ROC曲线')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['正常', '欺诈'], yticklabels=['正常', '欺诈'])
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("评估结果已保存至 roc_curve.png 和 confusion_matrix.png")