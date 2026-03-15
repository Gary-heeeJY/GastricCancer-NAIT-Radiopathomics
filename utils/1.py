import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix


def calculate_metrics(file_path):
    # 1. 加载数据
    df = pd.read_excel(file_path)

    # 2. 提取真实标签和预测概率
    y_true = df['true_label'].values
    y_prob = df['oof_val_prob'].values
    # 3. 生成预测类别 (阈值 0.5)
    y_pred = (y_prob >= 0.5).astype(int)

    # 4. 计算指标
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # 混淆矩阵提取 TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # 5. 打印结果
    print("\n" + "=" * 50)
    print("🌟 总体盲测拼接总指标 (重新计算):")
    print("=" * 50)
    print(f"✅ AUC         : {auc:.4f}")
    print(f"✅ ACC         : {acc:.4f}")
    print(f"✅ F1          : {f1:.4f}")
    print(f"✅ SENSITIVITY : {sensitivity:.4f}")
    print(f"✅ SPECIFICITY : {specificity:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    FILE_PATH = "/server03_data/LJC/datasets/zhongzhon2/zsh/Result/FINAL/Bimodel/OOF_Predictions.xlsx"
    calculate_metrics(FILE_PATH)