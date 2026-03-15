import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_multimodal_dca(data_path, save_path):
    df = pd.read_excel(data_path)

    # 定义阈值范围
    thresholds = np.linspace(0.01, 0.99, 100)
    all_net_benefits = []

    # 分折计算净获益，然后取平均
    for f in range(1, 6):
        fold_df = df[df['val_fold_idx'] == f]
        y_true = fold_df['true_label'].values
        y_prob = fold_df['oof_val_prob'].values
        n = len(y_true)

        fold_nb = []
        for thresh in thresholds:
            tp = np.sum((y_prob >= thresh) & (y_true == 1))
            fp = np.sum((y_prob >= thresh) & (y_true == 0))
            # 净获益公式: NB = (TP/N) - (FP/N) * (Thresh / (1 - Thresh))
            nb = (tp / n) - (fp / n) * (thresh / (1 - thresh))
            fold_nb.append(nb)
        all_net_benefits.append(fold_nb)

    # 均值净获益
    mean_net_benefit = np.mean(all_net_benefits, axis=0)

    # 计算基准线 (基于总体患病率)
    prevalence = np.mean(df['true_label'].values)
    treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    treat_none = np.zeros_like(thresholds)

    # 绘图
    plt.figure(figsize=(10, 8), dpi=300)
    plt.plot(thresholds, mean_net_benefit, color='#9B59B6', lw=3, label='Tri-Modal Fusion Model')
    plt.plot(thresholds, treat_all, color='gray', linestyle='--', alpha=0.7, label='Treat All')
    plt.plot(thresholds, treat_none, color='black', lw=1, label='Treat None')

    # 样式微调
    plt.ylim([-0.05, prevalence + 0.1])  # 动态调整纵轴范围
    plt.xlim([0, 1.0])
    plt.xlabel('Threshold Probability', fontsize=14, fontweight='bold')
    plt.ylabel('Net Benefit', fontsize=14, fontweight='bold')
    plt.title('Decision Curve Analysis: Clinical Utility', fontsize=15, fontweight='bold', pad=15)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ DCA 曲线已保存至: {save_path}")

# 使用示例
# plot_multimodal_dca(TRI_MODAL_PATH, "./DCA_Overall_OOF.png")