import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os

# ================= 全局样式同步 =================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'bold'


def plot_tri_modal_roc_fold_avg(output_path, data_path, label_name="Tri-modal Model"):
    if not os.path.exists(data_path):
        print(f"❌ 错误：找不到文件 {data_path}")
        return

    df = pd.read_excel(data_path)

    # 五折平均计算 AUC (保留科研严谨性)
    fold_aucs = []
    for f in range(1, 6):
        fold_data = df[df['val_fold_idx'] == f]
        if len(fold_data) > 0:
            fpr_f, tpr_f, _ = roc_curve(fold_data['true_label'], fold_data['oof_val_prob'])
            fold_aucs.append(auc(fpr_f, tpr_f))

    avg_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)

    # 绘制总体 OOF 曲线
    fpr_all, tpr_all, _ = roc_curve(df['true_label'], df['oof_val_prob'])

    plt.figure(figsize=(10, 9.5), dpi=300)
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', alpha=0.5)

    # --- 统一修改点：颜色改为红色 red，线宽 lw=4 ---
    plt.plot(fpr_all, tpr_all, color='red', lw=4,
             label=f"{label_name}\n(AUC = {avg_auc:.3f})")

    # 0.5 阈值星号：颜色同步为 red，大小 s=250
    y_pred = (df['oof_val_prob'] >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(df['true_label'], y_pred).ravel()
    plt.scatter(fp / (fp + tn), tp / (tp + fn), s=250, color='red',
                marker='*', edgecolors='white', linewidth=1.5, zorder=10)

    # --- 视觉参数大统一（同步上一份代码的超大字体） ---
    plt.xlabel('1 - Specificity', fontsize=22, fontweight='bold')
    plt.ylabel('Sensitivity', fontsize=22, fontweight='bold')
    plt.title('ROC Curve: Tri-modal Analysis', fontsize=24, fontweight='bold', pad=25)

    # 刻度数字大小同步至 18
    plt.tick_params(axis='both', which='major', labelsize=18)
    # 图例尺寸同步至 16
    plt.legend(loc="lower right", fontsize=16, frameon=True)
    plt.grid(True, linestyle=':', alpha=0.5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"✅ 三模态 ROC 已保存: {output_path}")
    print(f"📊 监控指标: {avg_auc:.3f} ± {std_auc:.3f}")


if __name__ == "__main__":
    # 路径保持不变
    TRI_MODAL_DATA = "/server03_data/LJC/datasets/zhongzhon2/zsh/Result/FINAL/TriModal/OOF_Blind_Test_Predictions.xlsx"
    SAVE_PATH = "/server03_data/LJC/datasets/zhongzhon2/zsh/Result/FINAL/Figure/TriModal_ROC.png"

    plot_tri_modal_roc_fold_avg(SAVE_PATH, TRI_MODAL_DATA)