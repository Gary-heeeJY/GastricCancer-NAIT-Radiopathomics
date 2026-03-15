import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os

# ================= 全局样式同步 =================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'bold'


def plot_multi_roc_with_threshold(output_png_path, curve_configs):
    # 增加画布尺寸
    plt.figure(figsize=(10, 9.5), dpi=300)
    plt.plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--', alpha=0.5)

    for cfg in curve_configs:
        if not os.path.exists(cfg['input_data_path']):
            print(f"⚠️ 跳过: {cfg['label']}")
            continue

        df = pd.read_excel(cfg['input_data_path']) if cfg['input_data_path'].endswith('.xlsx') else pd.read_csv(
            cfg['input_data_path'])
        y_true, y_prob = df[cfg['y_true_col']].astype(int), df[cfg['prob_col']]

        fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
        roc_auc = auc(fpr, tpr)

        # 绘图：同步线宽 lw=4
        plt.plot(fpr, tpr, color=cfg['color'], lw=4,
                 label=f"{cfg['label']} (AUC = {roc_auc:.3f})")

        # 绘制 0.5 阈值星号 s=250
        y_pred = (y_prob >= cfg.get('threshold', 0.5)).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        plt.scatter(fp / (fp + tn), tp / (tp + fn), s=250, c=cfg['color'],
                    marker='*', edgecolors='white', linewidth=1.5, zorder=10)

    # --- 视觉参数大统一 ---
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('1 - Specificity', fontsize=22, fontweight='bold')
    plt.ylabel('Sensitivity', fontsize=22, fontweight='bold')
    plt.title('ROC Curves: Single-Modal Comparison', fontsize=24, pad=25, fontweight='bold')

    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.legend(loc="lower right", frameon=True, fontsize=16)
    plt.grid(True, linestyle=':', alpha=0.5)

    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    plt.savefig(output_png_path, bbox_inches='tight')
    print(f"✅ 单模态 ROC 已保存: {output_png_path}")


if __name__ == "__main__":
    BASE_DIR = "/server03_data/LJC/datasets/zhongzhon2/zsh/Result/FINAL"
    OUTPUT_PATH = f"{BASE_DIR}/Figure/Single-Modal_ROC.png"

    CURVE_CONFIGS = [
        {"input_data_path": f"{BASE_DIR}/CT_Only/OOF_Predictions.xlsx", "y_true_col": "true_label",
         "prob_col": "oof_val_prob", "color": "red", "label": "CT Radiomics"},
        {"input_data_path": f"{BASE_DIR}/Text_Only/OOF_Predictions.xlsx", "y_true_col": "true_label",
         "prob_col": "oof_val_prob", "color": "blue", "label": "Pathology Text"},
        {"input_data_path": f"{BASE_DIR}/WSI_Only/OOF_Predictions.xlsx", "y_true_col": "true_label",
         "prob_col": "oof_val_prob", "color": "green", "label": "WSI (Pathology Image)"}
    ]
    plot_multi_roc_with_threshold(OUTPUT_PATH, CURVE_CONFIGS)