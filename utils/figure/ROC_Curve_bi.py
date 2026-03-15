import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os

# ================= 全局样式同步 =================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'bold'


def plot_multi_roc_with_threshold(output_png_path, curve_configs, plot_threshold_points=True):
    # 增加画布尺寸以匹配大字体
    plt.figure(figsize=(10, 9.5), dpi=300)
    plt.plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--', alpha=0.5)

    roc_results = []

    for cfg in curve_configs:
        if not os.path.exists(cfg['input_data_path']):
            print(f"⚠️ 跳过模型 {cfg['label']}，路径不存在。")
            continue

        df = pd.read_excel(cfg['input_data_path'])
        y_true, y_prob = df[cfg['y_true_col']].astype(int), df[cfg['prob_col']]

        fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
        roc_auc = auc(fpr, tpr)

        # 绘图：使用加粗线条 lw=4
        plt.plot(fpr, tpr, color=cfg['color'], lw=4,
                 label=f"{cfg['label']} (AUC = {roc_auc:.3f})")

        if plot_threshold_points:
            # 计算 0.5 阈值点
            y_pred = (y_prob >= cfg.get('threshold', 0.5)).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            fpr_t = fp / (fp + tn) if (fp + tn) > 0 else 0
            tpr_t = tp / (tp + fn) if (tp + fn) > 0 else 0

            # 增大星号大小 s=250
            plt.scatter(fpr_t, tpr_t, s=250, c=cfg['color'],
                        marker='*', edgecolors='white', linewidth=1.5, zorder=10)

            roc_results.append({
                "label": cfg['label'],
                "auc": roc_auc,
                "fpr": fpr_t,
                "tpr": tpr_t
            })

    # --- 样式同步修改 ---
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])

    # 坐标轴标签尺寸同步至 22
    plt.xlabel('1 - Specificity', fontsize=22, fontweight='bold')
    plt.ylabel('Sensitivity', fontsize=22, fontweight='bold')

    # 标题尺寸同步至 24
    plt.title('ROC Curves: Dual-Modal Comparison', fontsize=24, pad=25, fontweight='bold')

    # 刻度数字大小同步至 18
    plt.tick_params(axis='both', which='major', labelsize=18)

    # 图例尺寸同步至 16
    plt.legend(loc="lower right", frameon=True, fontsize=16)
    plt.grid(True, linestyle=':', alpha=0.5)

    plt.savefig(output_png_path, bbox_inches='tight')

    # 控制台输出指标检查
    print("\n" + "=" * 60)
    print(f"{'Modality':<25} | {'AUC':<10} | {'FPR':<10} | {'TPR':<10}")
    print("-" * 60)
    for r in roc_results:
        print(f"{r['label']:<25} | {r['auc']:.4f} | {r['fpr']:.4f} | {r['tpr']:.4f}")
    print("=" * 60)
    print(f"✅ 双模态 ROC 图片已保存: {output_png_path}")


if __name__ == "__main__":
    BASE_DIR = "/server03_data/LJC/datasets/zhongzhon2/zsh/Result/FINAL"
    OUTPUT_PATH = f"{BASE_DIR}/Figure/Dual-model_ROC.png"

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # 颜色统一：红、蓝、绿
    CURVE_CONFIGS = [
        {
            "input_data_path": f"{BASE_DIR}/Text_CT_Dual/OOF_Predictions.xlsx",
            "y_true_col": "true_label", "prob_col": "oof_val_prob",
            "color": "red", "label": "CT + Text"
        },
        {
            "input_data_path": f"{BASE_DIR}/WSI_CT_Dual/OOF_Predictions.xlsx",
            "y_true_col": "true_label", "prob_col": "oof_val_prob",
            "color": "blue", "label": "CT + WSI"
        },
        {
            "input_data_path": f"{BASE_DIR}/WSI_Text_Dual/OOF_Predictions.xlsx",
            "y_true_col": "true_label", "prob_col": "oof_val_prob",
            "color": "green", "label": "WSI + Text"
        }
    ]

    plot_multi_roc_with_threshold(OUTPUT_PATH, CURVE_CONFIGS)