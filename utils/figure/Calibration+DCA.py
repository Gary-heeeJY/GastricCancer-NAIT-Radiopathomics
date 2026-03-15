import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import os

# ================= 配置区 =================
# 路径请根据你的实际服务器目录核对
BASE_PATH = "/server03_data/LJC/datasets/zhongzhon2/zsh/Result/FINAL"
MODELS = {
    "Single-WSI": f"{BASE_PATH}/WSI_Only/OOF_Predictions.xlsx",
    "Dual-CT+WSI": f"{BASE_PATH}/WSI_CT_Dual/OOF_Predictions.xlsx",
    "Tri-Modal": f"{BASE_PATH}/TriModal_new/OOF_Blind_Test_Predictions.xlsx"
}
SAVE_DIR = f"{BASE_PATH}/Figure"
os.makedirs(SAVE_DIR, exist_ok=True)

# 样式配置
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
COLORS = {"Single-WSI": "#3498DB", "Dual-CT+WSI": "#E67E22", "Tri-Modal": "#9B59B6"}  # 蓝、橙、紫


def plot_combined_clinical_evaluation(models_dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=300)

    thresholds = np.linspace(0.01, 0.99, 100)

    # 用于计算 Treat All 的基准（以 Tri-modal 的数据总量为准）
    sample_df = pd.read_excel(list(models_dict.values())[-1])
    prevalence = np.mean(sample_df['true_label'].values)
    treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))

    for name, path in models_dict.items():
        if not os.path.exists(path):
            print(f"⚠️ 跳过: 找不到文件 {path}")
            continue

        df = pd.read_excel(path)
        y_true = df['true_label'].values
        y_prob = df['oof_val_prob'].values

        # ---------------------------------------------------------
        # 1. 校准曲线 (使用总体拼接计算)
        # ---------------------------------------------------------
        # n_bins 设为 10 比较标准
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
        ax1.plot(prob_pred, prob_true, marker='o', markersize=4, color=COLORS[name], lw=2, label=name)

        # 打印 MAE 监控
        mae = np.mean(np.abs(np.interp(np.linspace(0, 1, 100), prob_pred, prob_true) - np.linspace(0, 1, 100)))
        print(f"📈 {name} -> 校准曲线 MAE: {mae:.4f}")

        # ---------------------------------------------------------
        # 2. 决策曲线分析 (使用总体拼接计算)
        # ---------------------------------------------------------
        n = len(y_true)
        net_benefits = []
        for thresh in thresholds:
            tp = np.sum((y_prob >= thresh) & (y_true == 1))
            fp = np.sum((y_prob >= thresh) & (y_true == 0))
            nb = (tp / n) - (fp / n) * (thresh / (1 - thresh))
            net_benefits.append(nb)

        ax2.plot(thresholds, net_benefits, color=COLORS[name], lw=3, label=name)
        print(f"🚀 {name} -> 最大净获益: {max(net_benefits):.4f}")

    # 校准曲线修饰
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated', alpha=0.7)
    ax1.set_xlabel('Predicted Probability', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Actual Probability', fontsize=14, fontweight='bold')
    ax1.set_title('A. Calibration Curves', fontsize=16, fontweight='bold', pad=15)
    ax1.legend(loc='lower right', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # DCA 修饰
    ax2.plot(thresholds, treat_all, color='gray', linestyle='--', alpha=0.6, label='Treat All')
    ax2.axhline(y=0, color='black', lw=1, label='Treat None')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([-0.05, prevalence + 0.15])
    ax2.set_xlabel('Threshold Probability', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Net Benefit', fontsize=14, fontweight='bold')
    ax2.set_title('B. Decision Curve Analysis', fontsize=16, fontweight='bold', pad=15)
    ax2.legend(loc='upper right', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    save_path = f"{SAVE_DIR}/Combined_Clinical_Evaluation.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"✅ 组合评估图已保存至: {save_path}")


if __name__ == "__main__":
    plot_combined_clinical_evaluation(MODELS)