import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from scipy.interpolate import interp1d
import os

# ================= 配置区 =================
BASE_PATH = "/server03_data/LJC/datasets/zhongzhon2/zsh/Result/FINAL"
MODELS_PATHS = {
    "Trimodal": f"{BASE_PATH}/TriModal/OOF_Blind_Test_Predictions.xlsx",
    "Bimodal": f"{BASE_PATH}/WSI_CT_Dual/OOF_Predictions.xlsx",
    "Unimodal": f"{BASE_PATH}/WSI_Only/OOF_Predictions.xlsx"
}
SAVE_DIR = f"{BASE_PATH}/Figure"
os.makedirs(SAVE_DIR, exist_ok=True)

# 强制设置全局字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'bold'


def calculate_calibration_data(df, model_type='overall'):
    std_vals = np.linspace(0, 1, 100)
    if model_type == 'average':
        all_fractions = []
        for f in range(1, 6):
            fold_df = df[df['val_fold_idx'] == f]
            if len(fold_df) == 0: continue
            prob_true, prob_pred = calibration_curve(fold_df['true_label'], fold_df['oof_val_prob'], n_bins=10,
                                                     strategy='uniform')
            f_interp = interp1d(prob_pred, prob_true, bounds_error=False, fill_value=(0, 1))
            all_fractions.append(f_interp(std_vals))
        return std_vals, np.nanmean(all_fractions, axis=0)
    else:
        prob_true, prob_pred = calibration_curve(df['true_label'], df['oof_val_prob'], n_bins=10, strategy='uniform')
        f_interp = interp1d(prob_pred, prob_true, bounds_error=False, fill_value=(0, 1))
        return std_vals, f_interp(std_vals)


def calculate_dca_data(df, model_type='overall'):
    thresholds = np.linspace(0.01, 0.99, 100)
    if model_type == 'average':
        all_nb = []
        for f in range(1, 6):
            fold_df = df[df['val_fold_idx'] == f]
            y_true, y_prob = fold_df['true_label'].values, fold_df['oof_val_prob'].values
            n = len(y_true)
            nb = [(np.sum((y_prob >= t) & (y_true == 1)) / n) - (np.sum((y_prob >= t) & (y_true == 0)) / n) * (
                    t / (1 - t)) for t in thresholds]
            all_nb.append(nb)
        return thresholds, np.mean(all_nb, axis=0)
    else:
        y_true, y_prob = df['true_label'].values, df['oof_val_prob'].values
        n = len(y_true)
        nb = [(np.sum((y_prob >= t) & (y_true == 1)) / n) - (np.sum((y_prob >= t) & (y_true == 0)) / n) * (t / (1 - t))
              for t in thresholds]
        return thresholds, np.array(nb)


def plot_combined_evaluation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8.5), dpi=300)

    # 颜色统一：红、蓝、绿
    colors = {"Trimodal": "red", "Bimodal": "blue", "Unimodal": "green"}

    for name, path in MODELS_PATHS.items():
        if not os.path.exists(path): continue
        df = pd.read_excel(path)
        m_type = 'average' if name == "Trimodal" else 'overall'

        # 1. Calibration 计算与绘图
        x_cal, y_cal = calculate_calibration_data(df, m_type)
        mae = np.nanmean(np.abs(y_cal - x_cal))
        ax1.plot(x_cal, y_cal, color=colors[name], lw=4, label=f'{name} (MAE: {mae:.3f})')
        print(f"📈 {name} -> MAE: {mae:.4f}")

        # 2. DCA 计算与绘图
        x_dca, y_dca = calculate_dca_data(df, m_type)
        ax2.plot(x_dca, y_dca, color=colors[name], lw=4, label=name)

        # 打印 DCA 监控数值
        beneficial_idx = np.where(y_dca > 0)[0]  # 简化获益区间逻辑
        if len(beneficial_idx) > 0:
            print(f"🚀 {name} -> Max Net Benefit: {np.max(y_dca):.4f}")

    # --- A. Calibration 样式优化 ---
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal', lw=1.5)
    ax1.set_xlabel('Predicted Probability', fontsize=22, fontweight='bold')
    ax1.set_ylabel('Actual Probability', fontsize=22, fontweight='bold')
    ax1.set_title('Calibration Analysis', fontsize=24, fontweight='bold', pad=25)
    ax1.legend(loc='lower right', fontsize=16, frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.5)
    # 增大刻度数字大小
    ax1.tick_params(axis='both', which='major', labelsize=18)

    # --- B. DCA 样式优化 ---
    prevalence = 26 / 106
    threshold_range = np.linspace(0.01, 0.99, 100)
    treat_all = prevalence - (1 - prevalence) * (threshold_range / (1 - threshold_range))

    ax2.plot(threshold_range, treat_all, color='gray', linestyle='--', alpha=0.6, label='Treat All')
    ax2.axhline(y=0, color='black', lw=1.2, label='Treat None')

    ax2.set_xlim([0, 0.8])
    ax2.set_ylim([-0.05, 0.4])  # 稍微调高上限以适应加大的字体
    ax2.set_xlabel('Threshold Probability', fontsize=22, fontweight='bold')
    ax2.set_ylabel('Net Benefit', fontsize=22, fontweight='bold')
    ax2.set_title('Decision Curve Analysis', fontsize=24, fontweight='bold', pad=25)
    ax2.legend(loc='upper right', fontsize=16, frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.5)
    # 增大刻度数字大小
    ax2.tick_params(axis='both', which='major', labelsize=18)

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/Calibration_DCA_Combined.png", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_combined_evaluation()