import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

# ===================== 配置路径 =====================
EXCEL_PATH = "/server03_data/LJC/datasets/zhongzhon2/zsh/Data/Preprocess_Data.xlsx"
OUTPUT_DIR = "/server03_data/LJC/datasets/zhongzhon2/zsh/Data/Data_splite"

# 根据你的表格结构定义列名
ID_COL = 'patho_id'
LABEL_COL = 'pCR'  # 你的表格中 0/1 标签所在列


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"📂 正在读取总表: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)

    # 1. 基础清洗
    # 确保 ID 和 标签 都不为空，否则无法进行分层抽样
    initial_count = len(df)
    df = df.dropna(subset=[ID_COL, LABEL_COL]).reset_index(drop=True)
    print(f"📊 原始数据 {initial_count} 条，清洗空值后参与分折样本: {len(df)} 条。")

    # 2. 初始化分层五折交叉验证
    # random_state=42 确保每次运行分折结果一致
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold = 1
    # skf.split 会根据 LABEL_COL (pCR) 的比例进行分层，确保每一折正负样本分布一致
    for train_index, val_index in skf.split(df, df[LABEL_COL]):
        df_train = df.iloc[train_index].reset_index(drop=True)
        df_val = df.iloc[val_index].reset_index(drop=True)

        # 统计当前折的正样本数量
        train_pos = (df_train[LABEL_COL] == 1).sum()
        val_pos = (df_val[LABEL_COL] == 1).sum()

        # 3. 构建 DataFrame
        # 训练集和验证集行数不等，pandas 会自动用 NaN 填充较短的那一列
        fold_out = pd.concat([
            pd.DataFrame({
                'train_patho_id': df_train[ID_COL],
                'train_label': df_train[LABEL_COL].astype(int)
            }),
            pd.DataFrame({
                'val_patho_id': df_val[ID_COL],
                'val_label': df_val[LABEL_COL].astype(int)
            })
        ], axis=1)

        # 4. 保存为 Excel
        fold_path = os.path.join(OUTPUT_DIR, f"fold{fold}.xlsx")
        fold_out.to_excel(fold_path, index=False)

        print(f"🚀 Fold {fold} 已生成:")
        print(f"   - 训练集: {len(df_train):>3} 条 (pCR+ : {train_pos:>2})")
        print(f"   - 验证集: {len(df_val):>3} 条 (pCR+ : {val_pos:>2})")
        fold += 1

    print(f"\n✅ 五折数据已全部保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()