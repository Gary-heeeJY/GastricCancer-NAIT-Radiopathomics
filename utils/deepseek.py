import os
import time
import hashlib
import logging
from typing import Dict, Optional

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


# =========================
# 配置区
# =========================

# 建议把 API Key 放到环境变量里：
# export DEEPSEEK_API_KEY="your_key"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-acaf9ed6b10f49198eb9f58cf300ff3e")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

input_file = '/server03_data/LJC/datasets/zhongzhon2/zsh/Data/Preprocess_Data.xlsx'
out_dir = '/server03_data/LJC/datasets/zhongzhon2/zsh/1. Deepseek_preprocess_text'

os.makedirs(out_dir, exist_ok=True)

output_excel = os.path.join(out_dir, 'standardized_texts.xlsx')
checkpoint_csv = os.path.join(out_dir, 'checkpoint.csv')


# =========================
# 初始化 DeepSeek API
# =========================

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

# 屏蔽不必要日志
logging.basicConfig(level=logging.ERROR)


# =========================
# 工具函数
# =========================

def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def build_record_id(row: pd.Series) -> str:
    """
    为每一条记录构造稳定的唯一ID。
    优先使用 patho_id / accession_no；
    如果二者缺失，再结合文本内容生成 hash。

    这样即使旧结果只有部分标识，也尽量能匹配到已处理病例。
    """
    patho_id = safe_str(row.get('patho_id', ''))
    accession_no = safe_str(row.get('accession_no', ''))
    patho_text = safe_str(row.get('patho_text', ''))
    history_text = safe_str(row.get('现病史', ''))

    # 优先使用更稳定的业务ID
    if patho_id:
        base = f"patho_id::{patho_id}"
    elif accession_no:
        base = f"accession_no::{accession_no}"
    else:
        # 退化到文本内容哈希
        raw = f"{patho_text}\n{history_text}"
        digest = hashlib.md5(raw.encode('utf-8')).hexdigest()
        base = f"text_hash::{digest}"

    return base


def build_combined_text(row: pd.Series) -> str:
    p_text = safe_str(row.get('patho_text', ''))
    h_text = safe_str(row.get('现病史', ''))

    return f"【病理诊断】:\n{p_text}\n\n【现病史】:\n{h_text}"


def has_content_to_process(row: pd.Series) -> bool:
    p_text = safe_str(row.get('patho_text', ''))
    h_text = safe_str(row.get('现病史', ''))
    return not (len(p_text) < 5 and len(h_text) < 5)


def standardize_report(report_text: str, index: int) -> str:
    if not isinstance(report_text, str) or not report_text.strip() or len(report_text) < 5:
        return "Empty input"

    try:
        prompt = f"""
Please extract and translate the following gastric cancer-related pathological and clinical records into English, and strictly standardize them into the following format:

[Clinical History]: ...
[Gross Description]: ...
[Microscopic Findings]: ...
[Diagnosis]: ...
[Additional Comments]: ...

Report Content:
{report_text}

Requirements:
1. Extract all key indicators, especially clinically relevant pathological information such as:
   - IHC / immunohistochemistry results (e.g., HER2, mismatch repair proteins, Ki-67, PD-L1, etc.)
   - Lauren classification
   - differentiation grade
   - invasion depth
   - lymph node metastasis
   - margin status
   - TNM-related expressions if available
2. Translate into accurate medical English.
3. If a section has no information, mark as "Not specified."
4. Output ONLY the standardized text, with no extra explanation.
"""
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            timeout=60
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"\nRow {index} Error: {str(e)}")
        return f"ERROR: {str(e)[:200]}"


def load_input_dataframe(file_path: str) -> pd.DataFrame:
    """
    读取原始输入表。
    当前输入是 xlsx，因此使用 read_excel。
    """
    print(f"正在读取原始文件: {file_path}")
    df = pd.read_excel(file_path)

    # 清洗列名
    df.columns = df.columns.map(lambda x: str(x).strip())

    return df


def load_existing_checkpoint(file_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(file_path):
        return None

    print(f"检测到历史 checkpoint: {file_path}")
    df_ckpt = pd.read_csv(file_path, encoding='utf-8-sig')
    df_ckpt.columns = df_ckpt.columns.map(lambda x: str(x).strip())

    if 'record_id' not in df_ckpt.columns:
        # 兼容旧版 checkpoint：现算 record_id
        df_ckpt['record_id'] = df_ckpt.apply(build_record_id, axis=1)

    if 'standardized_report' not in df_ckpt.columns:
        df_ckpt['standardized_report'] = pd.NA

    return df_ckpt


def load_existing_output(file_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(file_path):
        return None

    print(f"检测到历史 standardized_texts: {file_path}")
    df_out = pd.read_excel(file_path)
    df_out.columns = df_out.columns.map(lambda x: str(x).strip())

    # 尝试兼容旧表
    # standardized_texts.xlsx 通常包含 accession_no, patho_id, label, text
    if 'record_id' not in df_out.columns:
        # 尽量根据已有字段构造 record_id
        # 注意：旧输出通常没有原始文本，因此这里优先靠 patho_id/accession_no
        def _build_from_output(row):
            patho_id = safe_str(row.get('patho_id', ''))
            accession_no = safe_str(row.get('accession_no', ''))
            if patho_id:
                return f"patho_id::{patho_id}"
            elif accession_no:
                return f"accession_no::{accession_no}"
            else:
                return pd.NA

        df_out['record_id'] = df_out.apply(_build_from_output, axis=1)

    return df_out


def build_processed_map(df_checkpoint: Optional[pd.DataFrame],
                        df_output: Optional[pd.DataFrame]) -> Dict[str, str]:
    """
    从 checkpoint.csv 和 standardized_texts.xlsx 中收集已处理记录。
    优先使用 checkpoint 中的 standardized_report；
    output_excel 中则使用 text 列。
    """
    processed = {}

    if df_output is not None and len(df_output) > 0:
        if 'record_id' in df_output.columns and 'text' in df_output.columns:
            tmp = df_output[['record_id', 'text']].dropna(subset=['record_id', 'text']).copy()
            for _, row in tmp.iterrows():
                rid = safe_str(row['record_id'])
                txt = safe_str(row['text'])
                if rid and txt:
                    processed[rid] = txt

    if df_checkpoint is not None and len(df_checkpoint) > 0:
        if 'record_id' in df_checkpoint.columns and 'standardized_report' in df_checkpoint.columns:
            tmp = df_checkpoint[['record_id', 'standardized_report']].dropna(subset=['record_id', 'standardized_report']).copy()
            for _, row in tmp.iterrows():
                rid = safe_str(row['record_id'])
                txt = safe_str(row['standardized_report'])
                if rid and txt:
                    processed[rid] = txt

    return processed


def merge_and_save_checkpoint(current_df: pd.DataFrame, checkpoint_path: str):
    """
    将当前完整 df 保存回 checkpoint。
    这里不是丢弃旧数据，而是因为 current_df 已经包含旧数据 + 新数据，所以保存后是完整累积结果。
    """
    current_df.to_csv(checkpoint_path, index=False, encoding='utf-8-sig')


def merge_and_save_output(df_current: pd.DataFrame,
                          old_output: Optional[pd.DataFrame],
                          output_path: str):
    """
    将旧 standardized_texts.xlsx 与本次新结果合并，按 record_id 去重后保存。
    最终输出列：
      accession_no, patho_id, label, text, record_id
    """
    final_df = df_current.copy()

    # 统一字段
    rename_map = {}
    if 'standardized_report' in final_df.columns:
        rename_map['standardized_report'] = 'text'
    if 'pCR' in final_df.columns:
        rename_map['pCR'] = 'label'
    final_df = final_df.rename(columns=rename_map)

    if 'text' not in final_df.columns:
        final_df['text'] = pd.NA
    if 'label' not in final_df.columns:
        final_df['label'] = pd.NA
    if 'record_id' not in final_df.columns:
        final_df['record_id'] = final_df.apply(build_record_id, axis=1)

    cols_priority = ['accession_no', 'patho_id', 'label', 'text', 'record_id']
    current_export = final_df[[c for c in cols_priority if c in final_df.columns]].copy()

    # 清理 ERROR
    if 'text' in current_export.columns:
        current_export = current_export[~current_export['text'].astype(str).str.contains('ERROR', na=False)]

    if old_output is not None and len(old_output) > 0:
        old_keep_cols = [c for c in cols_priority if c in old_output.columns]
        old_export = old_output[old_keep_cols].copy()

        # 对齐列
        all_cols = []
        for c in cols_priority:
            if c in current_export.columns or c in old_export.columns:
                all_cols.append(c)

        current_export = current_export.reindex(columns=all_cols)
        old_export = old_export.reindex(columns=all_cols)

        merged = pd.concat([old_export, current_export], axis=0, ignore_index=True)

        if 'record_id' in merged.columns:
            merged = merged.drop_duplicates(subset=['record_id'], keep='last')
        elif 'patho_id' in merged.columns:
            merged = merged.drop_duplicates(subset=['patho_id'], keep='last')
        elif 'accession_no' in merged.columns:
            merged = merged.drop_duplicates(subset=['accession_no'], keep='last')

    else:
        merged = current_export.copy()

    merged.to_excel(output_path, index=False)

    return merged


# =========================
# 主流程
# =========================

print("=" * 60)
print("开始处理病理报告（支持增量续跑，避免重复调用 DeepSeek）")
print("=" * 60)

# 1. 读取输入数据
df = load_input_dataframe(input_file)

# 2. 生成 record_id
df['record_id'] = df.apply(build_record_id, axis=1)

# 3. 读取历史结果
df_checkpoint_old = load_existing_checkpoint(checkpoint_csv)
df_output_old = load_existing_output(output_excel)

# 4. 收集历史已处理文本
processed_map = build_processed_map(df_checkpoint_old, df_output_old)

# 5. 初始化 standardized_report
if 'standardized_report' not in df.columns:
    df['standardized_report'] = pd.NA

# 6. 先把历史已处理结果回填到当前 df
recovered_count = 0
for i in range(len(df)):
    rid = safe_str(df.at[i, 'record_id'])
    if rid in processed_map:
        old_text = processed_map[rid]
        if old_text:
            df.at[i, 'standardized_report'] = old_text
            recovered_count += 1

print(f"已从历史 checkpoint / standardized_texts 中恢复 {recovered_count} 条处理结果。")

# 7. 找到真正需要新处理的病例
pending_indices = []
for i in range(len(df)):
    existing_text = safe_str(df.at[i, 'standardized_report'])
    if existing_text:
        continue
    if not has_content_to_process(df.loc[i]):
        df.at[i, 'standardized_report'] = "No content to process"
        continue
    pending_indices.append(i)

print(f"当前总记录数: {len(df)}")
print(f"无需重复处理 / 已恢复: {len(df) - len(pending_indices)}")
print(f"待新增处理: {len(pending_indices)}")

# 8. 开始处理新增病例
progress_bar = tqdm(total=len(pending_indices), desc="处理中", unit="条")

for idx_in_list, i in enumerate(pending_indices):
    combined_text = build_combined_text(df.loc[i])

    standardized = standardize_report(combined_text, i)
    df.at[i, 'standardized_report'] = standardized

    # 每 5 条保存一次 checkpoint
    if (idx_in_list + 1) % 5 == 0 or idx_in_list == len(pending_indices) - 1:
        merge_and_save_checkpoint(df, checkpoint_csv)

    progress_bar.update(1)
    time.sleep(0.5)

progress_bar.close()

# 9. 最终保存 checkpoint（完整表）
merge_and_save_checkpoint(df, checkpoint_csv)

# 10. 合并旧 standardized_texts.xlsx 与当前结果，去重后保存
merged_output = merge_and_save_output(df, df_output_old, output_excel)

# 11. 输出总结
valid_count = 0
if 'text' in merged_output.columns:
    valid_count = merged_output['text'].notna().sum()

print("\n" + "=" * 60)
print("处理完成！")
print(f"最终有效数据条数: {valid_count}")
print(f"checkpoint 已更新至: {checkpoint_csv}")
print(f"standardized_texts 已更新至: {output_excel}")
print("=" * 60)