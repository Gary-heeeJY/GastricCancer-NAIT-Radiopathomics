import os
import re
import pandas as pd
import sys

# ===================== 配置参数 =====================
csv_file_path = "/server03_data/LJC/datasets/zhongzhon2/zsh/Data/病例信息.csv"
feature_root_dir = "/server03_data/LJC/datasets/zhongzhon2/zsh/2. pathology_feature_MUSK"
output_txt_path = "/server03_data/LJC/datasets/zhongzhon2/zsh/Data/特征与病理信息对比结果1.txt"
csv_encoding = "utf-8"


# ===================== 工具类：同步输出到文件 =====================
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# ===================== 工具函数：清洗与提取 =====================
def clean_id(s):
    """提取字符串中所有的数字部分并拼接"""
    if pd.isna(s): return ""
    return "".join(re.findall(r'\d', str(s)))


def get_core_id(s):
    """针对带下划线的ID：只取下划线前的数字部分"""
    s_str = str(s).strip()
    prefix = s_str.split("_")[0]
    return "".join(re.findall(r'\d', prefix))


# ===================== 主逻辑 =====================
def run_matching():
    # 初始化日志记录
    sys.stdout = Logger(output_txt_path)

    print("========== 开始执行病例ID与特征文件夹匹配 ==========")

    # 1. 读取并清洗病例ID
    try:
        df = pd.read_csv(csv_file_path, encoding=csv_encoding, usecols=["patho_id"])
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file_path, encoding='gbk', usecols=["patho_id"])

    df = df.dropna(subset=["patho_id"])
    raw_ids = df["patho_id"].astype(str).str.strip().unique().tolist()

    # 2. 遍历文件夹
    if not os.path.exists(feature_root_dir):
        print(f"❌ 错误：路径不存在 {feature_root_dir}")
        return
    all_folders = [f for f in os.listdir(feature_root_dir) if os.path.isdir(os.path.join(feature_root_dir, f))]

    # 3. 匹配执行
    case_matches = {}
    matched_folders = set()

    for rid in raw_ids:
        c1 = clean_id(rid)
        c2 = get_core_id(rid)

        matches = []
        for folder in all_folders:
            f_clean = clean_id(folder)
            # 兼容逻辑：全数字匹配 或 核心前缀匹配
            if (c1 and c1 in f_clean) or (c2 and c2 in f_clean):
                matches.append(folder)
                matched_folders.add(folder)
        case_matches[rid] = list(set(matches))

    # 分类结果
    no_feature_cases = [r for r in raw_ids if not case_matches[r]]
    no_case_folders = [f for f in all_folders if f not in matched_folders]

    # ===================== 格式化输出 =====================
    print("\n" + "=" * 80)
    print("📊 最终精准匹配结果汇总（双策略匹配）")
    print("=" * 80)
    print(f"1. 总唯一原始病例数：{len(raw_ids)}")

    # 2. 有对应特征的病例 (全量列出)
    matched_cases = [r for r in raw_ids if case_matches[r]]
    print(f"\n2. 有对应特征的病例（共 {len(matched_cases)} 个）：")
    for rid in matched_cases:
        print(f"   - {rid}：{case_matches[rid]}")

    # 3. 无对应特征的病例
    print(f"\n3. 无对应特征的病例（共 {len(no_feature_cases)} 个）：")
    if no_feature_cases:
        print(f"   全部无匹配病例：{no_feature_cases}")
    else:
        print("   无")

    # 4. 无对应病例的特征文件夹
    print(f"\n4. 无对应病例的特征文件夹（共 {len(no_case_folders)} 个）：")
    if no_case_folders:
        print(f"   全部无匹配文件夹：{no_case_folders}")
    else:
        print("   无")

    print("=" * 80)
    print(f"\n✅ 匹配结果已同步保存至: {output_txt_path}")
    print("========== 匹配执行完成 ==========")


if __name__ == "__main__":
    run_matching()