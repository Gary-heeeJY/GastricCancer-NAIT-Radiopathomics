import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import swanlab

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data_process.dataset_generic import GastricCancerMultiModalDataset, collate_fn_batch_1
from Models.pcr_net import PCRFusionNet


# ===================== 工具类 =====================
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")  # 每次测试覆盖上次的记录

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.log.flush()


def calc_metrics(labels, probs, preds):
    """提取的通用指标计算模块"""
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.5

    return {'acc': acc, 'f1': f1, 'auc': auc, 'sensitivity': sensitivity, 'specificity': specificity}


def evaluate_all_and_ensemble(models, dataloader, device):
    """
    对测试集执行全量评估，保存每个模型与集成的概率和预测值
    返回: 汇总字典(每折和Ensemble的指标) + Pandas DataFrame
    """
    for model in models:
        model.eval()

    all_data_rows = []

    # 指标容器
    labels_list = []
    model_probs = {i: [] for i in range(len(models))}
    model_preds = {i: [] for i in range(len(models))}
    ensemble_probs = []
    ensemble_preds = []

    with torch.no_grad():
        for batch in dataloader:
            patient_id = batch['patient_id']
            img_feat = batch['image_features'].to(device)
            txt_feat = batch['text_feature'].to(device)
            label = batch['label'].item()

            labels_list.append(label)

            # 用于保存该患者在表格中的一行数据
            row_data = {
                'patient_id': patient_id,
                'true_label': label
            }

            batch_probs = []

            # 分别获取每个模型的预测
            for idx, model in enumerate(models):
                logits, _ = model(img_feat, txt_feat)
                prob = torch.sigmoid(logits).item()
                pred = 1 if prob >= 0.5 else 0

                model_probs[idx].append(prob)
                model_preds[idx].append(pred)
                batch_probs.append(prob)

                row_data[f'fold{idx + 1}_prob'] = prob
                row_data[f'fold{idx + 1}_pred'] = pred

            # 集成逻辑 (软投票 Soft Voting)
            ens_prob = np.mean(batch_probs)
            ens_pred = 1 if ens_prob >= 0.5 else 0

            ensemble_probs.append(ens_prob)
            ensemble_preds.append(ens_pred)

            row_data['ensemble_prob'] = ens_prob
            row_data['ensemble_pred'] = ens_pred

            all_data_rows.append(row_data)

    # 汇总各模型和Ensemble的指标
    metrics_summary = {}
    for idx in range(len(models)):
        metrics_summary[f'fold{idx + 1}'] = calc_metrics(labels_list, model_probs[idx], model_preds[idx])

    metrics_summary['ensemble'] = calc_metrics(labels_list, ensemble_probs, ensemble_preds)

    # 转换为 DataFrame
    df_results = pd.DataFrame(all_data_rows)

    return metrics_summary, df_results


@hydra.main(version_base="1.3", config_path="../Configs", config_name="config")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    # 初始化日志记录器
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    log_file = os.path.join(cfg.paths.output_dir, "test_log.txt")
    sys.stdout = Logger(log_file)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cfg.training.gpu_id}")
    else:
        device = torch.device("cpu")
    print(f"\n{'=' * 60}")
    print(f"🔥 使用设备: {device} | 评估日志: {log_file}")
    print(f"{'=' * 60}")

    test_file = os.path.join(cfg.paths.split_dir, "test.xlsx")
    print(f"\n加载独立测试集数据: {test_file}")
    test_dataset = GastricCancerMultiModalDataset(test_file, cfg.paths.feature_dir, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False,
                             collate_fn=collate_fn_batch_1)

    num_folds = 5
    ensemble_models = []

    print("\n📥 正在加载各折最优模型权重...")
    for fold_idx in range(1, num_folds + 1):
        save_dir = os.path.join(cfg.paths.output_dir, f"fold_{fold_idx}")
        model_path = os.path.join(save_dir, "best_model.pth")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到 Fold {fold_idx} 的模型文件: {model_path}")

        model = PCRFusionNet(cfg.model).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        ensemble_models.append(model)
        print(f"   ✅ Fold {fold_idx} 模型加载完毕")

    print("\n🚀 开始在独立测试集上进行全量与集成评估...")
    metrics_summary, df_results = evaluate_all_and_ensemble(ensemble_models, test_loader, device)

    # ================= 打印并记录结果 =================
    for fold_idx in range(1, num_folds + 1):
        m = metrics_summary[f'fold{fold_idx}']
        print(f"\n🔹 Fold {fold_idx} 测试集指标:")
        print(
            f"    AUC: {m['auc']:.4f} | ACC: {m['acc']:.4f} | F1: {m['f1']:.4f} | Sens: {m['sensitivity']:.4f} | Spec: {m['specificity']:.4f}")

    print("\n" + "=" * 50)
    print("🏆 独立测试集 (Test Set) 五折集成最终结果 (Ensemble)")
    print("=" * 50)
    ens_m = metrics_summary['ensemble']
    print(f"✅ Accuracy    : {ens_m['acc']:.4f}  ({ens_m['acc'] * 100:.2f}%)")
    print(f"✅ AUC         : {ens_m['auc']:.4f}")
    print(f"✅ F1-Score    : {ens_m['f1']:.4f}")
    print(f"✅ Sensitivity : {ens_m['sensitivity']:.4f}  (召回率/真阳性率)")
    print(f"✅ Specificity : {ens_m['specificity']:.4f}  (真阴性率)")
    print("=" * 50)

    # ================= 导出预测详细结果表格 =================
    excel_out_path = os.path.join(cfg.paths.output_dir, "test_predictions.xlsx")
    df_results.to_excel(excel_out_path, index=False)
    print(f"\n✅ 测试集所有患者的独立及预测结果大表已保存至: {excel_out_path}")

    # ================= SwanLab 记录 =================
    sw_cfg = cfg.model.swanlab if "swanlab" in cfg.model else cfg.swanlab
    if sw_cfg.enable:
        try:
            swanlab.login(api_key=sw_cfg.api_key)
            swanlab.init(
                project=sw_cfg.project,
                workspace=sw_cfg.workspace,
                name=f"{sw_cfg.name}_Ensemble_Test",
                config=OmegaConf.to_container(cfg, resolve=True)
                # 不设置 save_dir 以避免本地文件产生
            )
            swanlab.log({
                "Test_Ensemble/Accuracy": ens_m['acc'],
                "Test_Ensemble/AUC": ens_m['auc'],
                "Test_Ensemble/F1_Score": ens_m['f1'],
                "Test_Ensemble/Sensitivity": ens_m['sensitivity'],
                "Test_Ensemble/Specificity": ens_m['specificity']
            })
            swanlab.finish()
            print("✅ 集成测试结果已同步至 SwanLab。")
        except Exception as e:
            print(f"⚠️ SwanLab 记录测试结果失败: {e}")


if __name__ == "__main__":
    main()