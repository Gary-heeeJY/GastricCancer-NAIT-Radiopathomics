import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data_process.dataset_generic import GastricCancerMultiModalDataset, collate_fn_batch_1, clean_id


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.log.flush()


@hydra.main(version_base="1.3", config_path="../Configs", config_name="config")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    log_file = os.path.join(cfg.paths.output_dir, "evaluation_log.txt")
    sys.stdout = Logger(log_file)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cfg.training.gpu_id}")
    else:
        device = torch.device("cpu")

    print(f"\n{'=' * 60}")
    print("🏆 启动严谨 OOF (盲测) 数据后处理与指标分析")
    print(f"{'=' * 60}")

    use_ct_flag = cfg.model.get('use_ct', False)
    is_ct_only = (cfg.model.name == "mlp_ct" or cfg.model.name == "ct_only")

    # 1. 重建 patient_to_val_fold 映射
    patient_to_val_fold = {}
    for fold_idx in range(1, 6):
        fold_file = os.path.join(cfg.paths.split_dir, f"fold{fold_idx}.xlsx")
        df_fold = pd.read_excel(fold_file)
        for pid in df_fold['val_patho_id'].dropna():
            patient_to_val_fold[clean_id(pid)] = fold_idx

    # 2. 动态加载 5 个保存好的模型
    ensemble_models = []
    print("📥 正在加载已保存的 5 折最佳盲测模型...")
    for i in range(1, 6):
        model_path = os.path.join(cfg.paths.output_dir, f"fold_{i}", "best_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型 {model_path}，请确认路径参数是否正确！")

        if is_ct_only:
            from Models.ct_net import CTOnlyNet
            model = CTOnlyNet(cfg.model).to(device)
        else:
            from Models.pcr_net import PCRFusionNet
            model = PCRFusionNet(cfg.model).to(device)

        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        ensemble_models.append(model)
        print(f"  ✅ Fold {i} 权重加载成功")

    # 3. 读取全部样本
    original_excel = cfg.paths.original_excel_dir
    dataset_all = GastricCancerMultiModalDataset(
        excel_path=original_excel,
        feature_root_dir=cfg.paths.feature_dir,
        mode='all',
        use_ct=(use_ct_flag or is_ct_only),
        ct_csv_path=cfg.paths.ct_features_dir if (use_ct_flag or is_ct_only) else None
    )
    loader_all = DataLoader(dataset_all, batch_size=1, shuffle=False, collate_fn=collate_fn_batch_1)

    all_predictions = []

    print("\n🚀 开始执行严谨盲测 (OOF) 前向预测...")
    with torch.no_grad():
        for batch in loader_all:
            patient_id = batch['patient_id']
            true_label = int(batch['label'].item())
            val_fold_idx = patient_to_val_fold.get(clean_id(patient_id))

            if val_fold_idx is None:
                continue

            # 🎯 核心提速与净化逻辑：只用该病人作为盲测集的那一折模型来预测，不再算均值
            model = ensemble_models[val_fold_idx - 1]

            if is_ct_only:
                ct_feat = batch['ct_features'].to(device)
                logits, _ = model(ct_feat)
            else:
                img_feat = batch['image_features'].to(device)
                txt_feat = batch['text_feature'].to(device)
                if use_ct_flag:
                    ct_feat = batch['ct_features'].to(device)
                    logits, _ = model(img_feat, txt_feat, ct_feat)
                else:
                    logits, _ = model(img_feat, txt_feat)

            prob = torch.sigmoid(logits).item()
            pred = 1 if prob >= 0.5 else 0

            row_data = {
                'patient_id': patient_id,
                'true_label': true_label,
                'val_fold_idx': val_fold_idx,
                'oof_prob': prob,
                'oof_pred': pred
            }
            all_predictions.append(row_data)

    # 4. 保存纯净盲测结果大表
    df_results = pd.DataFrame(all_predictions)
    excel_out_path = os.path.join(cfg.paths.output_dir, "OOF_Blind_Test_Predictions.xlsx")
    df_results.to_excel(excel_out_path, index=False)

    # ================= 5. 计算并打印评估指标 =================
    fold_metrics = {'auc': [], 'acc': [], 'f1': [], 'sensitivity': [], 'specificity': []}

    print("\n" + "=" * 50)
    print("📊 每一折独立盲测 (OOF) 结果明细")
    print("=" * 50)

    for i in range(1, 6):
        df_fold = df_results[df_results['val_fold_idx'] == i]
        if len(df_fold) == 0: continue

        y_true = df_fold['true_label'].values
        y_prob = df_fold['oof_prob'].values
        y_pred = df_fold['oof_pred'].values

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.5

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        fold_metrics['acc'].append(acc)
        fold_metrics['f1'].append(f1)
        fold_metrics['auc'].append(auc)
        fold_metrics['sensitivity'].append(sens)
        fold_metrics['specificity'].append(spec)

        print(f"🔹 Fold {i} | AUC: {auc:.4f} | ACC: {acc:.4f} | F1: {f1:.4f} | SENS: {sens:.4f} | SPEC: {spec:.4f}")

    print("\n" + "=" * 50)
    print("📈 五折独立盲测 (OOF) 平均指标 (Mean ± SD)")
    print("=" * 50)
    print(f"✅ AUC         : {np.mean(fold_metrics['auc']):.4f} ± {np.std(fold_metrics['auc']):.4f}")
    print(f"✅ ACC         : {np.mean(fold_metrics['acc']):.4f} ± {np.std(fold_metrics['acc']):.4f}")
    print(f"✅ F1          : {np.mean(fold_metrics['f1']):.4f} ± {np.std(fold_metrics['f1']):.4f}")
    print(f"✅ SENSITIVITY : {np.mean(fold_metrics['sensitivity']):.4f} ± {np.std(fold_metrics['sensitivity']):.4f}")
    print(f"✅ SPECIFICITY : {np.mean(fold_metrics['specificity']):.4f} ± {np.std(fold_metrics['specificity']):.4f}")

    # 计算 111 个样本拼在一起的总盲测指标
    y_true_all = df_results['true_label'].values
    y_prob_all = df_results['oof_prob'].values
    y_pred_all = df_results['oof_pred'].values

    acc_all = accuracy_score(y_true_all, y_pred_all)
    f1_all = f1_score(y_true_all, y_pred_all, zero_division=0)
    try:
        auc_all = roc_auc_score(y_true_all, y_prob_all)
    except ValueError:
        auc_all = 0.5

    tn_a, fp_a, fn_a, tp_a = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1]).ravel()
    sens_all = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0.0
    spec_all = tn_a / (tn_a + fp_a) if (tn_a + fp_a) > 0 else 0.0

    print("\n" + "=" * 50)
    print("🌟 总体盲测拼接总指标 (Overall OOF Pool)")
    print("=" * 50)
    print(f"✅ AUC         : {auc_all:.4f}")
    print(f"✅ ACC         : {acc_all:.4f}")
    print(f"✅ F1          : {f1_all:.4f}")
    print(f"✅ SENSITIVITY : {sens_all:.4f}")
    print(f"✅ SPECIFICITY : {spec_all:.4f}")
    print("=" * 50)

    print(f"\n📁 评估完成！纯盲测概率大表已保存至: {excel_out_path}")
    print(f"📝 完整评估报告已记录至: {log_file}")


if __name__ == "__main__":
    main()