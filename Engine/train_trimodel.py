import os
import sys
import tempfile
import random

import hydra
import numpy as np
import pandas as pd
import swanlab
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data_process.dataset_generic import GastricCancerMultiModalDataset, clean_id, collate_fn_batch_1
from Engine.val import evaluate_model
from Models.pcr_net import PCRFusionNet

# ===================== 核心策略开关 =====================
EVALUATION_STRATEGY = 1  # 1: Val AUC 选优; 2: Train Loss 选优


def set_seed(seed=42):
    """固定所有可能产生随机性的种子，确保实验可复现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

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
    log_file = os.path.join(cfg.paths.output_dir, "train_log.txt")
    sys.stdout = Logger(log_file)

    seed = cfg.training.get("seed", 42)
    set_seed(seed)

    device = torch.device(f"cuda:{cfg.training.gpu_id}" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 60}")
    print(f"🔥 使用设备: {device} | 日志: {log_file}")
    print(f"🎯 策略: 方案 {EVALUATION_STRATEGY}")
    print(f"{'=' * 60}")

    sw_cfg = cfg.model.get('swanlab', cfg.swanlab)
    swan_cache_dir = os.path.join(tempfile.gettempdir(), "swanlab_cache")

    fold_results = {'auc': [], 'acc': [], 'f1': [], 'sensitivity': [], 'specificity': []}
    patient_to_val_fold = {}

    for fold_idx in range(1, 6):
        print("\n" + "=" * 50)
        print(f"🚀 正在启动 Fold {fold_idx}/5 训练流程 (纯血三模态: 病理+文本+CT)")
        print("=" * 50)

        if sw_cfg.enable:
            try:
                swanlab.login(api_key=sw_cfg.api_key)
                swanlab.init(
                    project=cfg.swanlab.project,
                    workspace=cfg.swanlab.workspace,
                    name=f"{cfg.swanlab.name}_Fold{fold_idx}_S{EVALUATION_STRATEGY}",
                    config=OmegaConf.to_container(cfg, resolve=True),
                    save_dir=swan_cache_dir
                )
            except Exception:
                pass

        fold_file = os.path.join(cfg.paths.split_dir, f"fold{fold_idx}.xlsx")
        df_fold = pd.read_excel(fold_file)
        for pid in df_fold['val_patho_id'].dropna():
            patient_to_val_fold[clean_id(pid)] = fold_idx

        g = torch.Generator()
        g.manual_seed(seed)

        # 👉 移除了 use_ct 判断，直接硬编码指定使用 CT，代码更直白
        train_dataset = GastricCancerMultiModalDataset(fold_file, cfg.paths.feature_dir, mode='train', use_ct=True,
                                                       ct_csv_path=cfg.paths.ct_features_dir)
        val_dataset = GastricCancerMultiModalDataset(fold_file, cfg.paths.feature_dir, mode='val', use_ct=True,
                                                     ct_csv_path=cfg.paths.ct_features_dir)

        train_loader = DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True,
                                  collate_fn=collate_fn_batch_1, worker_init_fn=np.random.seed(seed), generator=g)
        val_loader = DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False,
                                collate_fn=collate_fn_batch_1)
        train_eval_loader = DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, shuffle=False,
                                       collate_fn=collate_fn_batch_1)

        model = PCRFusionNet(cfg.model).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg.training.pos_weight]).to(device))
        optimizer = optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-6)

        epochs = cfg.training.epochs
        accumulation_steps = cfg.training.gradient_accumulation_steps
        best_val_auc = 0.0
        best_train_loss = float('inf')

        # 👉 修复早停变量未定义的 Bug
        use_early_stopping = cfg.training.early_stopping.enable
        patience = cfg.training.early_stopping.patience
        trigger_times = 0

        save_dir = os.path.join(cfg.paths.output_dir, f"fold_{fold_idx}")
        os.makedirs(save_dir, exist_ok=True)
        best_model_path = os.path.join(save_dir, "best_model.pth")

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(train_loader):
                # 👉 移除了判断，硬核直接喂入三个模态
                img_feat = batch['image_features'].to(device)
                txt_feat = batch['text_feature'].to(device)
                ct_feat = batch['ct_features'].to(device)

                logits, _ = model(img_feat, txt_feat, ct_feat)

                label = batch['label'].to(device)
                loss = criterion(logits.view(-1), label.view(-1))
                loss = loss / accumulation_steps
                loss.backward()

                if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                    optimizer.step()
                    optimizer.zero_grad()

            train_metrics = evaluate_model(model, train_eval_loader, criterion, device)
            val_metrics = evaluate_model(model, val_loader, criterion, device)
            scheduler.step()

            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"Fold {fold_idx} | Epoch [{epoch}/{epochs}] Train Loss: {train_metrics['loss']:.4f} | Val AUC: {val_metrics['auc']:.4f}")

            is_best = False
            if EVALUATION_STRATEGY == 1 and val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                is_best = True
            elif EVALUATION_STRATEGY == 2 and train_metrics['loss'] < best_train_loss:
                best_train_loss = train_metrics['loss']
                is_best = True

            if is_best:
                torch.save(model.state_dict(), best_model_path)
                trigger_times = 0
            else:
                trigger_times += 1
                if use_early_stopping and trigger_times >= patience:
                    print(f"⏹ 触发早停机制，连续 {patience} 轮未提升。")
                    break

        print(f"📥 提取 Fold {fold_idx} 的最佳权重验证...")
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        final_fold_metrics = evaluate_model(model, val_loader, criterion, device)

        print(
            f"🎉 Fold {fold_idx} 验证集表现: AUC: {final_fold_metrics['auc']:.4f} | ACC: {final_fold_metrics['acc']:.4f} | F1: {final_fold_metrics['f1']:.4f}")

        for k in fold_results.keys():
            fold_results[k].append(final_fold_metrics[k])

        if sw_cfg.enable: swanlab.finish()

    # ================= 3. OOF 盲测汇总 (专注纯盲测) =================
    print("\n" + "=" * 60)
    print("🏆 启动全量数据 OOF (盲测) 后处理分析")
    print("=" * 60)

    ensemble_models = []
    for i in range(1, 6):
        model = PCRFusionNet(cfg.model).to(device)
        model.load_state_dict(
            torch.load(os.path.join(cfg.paths.output_dir, f"fold_{i}", "best_model.pth"), map_location=device,
                       weights_only=True))
        model.eval()
        ensemble_models.append(model)

    dataset_all = GastricCancerMultiModalDataset(cfg.paths.original_excel_dir, cfg.paths.feature_dir, mode='all',
                                                 use_ct=True, ct_csv_path=cfg.paths.ct_features_dir)
    loader_all = DataLoader(dataset_all, batch_size=1, shuffle=False, collate_fn=collate_fn_batch_1)

    all_predictions = []
    with torch.no_grad():
        for batch in loader_all:
            patient_id, true_label = batch['patient_id'], int(batch['label'].item())
            val_fold_idx = patient_to_val_fold.get(clean_id(patient_id))
            if val_fold_idx is None: continue

            # 只取对应折数的模型预测
            model = ensemble_models[val_fold_idx - 1]
            img_feat, txt_feat, ct_feat = batch['image_features'].to(device), batch['text_feature'].to(device), batch[
                'ct_features'].to(device)

            # 👉 彻底去掉了 if use_ct_flag，强制三模态预测
            logits, _ = model(img_feat, txt_feat, ct_feat)
            prob = torch.sigmoid(logits).item()

            all_predictions.append({
                'patient_id': patient_id,
                'true_label': true_label,
                'val_fold_idx': val_fold_idx,
                'oof_val_prob': prob,
                'oof_val_pred': 1 if prob >= 0.5 else 0
            })

    df_results = pd.DataFrame(all_predictions)
    excel_out_path = os.path.join(cfg.paths.output_dir, "OOF_Blind_Test_Predictions.xlsx")
    df_results.to_excel(excel_out_path, index=False)

    print("\n📊 五折盲测集 (OOF) 平均指标:")
    for k, v in fold_results.items():
        print(f"✅ {k.upper():<12}: {np.mean(v):.4f} ± {np.std(v):.4f}")

    # 计算整体 OOF 总指标
    y_true_all, y_prob_all, y_pred_all = df_results['true_label'].values, df_results['oof_val_prob'].values, df_results[
        'oof_val_pred'].values
    try:
        auc_all = roc_auc_score(y_true_all, y_prob_all)
    except ValueError:
        auc_all = 0.5
    acc_all, f1_all = accuracy_score(y_true_all, y_pred_all), f1_score(y_true_all, y_pred_all, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1]).ravel()

    print("\n" + "=" * 50)
    print("🌟 总体盲测拼接总指标 (Overall OOF Pool):")
    print("=" * 50)
    print(f"✅ AUC         : {auc_all:.4f}")
    print(f"✅ ACC         : {acc_all:.4f}")
    print(f"✅ F1          : {f1_all:.4f}")
    print(f"✅ SENSITIVITY : {tp / (tp + fn) if (tp + fn) > 0 else 0.0:.4f}")
    print(f"✅ SPECIFICITY : {tn / (tn + fp) if (tn + fp) > 0 else 0.0:.4f}")
    print("=" * 50)
    print(f"\n✅ 表格已保存至: {excel_out_path}")


if __name__ == "__main__":
    main()