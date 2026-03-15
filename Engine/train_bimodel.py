import os
import sys
import tempfile
import random
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import swanlab

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data_process.dataset_generic import GastricCancerMultiModalDataset, collate_fn_batch_1, clean_id
from Models.pcr_net import PCRFusionNet
from Engine.val import evaluate_model

EVALUATION_STRATEGY = 2

def set_seed(seed=42):
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
    print(f"🎯 策略: 方案 {EVALUATION_STRATEGY} (Train Loss 选优, 纯盲测)")
    print(f"{'=' * 60}")

    sw_cfg = cfg.model.get('swanlab', cfg.swanlab)
    swan_cache_dir = os.path.join(tempfile.gettempdir(), "swanlab_cache")

    fold_results = {'auc': [], 'acc': [], 'f1': [], 'sensitivity': [], 'specificity': []}
    patient_to_val_fold = {}

    for fold_idx in range(1, 6):
        print("\n" + "=" * 50)
        print(f"🚀 正在启动 Fold {fold_idx}/5 训练流程 (双模态: 病理+文本)")
        print("=" * 50)

        if sw_cfg.enable:
            try:
                swanlab.login(api_key=sw_cfg.api_key)
                swanlab.init(
                    project=cfg.swanlab.project, workspace=cfg.swanlab.workspace,
                    name=f"{cfg.swanlab.name}_Fold{fold_idx}",
                    config=OmegaConf.to_container(cfg, resolve=True), save_dir=swan_cache_dir
                )
            except Exception: pass

        fold_file = os.path.join(cfg.paths.split_dir, f"fold{fold_idx}.xlsx")
        df_fold = pd.read_excel(fold_file)
        for pid in df_fold['val_patho_id'].dropna():
            patient_to_val_fold[clean_id(pid)] = fold_idx

        g = torch.Generator()
        g.manual_seed(seed)

        # 👉 双模态不传入 ct_csv_path
        train_dataset = GastricCancerMultiModalDataset(fold_file, cfg.paths.feature_dir, mode='train')
        val_dataset = GastricCancerMultiModalDataset(fold_file, cfg.paths.feature_dir, mode='val')

        train_loader = DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, collate_fn=collate_fn_batch_1, worker_init_fn=np.random.seed(seed), generator=g)
        val_loader = DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, collate_fn=collate_fn_batch_1)
        train_eval_loader = DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, collate_fn=collate_fn_batch_1)

        model = PCRFusionNet(cfg.model).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg.training.pos_weight]).to(device))
        optimizer = optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-6)

        epochs, accumulation_steps = cfg.training.epochs, cfg.training.gradient_accumulation_steps
        best_val_auc, best_train_loss = 0.0, float('inf')
        trigger_times, patience = 0, cfg.training.early_stopping.patience

        save_dir = os.path.join(cfg.paths.output_dir, f"fold_{fold_idx}")
        os.makedirs(save_dir, exist_ok=True)
        best_model_path = os.path.join(save_dir, "best_model.pth")

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(train_loader):
                img_feat, txt_feat = batch['image_features'].to(device), batch['text_feature'].to(device)
                logits, _ = model(img_feat, txt_feat)
                loss = criterion(logits.view(-1), batch['label'].to(device).view(-1))
                loss = loss / accumulation_steps
                loss.backward()

                if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                    optimizer.step()
                    optimizer.zero_grad()

            train_metrics = evaluate_model(model, train_eval_loader, criterion, device)
            val_metrics = evaluate_model(model, val_loader, criterion, device)
            scheduler.step()

            if epoch % 5 == 0 or epoch == 1:
                print(f"Fold {fold_idx} | Epoch [{epoch}/{epochs}] Train Loss: {train_metrics['loss']:.4f} | Val AUC: {val_metrics['auc']:.4f}")

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
                if cfg.training.early_stopping.enable and trigger_times >= patience: break

        print(f"📥 提取 Fold {fold_idx} 最终盲测验证...")
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        final_fold_metrics = evaluate_model(model, val_loader, criterion, device)
        print(f"🎉 Fold {fold_idx} 表现: AUC: {final_fold_metrics['auc']:.4f}")

        for k in fold_results.keys(): fold_results[k].append(final_fold_metrics[k])
        if sw_cfg.enable: swanlab.finish()

    # ================= 3. OOF 盲测汇总 =================
    print("\n" + "=" * 60 + "\n🏆 启动 OOF 盲测汇总 (双模态)\n" + "=" * 60)

    ensemble_models = []
    for i in range(1, 6):
        model = PCRFusionNet(cfg.model).to(device)
        model.load_state_dict(torch.load(os.path.join(cfg.paths.output_dir, f"fold_{i}", "best_model.pth"), map_location=device, weights_only=True))
        model.eval()
        ensemble_models.append(model)

    dataset_all = GastricCancerMultiModalDataset(cfg.paths.original_excel_dir, cfg.paths.feature_dir, mode='all')
    loader_all = DataLoader(dataset_all, batch_size=1, shuffle=False, collate_fn=collate_fn_batch_1)

    all_predictions = []
    with torch.no_grad():
        for batch in loader_all:
            patient_id, true_label = batch['patient_id'], int(batch['label'].item())
            val_fold_idx = patient_to_val_fold.get(clean_id(patient_id))
            if val_fold_idx is None: continue

            model = ensemble_models[val_fold_idx - 1]
            img_feat, txt_feat = batch['image_features'].to(device), batch['text_feature'].to(device)
            logits, _ = model(img_feat, txt_feat)
            prob = torch.sigmoid(logits).item()

            all_predictions.append({
                'patient_id': patient_id, 'true_label': true_label, 'val_fold_idx': val_fold_idx,
                'oof_val_prob': prob, 'oof_val_pred': 1 if prob >= 0.5 else 0
            })

    df_results = pd.DataFrame(all_predictions)
    excel_out_path = os.path.join(cfg.paths.output_dir, "OOF_Blind_Test_Predictions_Bimodal.xlsx")
    df_results.to_excel(excel_out_path, index=False)

    print("\n📊 五折盲测集 (OOF) 独立指标汇总:")
    for k, v in fold_results.items(): print(f"✅ {k.upper():<12}: {np.mean(v):.4f} ± {np.std(v):.4f}")

    y_true_all, y_prob_all, y_pred_all = df_results['true_label'].values, df_results['oof_val_prob'].values, df_results['oof_val_pred'].values
    try: auc_all = roc_auc_score(y_true_all, y_prob_all)
    except ValueError: auc_all = 0.5
    acc_all, f1_all = accuracy_score(y_true_all, y_pred_all), f1_score(y_true_all, y_pred_all, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1]).ravel()

    print("\n" + "=" * 50 + "\n🌟 总体盲测拼接总指标 (Overall OOF Pool):\n" + "=" * 50)
    print(f"✅ AUC         : {auc_all:.4f}\n✅ ACC         : {acc_all:.4f}\n✅ F1          : {f1_all:.4f}")
    print(f"✅ SENSITIVITY : {tp / (tp + fn) if (tp + fn) > 0 else 0.0:.4f}")
    print(f"✅ SPECIFICITY : {tn / (tn + fp) if (tn + fp) > 0 else 0.0:.4f}\n" + "=" * 50)
    print(f"\n✅ OOF 表格已保存至: {excel_out_path}")

if __name__ == "__main__":
    main()