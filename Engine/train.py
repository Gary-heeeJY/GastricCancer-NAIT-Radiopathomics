import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import swanlab
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data_process.dataset_generic import GastricCancerMultiModalDataset, collate_fn_batch_1
from Models.pcr_net import PCRFusionNet
from Engine.val import evaluate_model


# ===================== 工具类：同步输出到终端和 TXT 文件 =====================
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        # 使用追加模式 'a'，五折的内容会顺次记录在同一个文件中
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

    # 初始化日志记录器，保存到 Outputs/train_log.txt
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    log_file = os.path.join(cfg.paths.output_dir, "train_log.txt")
    sys.stdout = Logger(log_file)

    # ================= 1. GPU / Device 设置 =================
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cfg.training.gpu_id}")
    else:
        device = torch.device("cpu")
    print(f"\n{'=' * 50}")
    print(f"🔥 使用设备: {device} | 日志文件: {log_file}")
    print(f"{'=' * 50}")

    sw_cfg = cfg.model.swanlab if "swanlab" in cfg.model else cfg.swanlab
    sys_temp_dir = tempfile.gettempdir()
    swan_cache_dir = os.path.join(sys_temp_dir, "swanlab_cache")

    # ================= 2. 五折交叉验证外循环 =================
    for fold_idx in range(1, 6):
        print("\n" + "=" * 50)
        print(f"🚀 正在启动 Fold {fold_idx}/5 训练流程")
        print("=" * 50)

        if sw_cfg.enable:
            try:
                swanlab.login(api_key=sw_cfg.api_key)
                swanlab.init(
                    project=sw_cfg.project,
                    workspace=sw_cfg.workspace,
                    name=f"{sw_cfg.name}_Fold{fold_idx}",
                    config=OmegaConf.to_container(cfg, resolve=True),
                    save_dir=swan_cache_dir  # 👈 强行定向到系统垃圾桶
                    # 去掉 save_dir，让 swanlab 使用系统默认缓存路径，不污染 Outputs
                )
            except Exception as e:
                print(f"⚠️ SwanLab 初始化失败: {e}")

        fold_file = os.path.join(cfg.paths.split_dir, f"fold{fold_idx}.xlsx")

        train_dataset = GastricCancerMultiModalDataset(fold_file, cfg.paths.feature_dir, mode='train')
        val_dataset = GastricCancerMultiModalDataset(fold_file, cfg.paths.feature_dir, mode='val')

        train_loader = DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True,
                                  collate_fn=collate_fn_batch_1)
        val_loader = DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False,
                                collate_fn=collate_fn_batch_1)
        train_eval_loader = DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, shuffle=False,
                                       collate_fn=collate_fn_batch_1)

        model = PCRFusionNet(cfg.model).to(device)
        pos_weight = torch.tensor([cfg.training.pos_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-6)

        epochs = cfg.training.epochs
        accumulation_steps = cfg.training.gradient_accumulation_steps
        best_auc = 0.0

        use_early_stopping = cfg.training.early_stopping.enable
        patience = cfg.training.early_stopping.patience
        trigger_times = 0

        save_dir = os.path.join(cfg.paths.output_dir, f"fold_{fold_idx}")
        os.makedirs(save_dir, exist_ok=True)
        best_model_path = os.path.join(save_dir, "best_model.pth")

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss_accumulated = 0.0
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(train_loader):
                img_feat = batch['image_features'].to(device)
                txt_feat = batch['text_feature'].to(device)
                label = batch['label'].to(device)

                logits, _ = model(img_feat, txt_feat)
                loss = criterion(logits.view(-1), label.view(-1))
                loss = loss / accumulation_steps
                loss.backward()

                actual_loss = loss.item() * accumulation_steps
                epoch_loss_accumulated += actual_loss

                if sw_cfg.enable:
                    swanlab.log({"Train/Batch_Loss": actual_loss})

                if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                    optimizer.step()
                    optimizer.zero_grad()

            train_metrics = evaluate_model(model, train_eval_loader, criterion, device)
            val_metrics = evaluate_model(model, val_loader, criterion, device)

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            print(f"Fold {fold_idx} | Epoch [{epoch}/{epochs}] LR: {current_lr:.6f}\n"
                  f"  Train -> Loss: {train_metrics['loss']:.4f} | ACC: {train_metrics['acc']:.4f} | AUC: {train_metrics['auc']:.4f}\n"
                  f"  Val   -> Loss: {val_metrics['loss']:.4f} | ACC: {val_metrics['acc']:.4f} | AUC: {val_metrics['auc']:.4f} | F1: {val_metrics['f1']:.4f}")

            if sw_cfg.enable:
                swanlab.log({
                    "Train/LR": current_lr,
                    "Train/Epoch_Loss": train_metrics['loss'],
                    "Train/Accuracy": train_metrics['acc'],
                    "Train/AUC": train_metrics['auc'],
                    "Val/Loss": val_metrics['loss'],
                    "Val/Accuracy": val_metrics['acc'],
                    "Val/AUC": val_metrics['auc'],
                    "Val/F1_Score": val_metrics['f1'],
                    "Val/Sensitivity": val_metrics['sensitivity'],
                    "Val/Specificity": val_metrics['specificity']
                }, step=epoch)

            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                torch.save(model.state_dict(), best_model_path)
                if use_early_stopping:
                    trigger_times = 0
                print(f"   🌟 [Fold {fold_idx}] 发现最佳模型! 保存至: {best_model_path}")
            else:
                if use_early_stopping:
                    trigger_times += 1
                    if trigger_times >= patience:
                        print(f"\n⏹ 触发早停机制。")
                        break

        print(f"\n🎉 Fold {fold_idx} 训练结束! 最佳验证集 AUC: {best_auc:.4f}")

        if sw_cfg.enable:
            swanlab.finish()

    print("\n" + "=" * 50)
    print("✅ 五折交叉验证全部完成！所有日志已写入 train_log.txt")
    print("=" * 50)


if __name__ == "__main__":
    main()