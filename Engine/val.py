import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_model(model, dataloader, criterion, device):
    """
    运行验证集/测试集，并计算综合指标
    """
    model.eval()
    total_loss = 0.0

    all_labels = []
    all_preds = []  # 0或1的硬标签
    all_probs = []  # 概率值，用于算 AUC

    with torch.no_grad():
        for batch in dataloader:
            # 此时 batch_size=1, batch 剥去了最外层
            img_feat = batch['image_features'].to(device)
            txt_feat = batch['text_feature'].to(device)
            label = batch['label'].to(device)

            # 👉 修复点：通过字典判断智能传参
            if 'ct_features' in batch:
                ct_feat = batch['ct_features'].to(device)
                logits, _ = model(img_feat, txt_feat, ct_feat)
            else:
                logits, _ = model(img_feat, txt_feat)

            loss = criterion(logits.view(-1), label.view(-1))

            total_loss += loss.item()

            # 计算概率与预测
            prob = torch.sigmoid(logits).item()
            pred = 1 if prob >= 0.5 else 0

            all_labels.append(label.item())
            all_probs.append(prob)
            all_preds.append(pred)

    # 计算指标
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # 算敏感性(Sensitivity/Recall)和特异性(Specificity)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # 计算 AUC (如果全部是同一类，AUC会报错，做个保护)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5

    metrics = {
        'loss': avg_loss,
        'acc': acc,
        'f1': f1,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }

    return metrics