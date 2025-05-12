import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from models import deeplabv3plus
from utils import data_loader
from utils.pseudo_labeling import generate_pseudo_labels
from utils import losses
from sklearn.model_selection import KFold
from sklearn.metrics import jaccard_score
import csv
from copy import deepcopy
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchvision.transforms import RandomCrop
import torchvision.transforms as T
import torchvision.transforms.functional as TF
# from utils.lr_scheduler import PolyLR


@torch.no_grad()
def update_teacher_model(student_model, teacher_model, ema_decay=0.99):
    """
    Update the teacher model using exponential moving average (EMA) of the student model's weights.
    """
    for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
        teacher_param.data = ema_decay * teacher_param.data + (1 - ema_decay) * student_param.data

def get_random_crop_coords(img_h, img_w, crop_h, crop_w):
    top = torch.randint(0, img_h - crop_h + 1, (1,)).item()
    left = torch.randint(0, img_w - crop_w + 1, (1,)).item()
    return top, left

def train(
        num_folds=6, num_epochs=50, batch_size=8, threshold=0.5,
        learning_rate=0.001, temperature=0.1, neighborhood_size=5,
        weight_pixel=1.0, weight_local=1.0, weight_directional=1.0,
        supervised_loss='cross_entropy', contrastive_loss='pixel',
        dynamic_threshold=True, class_balanced=True, ema_decay=0.99
):
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    student_model = deeplabv3plus.resnet101_deeplabv3plus_imagenet(num_classes=2, pretrained=True).to(device)
    teacher_model = deepcopy(student_model).to(device)
    teacher_model.eval()
    student_model = DDP(student_model, device_ids=[local_rank])

    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate, weight_decay=5e-4)

    if supervised_loss == 'cross_entropy':
        criterion = losses.MaskCrossEntropyLoss2d()
    elif supervised_loss == 'dice':
        criterion = losses.DiceLoss()
    elif supervised_loss == 'ce_dice':
        criterion = losses.CE_DiceLoss()
    else:
        raise ValueError(f"Unsupported supervised_loss: {supervised_loss}")

    if contrastive_loss == 'pixel':
        contrastive_loss_fn = losses.PixelContrastiveLoss(temperature=temperature)
    elif contrastive_loss == 'local':
        contrastive_loss_fn = losses.LocalContrastiveLoss(
            temperature=temperature, no_of_pos_eles=neighborhood_size, no_of_neg_eles=neighborhood_size
        )
    elif contrastive_loss == 'directional':
        contrastive_loss_fn = losses.DirectionalContrastiveLoss(temperature=temperature)
    elif contrastive_loss == 'hybrid':
        contrastive_loss_fn = losses.HybridContrastiveLoss(
            temperature=temperature, neighborhood_size=neighborhood_size,
            weight_pixel=weight_pixel, weight_local=weight_local, weight_directional=weight_directional
        )
    else:
        raise ValueError(f"Unsupported contrastive_loss: {contrastive_loss}")

    train_dataset = data_loader.get_ijmond_seg_dataset('data/IJMOND_SEG', split='train')
    train_dataset_size = len(train_dataset)

    # Unlabeled data loader with strong augmentations
    unlabeled_dataloader = data_loader.get_unlabelled_dataloader(
        ['data/IJMOND_VID/frames', 'data/RISE/frames'],
        batch_size=batch_size,
        shuffle=True,
        rank=local_rank,
        world_size=dist.get_world_size()
    )

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    best_model_path = f'models/best_model_{supervised_loss}_{contrastive_loss}.pth'
    best_miou = 0.0

    if local_rank == 0:
        # Initialize CSV file for logging training losses
        train_loss_csv = f'trainlosses_{supervised_loss}_{contrastive_loss}.csv'
        with open(train_loss_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["fold", "epoch", "supervised_loss", "semi_supervised_loss"])

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(train_dataset_size))):
        train_dataloader = data_loader.get_ijmond_seg_dataloader_train(
            train_idx, split='train', batch_size=batch_size, shuffle=True, rank=local_rank, world_size=dist.get_world_size()
        )
        val_dataloader = data_loader.get_ijmond_seg_dataloader_validation(
            val_idx, split='train', batch_size=batch_size, shuffle=True, rank=local_rank, world_size=dist.get_world_size()
        )

        # iters_per_epoch = len(train_dataloader) + len(unlabeled_dataloader)
        # scheduler = PolyLR(optimizer, num_epochs=num_epochs, iters_per_epoch=iters_per_epoch)

        for epoch in range(num_epochs):
            train_dataloader.sampler.set_epoch(epoch)
            student_model.train()
            epoch_loss = 0

            # Supervised training on labeled data (weak augmentations)
            for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} - Supervised"):
                images, labels = images.to(device), labels.to(device)
                outputs = student_model(images)
                sup_loss = criterion(outputs, labels)
                optimizer.zero_grad()
                sup_loss.backward()
                optimizer.step()
                # scheduler.step()
                epoch_loss += sup_loss.item()

            print(f"Epoch {epoch + 1} Supervised Loss: {epoch_loss / len(train_dataloader):.4f}")

            semi_supervised_loss = 0
            pseudo_batches_count = 0
            if epoch >= 0:  # if epoch>10
                try:
                    pseudo_batches = generate_pseudo_labels(
                        teacher_model, unlabeled_dataloader, threshold, device,
                        dynamic_threshold=dynamic_threshold, class_balanced=class_balanced
                    )

                    for images, pseudo_labels in tqdm(
                            pseudo_batches,
                            desc=f"Epoch {epoch + 1}/{num_epochs} - Semi-supervised"
                    ):
                        try:
                            images, pseudo_labels = images.to(device), pseudo_labels.to(device)
                            student_model.train()
                            if contrastive_loss in ['directional', 'hybrid']:
                                # Generate two crops with overlap
                                b, c, H, W = images.shape
                                crop_h, crop_w = 256, 256

                                top1, left1 = get_random_crop_coords(H, W, crop_h, crop_w)
                                top2, left2 = get_random_crop_coords(H, W, crop_h, crop_w)

                                overlap_top = max(top1, top2)
                                overlap_left = max(left1, left2)
                                overlap_bottom = min(top1 + crop_h, top2 + crop_h)
                                overlap_right = min(left1 + crop_w, left2 + crop_w)

                                if overlap_bottom - overlap_top <= 0 or overlap_right - overlap_left <= 0:
                                    continue  # No valid overlap

                                xu1 = images[:, :, top1:top1 + crop_h, left1:left1 + crop_w]
                                xu2 = images[:, :, top2:top2 + crop_h, left2:left2 + crop_w]

                                low_level_augment = T.Compose([
                                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                    T.RandomGrayscale(p=0.2),
                                    T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                                ])
                                xu1_aug = torch.stack(
                                    [low_level_augment(TF.to_pil_image(img.cpu())).convert("RGB") for img in xu1.cpu()])
                                xu2_aug = torch.stack(
                                    [low_level_augment(TF.to_pil_image(img.cpu())).convert("RGB") for img in xu2.cpu()])
                                xu1 = xu1_aug.permute(0, 3, 1, 2).to(device) / 255.0
                                xu2 = xu2_aug.permute(0, 3, 1, 2).to(device) / 255.0

                                # Pass through model
                                logits1, feat1 = student_model(xu1, feature_maps=True)
                                logits2, feat2 = student_model(xu2, feature_maps=True)

                                pseudo_label1 = torch.argmax(logits1, dim=1)
                                pseudo_label2 = torch.argmax(logits2, dim=1)

                                pseudo_logits1 = F.softmax(logits1, dim=1).max(1)[0]
                                pseudo_logits2 = F.softmax(logits2, dim=1).max(1)[0]

                                # Extract overlapping regions in features
                                o_top = overlap_top - top1
                                o_left = overlap_left - left1
                                o_bottom = o_top + (overlap_bottom - overlap_top)
                                o_right = o_left + (overlap_right - overlap_left)

                                o2_top = overlap_top - top2
                                o2_left = overlap_left - left2
                                o2_bottom = o2_top + (overlap_bottom - overlap_top)
                                o2_right = o2_left + (overlap_right - overlap_left)

                                output_feat1 = feat1[:, :, o_top:o_bottom, o_left:o_right].reshape(b, -1,feat1.shape[1])
                                output_feat2 = feat2[:, :, o2_top:o2_bottom, o2_left:o2_right].reshape(b, -1,feat2.shape[1])

                                label1 = pseudo_label1[:, o_top:o_bottom, o_left:o_right].reshape(b, -1)
                                label2 = pseudo_label2[:, o2_top:o2_bottom, o2_left:o2_right].reshape(b, -1)
                                conf1 = pseudo_logits1[:, o_top:o_bottom, o_left:o_right].reshape(b, -1)
                                conf2 = pseudo_logits2[:, o2_top:o2_bottom, o2_left:o2_right].reshape(b, -1)

                                if (o_bottom - o_top <= 0) or (o_right - o_left <= 0):
                                    print("batch skipped as overlap is too small")
                                    continue  # Skip batch if overlap is too small

                                output_feat1 = output_feat1.view(-1, feat1.shape[1])
                                output_feat2 = output_feat2.view(-1, feat2.shape[1])
                                label1 = label1.view(-1)
                                label2 = label2.view(-1)
                                conf1 = conf1.view(-1)
                                conf2 = conf2.view(-1)
                            else:
                                logits, features = student_model(images, feature_maps=True)
                                predicted_labels = torch.argmax(logits, dim=1)
                        except Exception as e:
                            print(f"Error during model prediction: {e}, images shape: {images.shape},"
                                  f" pseudo_labels shape: {pseudo_labels.shape}")
                            raise

                        # Compute contrastive loss
                        if contrastive_loss == 'pixel':
                            try:
                                unsup_loss = contrastive_loss_fn(
                                    features=features, labels=pseudo_labels, predict=predicted_labels
                                )
                            except Exception as e:
                                print("Error in pixel contrastive loss calculation:", e)
                                raise
                        elif contrastive_loss == 'directional':
                            unsup_loss = contrastive_loss_fn(
                                output_feat1=output_feat1, output_feat2=output_feat2,
                                pseudo_label1=label1, pseudo_label2=label2,
                                pseudo_logits1=conf1, pseudo_logits2=conf2,
                                output_ul1=feat1, output_ul2=feat2
                            )

                        elif contrastive_loss == "local":
                            unsup_loss = contrastive_loss_fn(features=features, labels=pseudo_labels)
                        elif contrastive_loss == "hybrid":
                            unsup_loss = contrastive_loss_fn(
                                output_feat1=output_feat1, output_feat2=output_feat2,
                                pseudo_label1=label1, pseudo_label2=label2,
                                pseudo_logits1=conf1, pseudo_logits2=conf2,
                                output_ul1=feat1, output_ul2=feat2
                            )
                        else:
                            raise NotImplementedError(f"Unsupported contrastive_loss: {contrastive_loss}")

                        try:
                            # Dummy supervised CE loss to keep classifier parameters active
                            dummy_ce = F.cross_entropy(logits, pseudo_labels, ignore_index=255)
                            unsup_loss += 0.0 * dummy_ce

                            optimizer.zero_grad()
                            unsup_loss.backward()
                            optimizer.step()
                            # scheduler.step()
                            semi_supervised_loss += unsup_loss.item()
                            pseudo_batches_count += 1
                        except Exception as e:
                            print(f"Error during optimization step: {e}")
                            raise
                except Exception as e:
                    print(f"Error during semi-supervised training: {e}")
                    raise

            # Update teacher model using EMA
            update_teacher_model(student_model, teacher_model, ema_decay=ema_decay)

            # Log losses to CSV
            if local_rank == 0:
                with open(train_loss_csv, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        fold + 1, epoch + 1,
                        epoch_loss / len(train_dataloader),
                        semi_supervised_loss / pseudo_batches_count if pseudo_batches_count > 0 else 0
                    ])

        # Validation
        student_model.eval()
        total_loss = 0
        iou_scores = []

        with torch.no_grad():
            for images, labels in tqdm(val_dataloader, desc=f"Validation Fold {fold + 1}"):
                images, labels = images.to(device), labels.to(device)
                outputs = student_model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)  # Shape: [N, H, W]
                for pred, label in zip(preds, labels):
                    if pred.shape != label.shape:  # Resize pred to match label shape if needed
                        pred = F.interpolate(pred.unsqueeze(0).float(), size=label.shape[-2:], mode='nearest').squeeze(0)

                    # Sanity check
                    if pred.shape != label.shape:
                        print(
                            f"Skipping IoU: shape mismatch after resize → pred: {pred.shape}, label: {label.shape}")
                        continue

                    # Ensure binary and int for compatibility
                    pred_flat = pred.long().cpu().numpy().flatten()
                    label_flat = label.long().cpu().numpy().flatten()

                    iou_scores.append(
                        jaccard_score(
                            label_flat,
                            pred_flat,
                            average='binary'
                        )
                    )

        avg_loss = total_loss / len(val_dataloader)
        avg_iou = sum(iou_scores) / len(iou_scores)

        print(f"Fold {fold + 1}, Epoch {epoch + 1} Validation Loss: {avg_loss:.4f}, mIoU: {avg_iou:.4f}")

        if avg_iou > best_miou and local_rank == 0:  # Save model based on mIoU
            best_miou = avg_iou
            torch.save(student_model.state_dict(), best_model_path)
            print(f"Best model saved with mIoU: {best_miou:.4f}")
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_folds", type=int, default=6)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=0.00012)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--neighborhood_size", type=int, default=3)
    parser.add_argument("--weight_pixel", type=float, default=1.0)
    parser.add_argument("--weight_local", type=float, default=1.0)
    parser.add_argument("--weight_directional", type=float, default=1.0)
    parser.add_argument("--supervised_loss", type=str, default="cross_entropy")
    parser.add_argument("--contrastive_loss", type=str, default="pixel")
    parser.add_argument("--dynamic_threshold", action="store_true", help="Enable dynamic thresholding")
    parser.add_argument("--class_balanced", action="store_true", help="Enable class-balanced pseudo-labeling")
    parser.add_argument("--ema_decay", type=float, default=0.99, help="EMA decay rate for teacher model")
    args = parser.parse_args()

    train(
        num_folds=args.num_folds,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        threshold=args.threshold,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        neighborhood_size=args.neighborhood_size,
        weight_pixel=args.weight_pixel,
        weight_local=args.weight_local,
        weight_directional=args.weight_directional,
        supervised_loss=args.supervised_loss,
        contrastive_loss=args.contrastive_loss,
        dynamic_threshold=args.dynamic_threshold,
        class_balanced=args.class_balanced,
        ema_decay=args.ema_decay
    )

