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

# TODO: hyperparameter tuning (num_folds, num_epochs, batch_size, threshold, learning rate (lr), temperature,
#  neighborhood_size, weights in hybrid loss)
def train(
        num_folds=6, num_epochs=50, batch_size=8, threshold=0.5,
        learning_rate=0.001, temperature=0.1, neighborhood_size=5,
        weight_pixel=1.0, weight_local=1.0, weight_directional=1.0,
        supervised_loss='cross_entropy', contrastive_loss='pixel'
):
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = deeplabv3plus.resnet101_deeplabv3plus_imagenet(num_classes=num_classes, pretrained=True)
    model = model.to(device)  # Ensure model is on the correct device
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if supervised_loss == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif supervised_loss == 'dice':
        criterion = losses.DiceLoss()
    else:
        raise ValueError(f"Unsupported supervised_loss: {supervised_loss}")

    if contrastive_loss == 'pixel':
        contrastive_loss_fn = losses.PixelContrastiveLoss(temperature=temperature)
    elif contrastive_loss == 'local':
        contrastive_loss_fn = losses.LocalContrastiveLoss(temperature=temperature, neighborhood_size=5)
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
        shuffle=True
    )
    unlabeled_dataset = unlabeled_dataloader.dataset

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    best_model_path = f'models/best_model_{supervised_loss}_{contrastive_loss}.pth'
    best_miou = 0.0
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(train_dataset_size))):
        train_dataloader = data_loader.get_ijmond_seg_dataloader_train(
            train_idx, split='train', batch_size=batch_size, shuffle=True
        )
        val_dataloader = data_loader.get_ijmond_seg_dataloader_validation(
            val_idx, split='train', batch_size=batch_size, shuffle=True
        )

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0

            # Supervised training on labeled data (weak augmentations)
            for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} - Supervised"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1} Supervised Loss: {epoch_loss / len(train_dataloader):.4f}")

            # Generate pseudo-labels for unlabeled data
            try:
                pseudo_labels = generate_pseudo_labels(
                    model, unlabeled_dataloader, threshold=threshold, device=device
                )
                unlabeled_dataset.update_pseudo_labels(pseudo_labels)
            except Exception as e:
                print(f"Error generating pseudo-labels: {e}")
                continue

            # Semi-supervised training on pseudo-labeled data (strong augmentations)
            for images, pseudo_labels in tqdm(
                    unlabeled_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} - Semi-supervised"
            ):
                images, pseudo_labels = images.to(device), pseudo_labels.to(device)
                outputs = model(images)
                loss = contrastive_loss_fn(outputs, pseudo_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Validation
        model.eval()
        total_loss = 0
        iou_scores = []

        with torch.no_grad():
            for images, labels in tqdm(val_dataloader, desc=f"Validation Fold {fold + 1}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
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

        if avg_iou > best_miou:  # Save model based on mIoU
            best_miou = avg_iou
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_folds", type=int, default=6) # todo
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=14)
    parser.add_argument("--threshold", type=float, default=0.5) # todo
    parser.add_argument("--learning_rate", type=float, default=0.001) # todo
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--neighborhood_size", type=int, default=5) # todo
    parser.add_argument("--weight_pixel", type=float, default=1.0)
    parser.add_argument("--weight_local", type=float, default=1.0)
    parser.add_argument("--weight_directional", type=float, default=1.0)
    parser.add_argument("--supervised_loss", type=str, default="cross_entropy")
    parser.add_argument("--contrastive_loss", type=str, default="pixel")
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
        contrastive_loss=args.contrastive_loss
    )