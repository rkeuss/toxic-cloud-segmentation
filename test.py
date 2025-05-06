import argparse
import torch
import torch.nn.functional as F
from models.deeplabv3plus import resnet101_deeplabv3plus_imagenet
from utils.data_loader import get_ijmond_seg_dataloader_validation, get_ijmond_seg_dataset
from sklearn.metrics import jaccard_score
import numpy as np
import csv
import os


def dice_coefficient(pred, target, epsilon=1e-6):
    """
    Compute Dice coefficient between binary prediction and target masks.
    Both inputs must be 1D, binary (0 or 1) arrays or tensors.
    """
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)

    pred = pred.contiguous().view(-1).float()
    target = target.contiguous().view(-1).float()

    intersection = (pred * target).sum()
    dice = (2.0 * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)
    return dice.item()


def evaluate_model(model, dataloader, device):
    """Evaluate the model on the test set."""
    model.eval()
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for pred, mask in zip(preds, masks):
                # Ensure shape match before flattening
                if pred.shape[-2:] != mask.shape[-2:]:
                    pred = F.interpolate(pred.unsqueeze(0), size=mask.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)

                pred_bin = (pred > 0.5).float() # Ensure binary masks
                mask = mask.float()

                # Flatten for metric computation
                pred_np = pred_bin.cpu().numpy().astype(np.uint8).flatten()
                mask_np = mask.cpu().numpy().astype(np.uint8).flatten()

                dice_scores.append(dice_coefficient(pred_np, mask_np))
                iou_scores.append(jaccard_score(mask_np, pred_np, average='binary'))

    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)

    print(f"Average Dice Coefficient (DSC): {avg_dice:.4f}")
    print(f"Average IoU (mIoU): {avg_iou:.4f}")

    return avg_dice, avg_iou


def test(supervised_loss, contrastive_loss):
    num_classes = 2

    test_dataset = get_ijmond_seg_dataset('data/IJMOND_SEG', split='test')
    test_idx = list(range(len(test_dataset)))
    test_dataloader = get_ijmond_seg_dataloader_validation(
        test_idx, split='test', batch_size=8, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet101_deeplabv3plus_imagenet(num_classes=num_classes, pretrained=False)
    try:
        model.load_state_dict(
            torch.load(f'models/best_model_{supervised_loss}_{contrastive_loss}.pth', map_location=device))
    except FileNotFoundError:
        print("Error: Best model file not found. Please ensure 'models/best_model.pth' exists.")
        exit(1)

    model = model.to(device)
    avg_dice, avg_iou = evaluate_model(model, test_dataloader, device)

    # Save to CSV
    csv_file = "evaluation_results.csv"
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["supervised_loss", "contrastive_loss", "dice", "iou"])
        writer.writerow([supervised_loss, contrastive_loss, avg_dice, avg_iou])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervised_loss", type=str, default="cross_entropy")
    parser.add_argument("--contrastive_loss", type=str, default="pixel")
    args = parser.parse_args()
    test(
        supervised_loss=args.supervised_loss,
        contrastive_loss=args.contrastive_loss
    )
