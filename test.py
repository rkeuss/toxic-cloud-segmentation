import argparse
import torch
import torch.nn.functional as F
from models.deeplabv3plus import resnet101_deeplabv3plus_imagenet
from utils.data_loader import get_ijmond_seg_dataloader_validation, get_ijmond_seg_dataset
from sklearn.metrics import jaccard_score
import numpy as np
import csv
import os
import time
import torch.distributed as dist
import torchvision.transforms as T

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

def apply_perturbations(images):
    """
    Apply perturbations to the input images for mIoU-P computation.
    """
    perturbations = [
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomHorizontalFlip(p=1.0),
        T.RandomRotation(degrees=15),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    ]
    perturbed_images = []
    for transform in perturbations:
        perturbed_images.append(torch.stack([transform(img) for img in images]))
    return perturbed_images


def evaluate_model(model, dataloader, device):
    """Evaluate the model on the test set."""
    model.eval()
    dice_scores = []
    iou_scores = []
    miou_p_scores = []
    inference_times = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # get inference time
            start_time = time.time()
            outputs = model(images)
            inference_times.append((time.time() - start_time) / len(images))

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

            # Compute mIoU-P
            perturbed_images = apply_perturbations(images)
            for perturbed in perturbed_images:
                perturbed = perturbed.to(device)
                perturbed_outputs = model(perturbed)
                perturbed_preds = torch.argmax(perturbed_outputs, dim=1)

                for perturbed_pred, mask in zip(perturbed_preds, masks):
                    if perturbed_pred.shape[-2:] != mask.shape[-2:]:
                        perturbed_pred = F.interpolate(
                            perturbed_pred.unsqueeze(0), size=mask.shape[-2:], mode='bilinear', align_corners=False
                        ).squeeze(0)

                    perturbed_pred_bin = (perturbed_pred > 0.5).float()
                    mask = mask.float()

                    perturbed_pred_np = perturbed_pred_bin.cpu().numpy().astype(np.uint8).flatten()
                    mask_np = mask.cpu().numpy().astype(np.uint8).flatten()

                    miou_p_scores.append(jaccard_score(mask_np, perturbed_pred_np, average='binary'))

    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    avg_miou_p = np.mean(miou_p_scores)
    avg_inference_time = np.mean(inference_times)

    print(f"Average Dice Coefficient (DSC): {avg_dice:.4f}")
    print(f"Average IoU (mIoU): {avg_iou:.4f}")
    print(f"Average mIoU-P: {avg_miou_p:.4f}")
    print(f"Average Inference Time per Frame: {avg_inference_time:.4f} seconds")

    return avg_dice, avg_iou, avg_miou_p, avg_inference_time


def test(supervised_loss, contrastive_loss):
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")

    test_dataset = get_ijmond_seg_dataset('data/IJMOND_SEG', split='test')
    test_idx = list(range(len(test_dataset)))
    test_dataloader = get_ijmond_seg_dataloader_validation(
        test_idx, split='test', batch_size=8, shuffle=False, rank=rank, world_size=world_size
    )

    model = resnet101_deeplabv3plus_imagenet(num_classes=2, pretrained=False)
    try:
        model.load_state_dict(
            torch.load(f'models/best_model_{supervised_loss}_{contrastive_loss}.pth', map_location=device))
    except FileNotFoundError:
        print(f"Error: Best model file not found. "
              f"Please ensure 'models/best_model_{supervised_loss}_{contrastive_loss}.pth' exists.")
        exit(1)

    model = model.to(device)
    avg_dice, avg_iou, avg_miou_p, avg_inference_time = evaluate_model(model, test_dataloader, device)

    # Save to CSV
    csv_file = "evaluation_results.csv"
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["supervised_loss", "contrastive_loss", "dice", "iou", "miou_p", "inference_time"])
        writer.writerow([supervised_loss, contrastive_loss, avg_dice, avg_iou, avg_miou_p, avg_inference_time])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervised_loss", type=str, default="cross_entropy")
    parser.add_argument("--contrastive_loss", type=str, default="pixel")
    args = parser.parse_args()
    test(
        supervised_loss=args.supervised_loss,
        contrastive_loss=args.contrastive_loss
    )
