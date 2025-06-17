from collections import defaultdict
import json
import torch
import torch.nn.functional as F
from models.deeplabv3plus import resnet101_deeplabv3plus_imagenet
from utils.data_loader import get_ijmond_seg_dataloader_validation, get_ijmond_seg_dataset
from sklearn.metrics import jaccard_score
import numpy as np
import csv
import os
import time
import torchvision.transforms as T

def group_images_by_opacity(annotation_json_path, valid_image_ids):
    with open(annotation_json_path, 'r') as f:
        data = json.load(f)

    image_id_to_opacity = defaultdict(set)
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in valid_image_ids:
            continue
        category_id = ann['category_id']
        image_id_to_opacity[image_id].add(category_id)

    high_opacity_ids = set()
    low_opacity_ids = set()

    for image_id, categories in image_id_to_opacity.items():
        if 1 in categories and 2 not in categories:
            high_opacity_ids.add(image_id)
        elif 2 in categories and 1 not in categories:
            low_opacity_ids.add(image_id)

    return high_opacity_ids, low_opacity_ids


def get_dataloader_by_image_ids(dataset, image_ids, batch_size):
    index_subset = [
        i for i, img_info in enumerate(dataset.coco.dataset['images'])
        if img_info['id'] in image_ids
    ]
    return get_ijmond_seg_dataloader_validation(
        index_subset, split='test', batch_size=batch_size, shuffle=False
    )

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
    device = torch.device("cpu")

    test_dataset = get_ijmond_seg_dataset('data/IJMOND_SEG/cropped', split='test')
    high_opacity_ids, low_opacity_ids = group_images_by_opacity("data/IJMOND_SEG/cropped/cropped_annotations.json")

    # Get dataloaders for each group
    high_dataloader = get_dataloader_by_image_ids(test_dataset, high_opacity_ids, 8)
    low_dataloader = get_dataloader_by_image_ids(test_dataset, low_opacity_ids, 8)

    try:
        model = resnet101_deeplabv3plus_imagenet(num_classes=2, pretrained=False)
        checkpoint_path = f'/Users/rkeuss/PycharmProjects/toxic-cloud-segmentation/models/best_model_{supervised_loss}_{contrastive_loss}.pth'

        state_dict = torch.load(checkpoint_path, map_location=device)
        # Handle DataParallel or DDP "module." prefix
        if any(k.startswith('module.') for k in state_dict.keys()):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k.replace('module.', '')] = v
            state_dict = new_state_dict
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"Error: Best model file not found: {checkpoint_path}")
        exit(1)
    except RuntimeError as e:
        print(f"RuntimeError when loading state dict: {e}")
        exit(1)

    model = model.to(device)

    print("Evaluating on high-opacity smoke...")
    avg_dice_high, avg_iou_high, avg_miou_p_high, avg_inference_time_high = evaluate_model(model, high_dataloader, device)
    print("\nEvaluating on low-opacity smoke...")
    avg_dice_low, avg_iou_low, avg_miou_p_low, avg_inference_time_low = evaluate_model(model, low_dataloader, device)

    # Save to CSV
    csv_file = "evaluation_results_opacity.csv"
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                ["supervised_loss", "contrastive_loss", "dice_high", "dice_low",
                 "miou_high", "miou_low", "miou_p_high", "miou_p_low",
                 "inference_time_high", "inference_time_low"
                 ])
        writer.writerow([
            supervised_loss, contrastive_loss, avg_dice_high, avg_dice_low,
            avg_iou_high, avg_iou_low, avg_miou_p_high, avg_miou_p_low,
            avg_inference_time_high, avg_inference_time_low
        ])


if __name__ == "__main__":
    for supervised_loss in ["cross_entropy", "dice"]:
        for contrastive_loss in ["pixel", "local", "directional", "hybrid"]:
            print(f"Testing with supervised_loss={supervised_loss}, contrastive_loss={contrastive_loss}")
            test(
                supervised_loss=supervised_loss,
                contrastive_loss=contrastive_loss
            )