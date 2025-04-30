import torch
import torch.nn.functional as F
from models.deeplabv3plus import resnet101_deeplabv3plus_imagenet
from utils.data_loader import get_ijmond_seg_dataloader
from sklearn.metrics import jaccard_score
import numpy as np


def dice_coefficient(pred, target):
    """Calculate Dice similarity coefficient."""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2.0 * intersection) / (pred.sum() + target.sum() + 1e-6)


def evaluate_model(model, dataloader):
    """Evaluate the model on the test set."""
    model.eval()
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(model.device)
            masks = masks.to(model.device)

            outputs = model(images)
            preds = torch.sigmoid(outputs)

            for pred, mask in zip(preds, masks):
                pred = pred.cpu().numpy().flatten()
                mask = mask.cpu().numpy().flatten()

                dice_scores.append(dice_coefficient(torch.tensor(pred), torch.tensor(mask)).item())
                iou_scores.append(jaccard_score(mask, pred > 0.5, average='binary'))

    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)

    print(f"Average Dice Coefficient (DSC): {avg_dice:.4f}")
    print(f"Average IoU (mIoU): {avg_iou:.4f}")


if __name__ == "__main__":
    supervised_loss = 'cross_entropy'
    contrastive_loss = 'pixel'

    num_classes = 2
    test_dataloader = get_ijmond_seg_dataloader('data/IJMOND_SEG', split='test', batch_size=8, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet101_deeplabv3plus_imagenet(num_classes=num_classes, pretrained=False)
    try:
        model.load_state_dict(torch.load(f'models/best_model_{supervised_loss}_{contrastive_loss}.pth', map_location=device))
    except FileNotFoundError:
        print("Error: Best model file not found. Please ensure 'models/best_model.pth' exists.")
        exit(1)

    model = model.to(device)
    evaluate_model(model, test_dataloader)
