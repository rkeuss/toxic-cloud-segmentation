import torch
import logging
import warnings
import numpy as np

def generate_pseudo_labels(
        teacher_model, dataloader, threshold=0.5, device='cuda',
        dynamic_threshold=True, class_balanced=True
):
    """
    Generate pseudo-labels for unlabeled data using the teacher model's predictions.

    Args:
        teacher_model (torch.nn.Module): The trained teacher model.
        dataloader (torch.utils.data.DataLoader): DataLoader for unlabeled data.
        threshold (float): Confidence threshold for binary segmentation.
        device (str): Device to run the model on ('cuda' or 'cpu').
        dynamic_threshold (bool): Whether to use dynamic thresholding based on confidence distribution.
        class_balanced (bool): Whether to apply class-specific thresholds for balanced pseudo-labels.

    Yields:
        tuple: A batch of images and their corresponding pseudo-labels.
    """
    teacher_model.eval()  # Ensure teacher model is in evaluation mode
    for images, _ in dataloader:
        if images is None or images.size(0) == 0:
            warnings.warn("Empty or invalid batch, skipping.")
            continue

        images = images.to(device)
        with torch.no_grad():
            logits = teacher_model(images)
            probs = torch.softmax(logits, dim=1)
            confidence, pseudo_labels = torch.max(probs, dim=1)

            # Dynamic thresholding
            if dynamic_threshold:
                threshold = torch.quantile(probs, 0.75).item()  # Use 75th percentile as threshold

            # Class-balanced pseudo-labels
            if class_balanced:
                class_thresholds = compute_class_thresholds(probs, percentile=75)
                print("Class thresholds:", {c: round(class_thresholds[c], 4) for c in range(probs.shape[1])})
                for c in range(probs.shape[1]):
                    pseudo_labels[(confidence < class_thresholds[c]) & (pseudo_labels == c)] = 255  # Ignore low-confidence
            else:
                pseudo_labels[confidence < threshold] = 255  # Use ignore index for low-confidence regions


        yield images.cpu(), pseudo_labels.cpu()

        del logits, probs, confidence, pseudo_labels
        torch.cuda.empty_cache()
    teacher_model.train()

def compute_class_thresholds(probs, percentile=75):
    """
    Compute class-specific thresholds based on the confidence distribution.

    Args:
        probs (torch.Tensor): Probability tensor of shape (N, C, H, W).
        percentile (int): Percentile to use for thresholding.

    Returns:
        list: Class-specific thresholds.
    """
    thresholds = []
    for c in range(probs.shape[1]):
        class_probs = probs[:, c, :, :].flatten()
        thresholds.append(torch.quantile(class_probs, percentile / 100.0).item())
    return thresholds
