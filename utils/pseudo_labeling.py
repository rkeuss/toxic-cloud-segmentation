import torch
import logging


def generate_pseudo_labels(model, dataloader, threshold=0.5, device='cuda'):
    """
    Generate pseudo-labels for unlabeled data using the model's predictions.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for unlabeled data.
        threshold (float): Confidence threshold for binary segmentation.
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        list: A list of pseudo-label tensors for each image in the dataloader.
    """
    model.eval()

    total_images = 0
    with torch.no_grad():
        for images, _ in dataloader:
            if images is None or images.size(0) == 0:
                warnings.warn("Empty or invalid batch, skipping.")
                continue

            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            batch_pseudo_labels = (probs > threshold).to(dtype=torch.float32, device=outputs.device)

            if batch_pseudo_labels is None or batch_pseudo_labels.size(0) == 0:
                warnings.warn("No pseudo-labels generated for this batch.")
                continue # this warning is raised > needs to be fixed
            total_images += images.size(0)

            yield images.cpu(), batch_pseudo_labels.cpu()

    if len(pseudo_labels) != total_images:
        raise RuntimeError(f"Warning: Mismatch between image count ({total_images}) and pseudo-labels ({len(pseudo_labels)})")
