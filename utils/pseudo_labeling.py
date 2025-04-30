import torch

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
    pseudo_labels = []
    model = model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)  # Assuming dataloader returns (images, _)
            outputs = model(images)
            probs = torch.sigmoid(outputs)  # For binary segmentation

            batch_pseudo_labels = (probs > threshold).long()

            if batch_pseudo_labels is None or batch_pseudo_labels.size(0) == 0:
                print("Warning: No pseudo-labels generated for this batch.")
                continue

            pseudo_labels.extend(batch_pseudo_labels.cpu())

    return pseudo_labels
