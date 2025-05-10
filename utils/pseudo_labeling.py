import torch
import logging

logging.basicConfig(
    level=logging.DEBUG,
    filename='logs/debug.log',  # File path in the logs directory
    filemode='w',  # Overwrite the file each time the script runs
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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

    total_images = 0
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                raise Warning('Warning: Encountered a None batch, skipping')

            images, _ = batch
            if images is None or images.size(0) == 0:
                raise Warning("Warning: Empty or invalid batch, skipping.")

            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)  # For binary segmentation

            logging.debug(f"Model outputs: {outputs}")
            logging.debug(f"Probabilities: {probs}")
            logging.debug(f"Threshold: {threshold}")
            logging.debug(f"Predictions above threshold: {(probs > threshold).sum().item()}")

            batch_pseudo_labels = (probs > threshold).long()

            if batch_pseudo_labels is None or batch_pseudo_labels.size(0) == 0:
                raises Warning("Warning: No pseudo-labels generated for this batch.")
                continue # this warning is raised > needs to be fixed

            for pseudo in batch_pseudo_labels.cpu():
                pseudo_labels.append(pseudo)

            total_images += images.size(0)

    if len(pseudo_labels) != total_images:
        raise Warning(f"Warning: Mismatch between image count ({total_images}) and pseudo-labels ({len(pseudo_labels)})")

    return pseudo_labels
