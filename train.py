import torch
import torch.nn as nn
import torch.optim as optim
from models import deeplabv3plus
from utils.data_loader import UnlabelledDataset, get_ijmond_seg_dataloader, get_unlabelled_dataloader
from utils.pseudo_labeling import generate_pseudo_labels
from utils.contrastive_loss import PixelContrastiveLoss
from sklearn.model_selection import KFold


def train():
    num_classes = 3
    num_folds = 6
    model = deeplabv3plus.resnet101_deeplabv3plus_imagenet(num_classes=num_classes, pretrained=True)
    criterion = nn.CrossEntropyLoss()  # TODO: compare with Dice loss
    contrastive_loss = PixelContrastiveLoss()  # TODO: compare with other contrastive losses (directional/local/hybrid)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataloader = get_ijmond_seg_dataloader('data/IJMOND_SEG', split='train', batch_size=8)
    train_dataset = train_dataloader.dataset
    train_dataset_size = len(train_dataset)

    unlabeled_dataset = UnlabelledDataset(['data/IJMOND_VID/frames', 'data/RISE/frames'])
    unlabeled_dataloader = get_unlabelled_dataloader(
        ['data/IJMOND_VID/frames', 'data/RISE/frames'],
        batch_size=8,
        shuffle=True
    )

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    best_loss = float('inf')
    best_model_path = 'models/best_model.pth'
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(train_dataset_size))):
        train_subset = torch.utils.data.Subset(train_dataset, train_idx)
        val_subset = torch.utils.data.Subset(train_dataset, val_idx)

        train_dataloader = torch.utils.data.DataLoader(train_subset, batch_size=8, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_subset, batch_size=8, shuffle=False)

        for epoch in range(50):
            model.train()

            # Supervised training on labeled data
            for images, labels in train_dataloader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Generate pseudo-labels for unlabeled data
            pseudo_labels = generate_pseudo_labels(model, unlabeled_dataloader)
            unlabeled_dataset.update_pseudo_labels(pseudo_labels)

            # Semi-supervised training on unlabeled data
            for images, pseudo_labels in unlabeled_dataloader:
                outputs = model(images)
                loss = contrastive_loss(outputs, pseudo_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, labels in val_dataloader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_dataloader)
        print(f"Fold {fold + 1}, Epoch{epoch + 1} Validation Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with loss: {best_loss:.4f}")
            # now: best model in terms of validation loss, todo: best model in terms of mIoU


if __name__ == "__main__":
    train()


# in test.py
# test_dataloader = get_ijmond_seg_dataloader('data/IJMOND_SEG', split='test', batch_size=8, shuffle=False)
# Load the model
# model.load_state_dict(torch.load('model.pth'))
# evaluate using Dice similarity coefficient and IoU(/mIoU) metrics

