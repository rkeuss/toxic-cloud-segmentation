import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)  # Apply sigmoid to get probabilities
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice


class PixelContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(PixelContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Pixel Contrastive Loss as described in C3-SemiSeg.
        Args:
            features: Tensor of shape (N, C, H, W), feature maps.
            labels: Tensor of shape (N, H, W), ground truth or pseudo-labels.
        """
        N, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(-1, C)  # Flatten to (N*H*W, C)
        labels = labels.view(-1)  # Flatten to (N*H*W)

        mask = labels.unsqueeze(1) == labels.unsqueeze(0)  # Create similarity mask
        mask = mask.float()

        features = F.normalize(features, dim=1)  # Normalize features
        logits = torch.matmul(features, features.T) / self.temperature  # Cosine similarity
        exp_logits = torch.exp(logits) * mask
        loss = -torch.log(exp_logits / (exp_logits.sum(dim=1, keepdim=True) + 1e-6))  # Avoid division by zero
        return loss.mean()


class LocalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, neighborhood_size=5):
        super(LocalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.neighborhood_size = neighborhood_size

    def forward(self, features, labels):
        """
        Local Contrastive Loss as described in pseudo_label_contrastive_training.
        Args:
            features: Tensor of shape (N, C, H, W), feature maps.
            labels: Tensor of shape (N, H, W), ground truth or pseudo-labels.
        """
        N, C, H, W = features.shape
        features = F.normalize(features, dim=1)  # Normalize features
        loss = 0.0

        for i in range(H):
            for j in range(W):
                neighborhood = features[:, :, max(0, i - self.neighborhood_size):min(H, i + self.neighborhood_size + 1),
                                        max(0, j - self.neighborhood_size):min(W, j + self.neighborhood_size + 1)]
                center = features[:, :, i, j].unsqueeze(-1).unsqueeze(-1)
                logits = torch.sum(center * neighborhood, dim=1) / self.temperature
                labels_center = labels[:, i, j].unsqueeze(-1).unsqueeze(-1)
                mask = (labels_center == labels[:, max(0, i - self.neighborhood_size):min(H, i + self.neighborhood_size + 1),
                                                max(0, j - self.neighborhood_size):min(W, j + self.neighborhood_size + 1)]).float()
                exp_logits = torch.exp(logits) * mask
                loss += -torch.log(exp_logits / (exp_logits.sum(dim=1, keepdim=True) + 1e-6)).mean()  # Avoid division by zero

        return loss / (H * W)


class DirectionalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(DirectionalContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels, directions):
        """
        Directional Contrastive Loss as described in Context-Aware-Consistency.
        Args:
            features: Tensor of shape (N, C, H, W), feature maps.
            labels: Tensor of shape (N, H, W), ground truth or pseudo-labels.
            directions: Tensor of shape (N, 2, H, W), directional vectors.
        """
        N, C, H, W = features.shape
        features = F.normalize(features, dim=1)  # Normalize features
        loss = 0.0

        for i in range(H):
            for j in range(W):
                direction = directions[:, :, i, j]  # Directional vector at (i, j)
                neighbor_i = i + direction[:, 0].long()
                neighbor_j = j + direction[:, 1].long()

                valid_mask = (neighbor_i >= 0) & (neighbor_i < H) & (neighbor_j >= 0) & (neighbor_j < W)
                neighbor_i = neighbor_i[valid_mask]
                neighbor_j = neighbor_j[valid_mask]

                if len(neighbor_i) == 0:
                    continue

                neighbor_features = features[:, :, neighbor_i, neighbor_j]
                center_features = features[:, :, i, j].unsqueeze(-1)
                logits = torch.sum(center_features * neighbor_features, dim=1) / self.temperature
                mask = (labels[:, i, j] == labels[:, neighbor_i, neighbor_j]).float()
                exp_logits = torch.exp(logits) * mask
                loss += -torch.log(exp_logits / (exp_logits.sum(dim=1, keepdim=True) + 1e-6)).mean()

        return loss / (H * W)


class HybridContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, neighborhood_size=5, weight_pixel=1.0, weight_local=1.0, weight_directional=1.0):
        """
        Hybrid Contrastive Loss combining Pixel, Local, and Directional Contrastive Losses.
        Args:
            temperature (float): Temperature parameter for contrastive loss.
            neighborhood_size (int): Size of the local neighborhood for Local Contrastive Loss.
            weight_pixel (float): Weight for Pixel Contrastive Loss.
            weight_local (float): Weight for Local Contrastive Loss.
            weight_directional (float): Weight for Directional Contrastive Loss.
        """
        super(HybridContrastiveLoss, self).__init__()
        self.pixel_loss = PixelContrastiveLoss(temperature)
        self.local_loss = LocalContrastiveLoss(temperature, neighborhood_size)
        self.directional_loss = DirectionalContrastiveLoss(temperature)
        self.weight_pixel = weight_pixel
        self.weight_local = weight_local
        self.weight_directional = weight_directional

    def forward(self, features, labels, directions=None):
        """
        Compute the Hybrid Contrastive Loss.
        Args:
            features: Tensor of shape (N, C, H, W), feature maps.
            labels: Tensor of shape (N, H, W), ground truth or pseudo-labels.
            directions: Tensor of shape (N, 2, H, W), directional vectors (optional, required for directional loss).
        Returns:
            torch.Tensor: Combined loss value.
        """
        loss_pixel = self.pixel_loss(features, labels)
        loss_local = self.local_loss(features, labels)
        loss_directional = 0.0

        if directions is not None:
            loss_directional = self.directional_loss(features, labels, directions)

        combined_loss = (
            self.weight_pixel * loss_pixel +
            self.weight_local * loss_local +
            self.weight_directional * loss_directional
        )
        return combined_loss
