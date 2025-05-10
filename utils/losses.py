import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Code copied from C3-SemiSeg
class MaskCrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='none'):
        super(MaskCrossEntropyLoss2d, self).__init__()
        self.ignore_index = ignore_index
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target, mask=None):
        loss = self.CE(output, target)
        ignore_index_mask = (target != self.ignore_index).float()
        if mask is None:
            mask = ignore_index_mask
        else:
            mask = mask * ignore_index_mask
        if mask.nonzero().size(0) != 0:
            loss = (loss * mask).mean() * mask.numel() / mask.nonzero().size(0)
        else:
            loss = (loss * mask).mean() * mask.numel() / (mask.nonzero().size(0) + 1e-9)
        return loss


# Code based on code paper 'local contrastive loss'
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, labels):
        """
        Calculate a dice loss defined as `1-foreground_dice`. Default mode assumes that the 0 label
        denotes background and the remaining labels are foreground.
        input params:
            logits: Network output before softmax
            labels: ground truth label masks
            epsilon: A small constant to avoid division by 0
        returns:
            loss: Dice loss with background
        """
        prediction = F.softmax(logits, dim=1)

        intersection = torch.sum(prediction * labels, dim=(2, 3))
        l = torch.sum(prediction, dim=(2, 3))
        r = torch.sum(labels, dim=(2, 3))

        dices_per_subj = 2 * intersection / (l + r + self.smooth)
        loss = 1 - torch.mean(dices_per_subj)
        return loss


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
        # TODO improve (x positively labelled pixels per image are selected ?)
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
        # TODO improve
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
    def __init__(self, temperature=0.1, neighborhood_size=3, weight_pixel=1.0, weight_local=1.0, weight_directional=1.0):
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


# Code copied from C3-SemiSeg
class PixelContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, neg_num=256, memory_bank=None, mining=True):
        super(PixelContrastiveLoss, self).__init__()

        self.temperature = temperature
        self.base_temperature = 0.1
        self.ignore_label = 255
        self.max_samples = int(neg_num * 19)
        self.max_views = neg_num
        self.memory_bank = memory_bank
        self.mining = mining

    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        # filter each image, to find what class they have num > self.max_view pixel
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x > 0 and x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        # n_view = self.max_samples // total_classes
        # n_view = min(n_view, self.max_views)
        n_view = self.max_views
        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]
            # hard: predict wrong
            for cls_id in this_classes:
                if self.mining:
                    hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                    easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                    num_hard = hard_indices.shape[0]
                    num_easy = easy_indices.shape[0]

                    if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                        num_hard_keep = n_view // 2
                        num_easy_keep = n_view - num_hard_keep
                    elif num_hard >= n_view / 2:
                        num_easy_keep = num_easy
                        num_hard_keep = n_view - num_easy_keep
                    elif num_easy >= n_view / 2:
                        num_hard_keep = num_hard
                        num_easy_keep = n_view - num_hard_keep
                    else:
                        # Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                        raise Exception

                    perm = torch.randperm(num_hard)
                    hard_indices = hard_indices[perm[:num_hard_keep]]
                    perm = torch.randperm(num_easy)
                    easy_indices = easy_indices[perm[:num_easy_keep]]
                    indices = torch.cat((hard_indices, easy_indices), dim=0)

                    X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                    y_[X_ptr] = cls_id
                    X_ptr += 1
                else:
                    num_indice = (this_y_hat == cls_id).nonzero()
                    number = num_indice.shape[0]
                    perm = torch.randperm(number)
                    indices = num_indice[perm[:n_view]]
                    X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                    y_[X_ptr] = cls_id
                    X_ptr += 1
        return X_, y_

    def _hard_pair_sample_mining(self, X, X_2, y_hat, y, mask=None):

        batch_size, feat_dim = X.shape[0], X.shape[-1]
        if mask is not None:
            y_hat = mask * y_hat + (1 - mask) * 255
        classes = []
        total_classes = 0
        # filter each image, to find what class they have num > self.max_view pixel
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x > 0 and x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None, None

        # n_view = self.max_samples // total_classes
        # n_view = min(n_view, self.max_views)
        n_view = self.max_views
        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        X_2_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]
            # hard: predict wrong
            for cls_id in this_classes:
                if self.mining:
                    hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                    easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                    num_hard = hard_indices.shape[0]
                    num_easy = easy_indices.shape[0]

                    if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                        num_hard_keep = n_view // 2
                        num_easy_keep = n_view - num_hard_keep
                    elif num_hard >= n_view / 2:
                        num_easy_keep = num_easy
                        num_hard_keep = n_view - num_easy_keep
                    elif num_easy >= n_view / 2:
                        num_hard_keep = num_hard
                        num_easy_keep = n_view - num_hard_keep
                    else:
                        # Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                        raise Exception

                    perm = torch.randperm(num_hard)
                    hard_indices = hard_indices[perm[:num_hard_keep]]
                    perm = torch.randperm(num_easy)
                    easy_indices = easy_indices[perm[:num_easy_keep]]
                    indices = torch.cat((hard_indices, easy_indices), dim=0)

                    X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                    X_2_[X_ptr, :, :] = X_2[ii, indices, :].squeeze(1)
                    y_[X_ptr] = cls_id
                    X_ptr += 1
                else:

                    num_indice = (this_y_hat == cls_id).nonzero()
                    number = num_indice.shape[0]
                    perm = torch.randperm(number)
                    indices = num_indice[perm[:n_view]]
                    X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                    X_2_[X_ptr, :, :] = X_2[ii, indices, :].squeeze(1)
                    y_[X_ptr] = cls_id
                    X_ptr += 1

        return X_, X_2_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        # the reason for use unbind is that they need to repeat mask n_view times later
        # 对于每个class的样本，它有(n_view - 1)* ptr中有cls的图片的个数 个 正样本
        # 有 n_view * (ptr- cls的次数)个负样本
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        # max是自身
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask
        # set self = 0
        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def _contrastive_pair(self, feats_, feats_t, labels_, labels_t):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]
        anchor_num_t, n_view_t = feats_t.shape[0], feats_t.shape[1]
        labels_ = labels_.contiguous().view(-1, 1)
        labels_t = labels_t.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_t, 0, 1)).float().cuda()

        contrast_count = n_view_t
        # the reason for use unbind is that they need to repeat mask n_view times later
        # 对于每个class的样本，它有(n_view - 1)* ptr中有cls的图片的个数 个 正样本
        # 有 n_view * (ptr- cls的次数)个负样本
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)
        contrast_feature_t = torch.cat(torch.unbind(feats_t, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = n_view

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, torch.transpose(contrast_feature_t.detach(), 0, 1)),
            self.temperature)
        # max是自身
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask
        # set self = 0
        if anchor_num == anchor_num_t:
            logits_mask = torch.ones_like(mask).scatter_(1,
                                                         torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                         0)
            mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, feats_t=None, labels=None, predict=None, cb_mask=None):
        # feat from student, feats_2 from teacher if have

        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)
        if cb_mask is not None:
            mask_batch = cb_mask.shape[0]
            batch, _, h, w = feats.shape
            cb_mask = F.interpolate(cb_mask.float(), (h, w), mode='nearest')
            if mask_batch != batch: # when use both labeled and unlabeld data for loss, labeled do not need any ignore
                labeled_confidence_mask = torch.ones_like(cb_mask).to(cb_mask.dtype)
                cb_mask = torch.cat([labeled_confidence_mask, cb_mask])
            cb_mask = cb_mask.view(batch, -1)
        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])
        feats = F.normalize(feats, dim=2)
        if feats_t is None:
            feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)
            if feats_ is None:
                print('input do not have enough label, ignore (happened in beginning of BDD)')
                loss = 0 * feats.mean()
                return loss

            if self.memory_bank is not None:
                self.memory_bank.dequeue_and_enqueue(feats_, labels_)
                feats_t_, labels_t_ = self.memory_bank.get_valid_feat_and_label()

                loss = self._contrastive_pair(feats_, feats_t_, labels_, labels_t_)
            else:
                loss = self._contrastive(feats_, labels_)
        else:

            feats_t = feats_t.permute(0, 2, 3, 1)
            feats_t = feats_t.contiguous().view(feats_t.shape[0], -1, feats_t.shape[-1])
            feats_t = F.normalize(feats_t, dim=2)
            feats_, feats_t_, labels_ = self._hard_pair_sample_mining(feats, feats_t, labels, predict, cb_mask)
            if feats_ is None:
                print('input do not have enough label, ignore (happened in beginning of BDD)')
                loss = 0 * feats.mean()
                return loss

            if self.memory_bank is not None:

                self.memory_bank.dequeue_and_enqueue(feats_t_, labels_)
                feats_t_, labels_t_ = self.memory_bank.get_valid_feat_and_label()

                loss = self._contrastive_pair(feats_, feats_t_, labels_, labels_t_)
            else:
                loss = self._contrastive_pair(feats_, feats_t_, labels_, labels_)
        return loss





# Not used
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss


class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss


class EntropyLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(EntropyLoss, self).__init__()
        self.CE = nn.CrossEntropyLoss
        self.ignore_index = ignore_index

    def forward(self, output, target):
        b = F.softmax(output, dim=1) * F.log_softmax(output, dim=1)
        b = b * (target != self.ignore_index).long()[:, None, ...]
        b = -1.0 * b.sum() / (b.nonzero().size(0))
        return b


def sharpen(p, temp=0.5):
    pt = p ** (1 / temp)
    targets_u = pt / pt.sum(dim=1, keepdim=True)
    targets_u = targets_u.detach()
    return targets_u


def label_smooth(onehot, cls=19, eta=0.1):
    low_confidence = eta / cls
    new_label = (1 - eta) * onehot + low_confidence
    return new_label


def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target


def log_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    return float(1 - np.exp(-5.0 * current / rampup_length))