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
        mask = mask.to(loss.device)
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
    def __init__(self, temperature=0.1, no_of_pos_eles=3, no_of_neg_eles=3, local_loss_exp_no=0, lamda_local=0.1):
        super(LocalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.no_of_pos_eles = no_of_pos_eles
        self.no_of_neg_eles = no_of_neg_eles
        self.local_loss_exp_no = local_loss_exp_no
        self.lamda_local = lamda_local

    def cosine_similarity(self, x1, x2):
        return F.cosine_similarity(x1, x2, dim=-1)

    def forward(self, features, labels):
        B, C, H, W = features.shape
        loss = 0.0
        features = F.normalize(features, dim=1)
        labels = F.interpolate(labels.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1).long()

        for b in range(B):
            for cls in torch.unique(labels[b]):
                if cls == 255:
                    continue

                mask = labels[b] == cls
                if mask.sum() < self.no_of_pos_eles:
                    continue

                pos_indices = mask.nonzero(as_tuple=False)
                perm = torch.randperm(pos_indices.shape[0])[:self.no_of_pos_eles]
                pos_samples = pos_indices[perm]

                pos_embeddings = torch.stack([features[b, :, y, x] for y, x in pos_samples])
                pos_mean = pos_embeddings.mean(dim=0, keepdim=True)

                neg_embeddings = []
                for neg_cls in torch.unique(labels[b]):
                    if neg_cls == cls or neg_cls == 255:
                        continue
                    neg_mask = labels[b] == neg_cls
                    if neg_mask.sum() == 0:
                        continue
                    neg_indices = neg_mask.nonzero(as_tuple=False)
                    perm = torch.randperm(neg_indices.shape[0])[:self.no_of_neg_eles]
                    neg_samples = neg_indices[perm]
                    for y, x in neg_samples:
                        neg_embeddings.append(features[b, :, y, x])
                if len(neg_embeddings) == 0:
                    continue
                neg_embeddings = torch.stack(neg_embeddings)

                for emb in pos_embeddings:
                    sim_pos = torch.exp(self.cosine_similarity(emb.unsqueeze(0), pos_mean) / self.temperature)
                    sim_negs = torch.exp(self.cosine_similarity(emb.unsqueeze(0), neg_embeddings) / self.temperature).sum()
                    loss += -torch.log(sim_pos / (sim_pos + sim_negs + 1e-8))

        # Inter-image term if enabled
        if self.local_loss_exp_no == 1 and B >= 2:
            for b1 in range(B):
                for b2 in range(B):
                    if b1 == b2:
                        continue
                    for cls in torch.unique(labels[b1]):
                        if cls == 255 or cls not in labels[b2]:
                            continue

                        mask1 = labels[b1] == cls
                        mask2 = labels[b2] == cls
                        if mask1.sum() < self.no_of_pos_eles or mask2.sum() < self.no_of_pos_eles:
                            continue

                        pos1 = mask1.nonzero(as_tuple=False)
                        pos2 = mask2.nonzero(as_tuple=False)
                        perm1 = torch.randperm(pos1.shape[0])[:self.no_of_pos_eles]
                        perm2 = torch.randperm(pos2.shape[0])[:self.no_of_pos_eles]

                        pos_embeddings_1 = torch.stack([features[b1, :, y, x] for y, x in pos1[perm1]])
                        pos_embeddings_2 = torch.stack([features[b2, :, y, x] for y, x in pos2[perm2]])

                        mean_2 = pos_embeddings_2.mean(dim=0, keepdim=True)

                        neg_embeddings = []
                        for neg_cls in torch.unique(labels[b1]):
                            if neg_cls == cls or neg_cls == 255:
                                continue
                            neg_mask = labels[b1] == neg_cls
                            if neg_mask.sum() == 0:
                                continue
                            neg_indices = neg_mask.nonzero(as_tuple=False)
                            perm = torch.randperm(neg_indices.shape[0])[:self.no_of_neg_eles]
                            neg_samples = neg_indices[perm]
                            for y, x in neg_samples:
                                neg_embeddings.append(features[b1, :, y, x])
                        if len(neg_embeddings) == 0:
                            continue
                        neg_embeddings = torch.stack(neg_embeddings)

                        for emb in pos_embeddings_1:
                            sim_pos = torch.exp(self.cosine_similarity(emb.unsqueeze(0), mean_2) / self.temperature)
                            sim_negs = torch.exp(self.cosine_similarity(emb.unsqueeze(0), neg_embeddings) / self.temperature).sum()
                            loss += -torch.log(sim_pos / (sim_pos + sim_negs + 1e-8))

        return self.lamda_local * loss / (B * self.no_of_pos_eles + 1e-8)


class DirectionalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, pos_thresh=0.7, selected_num=2000):
        super(DirectionalContrastiveLoss, self).__init__()
        self.temp = temperature
        self.pos_thresh_value = pos_thresh
        self.selected_num = selected_num
        self.feature_bank = []
        self.pseudo_label_bank = []
        self.step_count = 0
        self.step_save = 5

    def forward(self, output_feat1, output_feat2, pseudo_label1, pseudo_label2,
                pseudo_logits1, pseudo_logits2, output_ul1, output_ul2):
        eps = 1e-8

        output_feat1 = output_feat1.detach().clone().requires_grad_(True)
        output_feat2 = output_feat2.detach().clone().requires_grad_(True)
        output_ul1 = output_ul1.detach().clone()
        output_ul2 = output_ul2.detach().clone()

        pos1 = (output_feat1 * output_feat2.detach()).sum(-1, keepdim=True) / self.temp
        pos2 = (output_feat1.detach() * output_feat2).sum(-1, keepdim=True) / self.temp

        b, c, h, w = output_ul1.size()
        output_ul1_flat = output_ul1.permute(0, 2, 3, 1).contiguous().view(b * h * w, c)
        output_ul2_flat = output_ul2.permute(0, 2, 3, 1).contiguous().view(b * h * w, c)

        pseudo_label1_flat = pseudo_label1.view(-1)
        pseudo_label2_flat = pseudo_label2.view(-1)

        max_idx1 = pseudo_label1_flat.size(0)
        max_idx2 = pseudo_label2_flat.size(0)

        selected_num1 = min(self.selected_num, max_idx1)
        selected_num2 = min(self.selected_num, max_idx2)

        selected_idx1 = torch.randperm(max_idx1, device=pseudo_label1.device)[:selected_num1]
        selected_idx2 = torch.randperm(max_idx2, device=pseudo_label2.device)[:selected_num2]

        output_ul_flatten_selected = torch.cat([
            output_ul1_flat[selected_idx1],
            output_ul2_flat[selected_idx2]
        ], dim=0)

        pseudo_label_flatten_selected = torch.cat([
            pseudo_label1_flat[selected_idx1],
            pseudo_label2_flat[selected_idx2]
        ], dim=0)

        self.feature_bank.append(output_ul_flatten_selected.detach().clone())
        self.pseudo_label_bank.append(pseudo_label_flatten_selected.detach().clone())

        if self.step_count > self.step_save:
            self.feature_bank = self.feature_bank[1:]
            self.pseudo_label_bank = self.pseudo_label_bank[1:]
        else:
            self.step_count += 1

        output_ul_all = torch.cat(self.feature_bank, dim=0)
        pseudo_label_all = torch.cat(self.pseudo_label_bank, dim=0)

        output_ul_all = output_ul_all.detach()
        logits1_down = self._contrastive_loss(
            pos=pos1,
            anchor=output_feat1,
            memory=output_ul_all,
            anchor_labels=pseudo_label1,
            memory_labels=pseudo_label_all
        )
        logits1 = torch.exp(pos1 - logits1_down['max']).squeeze(-1) / (logits1_down['sum'] + eps)
        pos_mask_1 = ((pseudo_logits2 > self.pos_thresh_value) & (pseudo_logits1 < pseudo_logits2)).float().detach()
        loss1 = -torch.log(logits1 + eps)
        loss1 = (loss1 * pos_mask_1).sum() / (pos_mask_1.sum() + 1e-12)

        logits2_down = self._contrastive_loss(
            pos=pos2,
            anchor=output_feat2,
            memory=output_ul_all,
            anchor_labels=pseudo_label2,
            memory_labels=pseudo_label_all
        )
        logits2 = torch.exp(pos2 - logits2_down['max']).squeeze(-1) / (logits2_down['sum'] + eps)
        pos_mask_2 = ((pseudo_logits1 > self.pos_thresh_value) & (pseudo_logits2 < pseudo_logits1)).float().detach()
        loss2 = -torch.log(logits2 + eps)
        loss2 = (loss2 * pos_mask_2).sum() / (pos_mask_2.sum() + 1e-12)

        return loss1 + loss2

    def _contrastive_loss(self, pos, anchor, memory, anchor_labels, memory_labels, chunk_size=1024):
        """
        Compute contrastive loss with chunking to prevent OOM. Avoids in-place ops.
        """
        device = anchor.device
        all_sum = []
        all_max = []

        for start in range(0, anchor.size(0), chunk_size):
            end = start + chunk_size
            anchor_chunk = anchor[start:end]  # [chunk_size, C]
            pos_chunk = pos[start:end]  # [chunk_size, 1]
            label_chunk = anchor_labels[start:end]  # [chunk_size]

            sim = torch.mm(anchor_chunk, memory.T) / self.temp  # [chunk_size, M]
            mask = (label_chunk.unsqueeze(1) != memory_labels.unsqueeze(0)).float()

            sim_all = torch.cat([pos_chunk, sim], dim=1)  # [chunk_size, M+1]
            mask_all = torch.cat([
                torch.ones(pos_chunk.size(0), 1, device=device),
                mask
            ], dim=1)

            sim_max = torch.max(sim_all, dim=1, keepdim=True)[0]
            sim_exp = torch.exp(sim_all - sim_max) * mask_all
            sim_sum = sim_exp.sum(dim=1)

            all_max.append(sim_max)
            all_sum.append(sim_sum)

        return {
            'max': torch.cat(all_max, dim=0),  # [N, 1]
            'sum': torch.cat(all_sum, dim=0)  # [N]
        }

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

    def forward(
            self, output_feat1, output_feat2, pseudo_label1, pseudo_label2,
            pseudo_logits1, pseudo_logits2, output_ul1, output_ul2, predicted_labels
    ):
        """
        Compute the Hybrid Contrastive Loss.
        Args:
            features: Tensor of shape (N, C, H, W), feature maps.
            labels: Tensor of shape (N, H, W), ground truth or pseudo-labels.
        Returns:
            torch.Tensor: Combined loss value.
        """
        loss_pixel = self.pixel_loss(features=output_feat1, labels=pseudo_label1, predict=predicted_labels)
        loss_local = self.local_loss(features=output_feat1, labels=pseudo_label1)
        loss_directional = self.directional_loss(
            output_feat1=output_feat1, output_feat2=output_feat2,
            pseudo_label1=pseudo_label1, pseudo_label2=pseudo_label2,
            pseudo_logits1=pseudo_logits1, pseudo_logits2=pseudo_logits2,
            output_ul1=output_ul1, output_ul2=output_ul2
        )

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
                    try:
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

                        try:
                            X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                        except Exception as e:
                            print(f"Error in _hard_anchor_sampling at X_[X_ptr, :, :] assignment: {e}")
                            print(f"X_[X_ptr, :, :].shape: {X_[X_ptr, :, :].shape}, X[ii, indices, :].shape: {X[ii, indices, :].shape}")
                            raise
                        y_[X_ptr] = cls_id
                        X_ptr += 1
                    except Exception as e:
                        print(f"Error in mining _hard_anchor_sampling: {e},"
                              f"this_y_hat shape: {this_y_hat.shape}, this_y shape: {this_y.shape}")
                        raise
                else:
                    try:
                        num_indice = (this_y_hat == cls_id).nonzero()
                        number = num_indice.shape[0]
                        perm = torch.randperm(number)
                        indices = num_indice[perm[:n_view]]

                        try:
                            X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                        except Exception as e:
                            print(f"Error in _hard_anchor_sampling at X_[X_ptr, :, :] assignment: {e}")
                            print(f"X_[X_ptr, :, :].shape: {X_[X_ptr, :, :].shape}, X[ii, indices, :].shape: {X[ii, indices, :].shape}")
                            raise
                        y_[X_ptr] = cls_id
                        X_ptr += 1
                    except Exception as e:
                        print(f"Error in mining _hard_anchor_sampling (line 814): {e},"
                              f"this_y_hat shape: {this_y_hat.shape}, this_y shape: {this_y.shape}, "
                              f"cls_id: {cls_id}, hard_indices shape: {hard_indices.shape}, "
                              f"easy_indices shape: {easy_indices.shape}")
                        raise
        return X_, y_

    def _hard_pair_sample_mining(self, X, X_2, y_hat, y, mask=None):
        try:
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

                        try:
                            X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                            X_2_[X_ptr, :, :] = X_2[ii, indices, :].squeeze(1)
                        except Exception as e:
                            print(f"Error in _hard_pair_sample_mining at X_ or X_2_ assignment: {e}")
                            print(f"X_[X_ptr, :, :].shape: {X_[X_ptr, :, :].shape}, X[ii, indices, :].shape: {X[ii, indices, :].shape}")
                            print(f"X_2_[X_ptr, :, :].shape: {X_2_[X_ptr, :, :].shape}, X_2[ii, indices, :].shape: {X_2[ii, indices, :].shape}")
                            raise
                        y_[X_ptr] = cls_id
                        X_ptr += 1
                    else:

                        num_indice = (this_y_hat == cls_id).nonzero()
                        number = num_indice.shape[0]
                        perm = torch.randperm(number)
                        indices = num_indice[perm[:n_view]]

                        try:
                            X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                            X_2_[X_ptr, :, :] = X_2[ii, indices, :].squeeze(1)
                        except Exception as e:
                            print(f"Error in _hard_pair_sample_mining at X_ or X_2_ assignment: {e}")
                            print(f"X_[X_ptr, :, :].shape: {X_[X_ptr, :, :].shape}, X[ii, indices, :].shape: {X[ii, indices, :].shape}")
                            print(f"X_2_[X_ptr, :, :].shape: {X_2_[X_ptr, :, :].shape}, X_2[ii, indices, :].shape: {X_2[ii, indices, :].shape}")
                            raise
                        y_[X_ptr] = cls_id
                        X_ptr += 1

            return X_, X_2_, y_
        except Exception as e:
            print(f"Error in _hard_pair_sample_mining: {e}")
            raise

    def _contrastive(self, feats_, labels_):
        try:
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
        except Exception as e:
            print(f"Error in _contrastive: {e}")
            print(f"mask.shape: {mask.shape}, anchor_count: {anchor_count}, contrast_count: {contrast_count}")
            raise

    def _contrastive_pair(self, feats_, feats_t, labels_, labels_t):
        try:
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
        except Exception as e:
            print(f"Error in _contrastive_pair: {e}")
            print(f"mask.shape: {mask.shape}, anchor_count: {anchor_count}, contrast_count: {contrast_count}")
            raise

    def forward(self, features, feats_t=None, labels=None, predict=None, cb_mask=None):
        # feat from student, feats_2 from teacher if have

        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,(features.shape[2], features.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == features.shape[-1], '{} {}'.format(labels.shape, features.shape)
        predict = predict.unsqueeze(1).float()
        predict = F.interpolate(predict, (features.shape[2], features.shape[3]), mode='nearest')
        predict = predict.squeeze(1).long()

        if cb_mask is not None:
            mask_batch = cb_mask.shape[0]
            batch, _, h, w = features.shape
            cb_mask = F.interpolate(cb_mask.float(), (h, w), mode='nearest')
            if mask_batch != batch: # when use both labeled and unlabeld data for loss, labeled do not need any ignore
                labeled_confidence_mask = torch.ones_like(cb_mask).to(cb_mask.dtype)
                cb_mask = torch.cat([labeled_confidence_mask, cb_mask])
            cb_mask = cb_mask.view(batch, -1)
        batch_size = features.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = features.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])
        feats = F.normalize(feats, dim=2)
        if feats_t is None:
            feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)
            if feats_ is None:
                print('input do not have enough label, ignore (happened in beginning of BDD)')
                loss = feats.sum() * 0.0
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
                loss = feats.sum() * 0.0
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