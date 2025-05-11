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
        # Ensure outputs and labels are properly connected to the computation graph
        assert output.requires_grad, "Outputs tensor is detached!"
        assert target.requires_grad is False, "Labels tensor should not require gradients!"

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

        assert loss.requires_grad, "Final loss tensor is detached!"
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
    def __init__(self, temperature=0.1, no_of_pos_eles=3, no_of_neg_eles=3, local_loss_exp_no=1, lamda_local=0.1):
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
        labels = labels.view(B, H, W)

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


    #         ###################################
    #         # Contrastive loss specific layers
    #         ###################################
    #         # Output of common encoder-decoder network - to be feed to a contrastive specific layers
    #         tmp_dec_layer=dec_c1_a
    #         tmp_no_filters=no_filters[1]
    #
    #         # h_\phi - small network with two 1x1 convolutions for contrastive loss computation
    #         tmp_dec_layer = tf.cond(train_phase, lambda: tf.gather(tmp_dec_layer, unl_indices), lambda: tmp_dec_layer)
    #         #print('tmp_dec_layer', tmp_dec_layer)
    #         cont_c1_a = layers.conv2d_layer(ip_layer=tmp_dec_layer,name='cont_c1_a', kernel_size=(1,1), num_filters=tmp_no_filters,use_bias=False, use_relu=True, use_batch_norm=True, training_phase=train_phase)
    #         cont_c1_b = layers.conv2d_layer(ip_layer=cont_c1_a, name='cont_c1_b', kernel_size=(1,1),num_filters=tmp_no_filters, use_bias=False, use_relu=False,use_batch_norm=False, training_phase=train_phase)
    #
    #         y_fin_tmp=cont_c1_b
    #
    #         # Define Local Contrastive loss
    #         if(inf==1):
    #             y_fin=y_fin_tmp
    #             local_loss=1
    #             net_local_loss=1#np.array(1, dtype=np.int32)
    #             bs,tmp_batch_size=20,10
    #             #bs=2*self.batch_size
    #         else:
    #             y_fin=y_fin_tmp
    #             print('y_fin_local',y_fin,pos_indx,neg_indx)
    #             print('start of for loop',time.ctime())
    #
    #             local_loss=0
    #             net_local_loss=0
    #
    #             for pos_index in range(0,batch_size_ft,1):
    #
    #                 index_pos1=pos_index
    #                 index_pos2=batch_size_ft+pos_index
    #
    #                 #indexes of positive pair of samples (f(x),f(x')) of input images (x,x') from the batch of feature maps.
    #                 num_i1=np.arange(index_pos1,index_pos1+1,dtype=np.int32)
    #                 num_i2=np.arange(index_pos2,index_pos2+1,dtype=np.int32)
    #
    #                 # gather required positive samples (f(x),f(x')) of (x,x') for the numerator term
    #                 x_num_i1=tf.gather(y_fin,num_i1)
    #                 x_num_i2=tf.gather(y_fin,num_i2)
    #                 #print('x_num_i1',index_pos1,index_pos2,x_num_i1,x_num_i2)
    #
    #                 x_zero_re=tf.constant(0,shape=(),dtype=np.float32)
    #                 x_zero_vec=tf.constant(0,dtype=np.float32,shape=(1,tmp_no_filters))
    #
    #                 #x_epsilon_vec=tf.constant(0.0,dtype=np.float32,shape=(1,tmp_no_filters))
    #                 #print('x_zero_vec',x_zero_vec)
    #
    #                 mask_i1 = tf.gather(y_l_reg, num_i1)
    #                 mask_i2 = tf.gather(y_l_reg, num_i2)
    #
    #                 if(self.dataset_name=='mmwhs'):
    #                     #pos_cls_ref = np.asarray([0,2,3,4,5,6])
    #                     #neg_cls_ref = np.asarray([1,3,4,5,6,7])
    #                     #pos_cls_ref = np.asarray([0,1,2,3,5,6])
    #                     #neg_cls_ref = np.asarray([1,2,3,4,6,7])
    #                     pos_cls_ref = np.asarray([0,1,2,3,4,5,6])
    #                     neg_cls_ref = np.asarray([1,2,3,4,5,6,7])
    #                     print('pos_cls_ref,neg_cls_ref',pos_cls_ref,neg_cls_ref)
    #                 elif(self.dataset_name=='acdc'):
    #                     pos_cls_ref = np.asarray([0,1,2])
    #                     neg_cls_ref = np.asarray([1,2,3])
    #                     print('pos_cls_ref,neg_cls_ref', pos_cls_ref, neg_cls_ref)
    #                 elif(self.dataset_name=='prostate_md'):
    #                     pos_cls_ref = np.asarray([0,1])
    #                     neg_cls_ref = np.asarray([1,2])
    #                     print('pos_cls_ref,neg_cls_ref', pos_cls_ref, neg_cls_ref)
    #
    #                 for pos_cls in pos_cls_ref:
    #                     pos_cls_ele_i1=pos_indx[pos_index][pos_cls]
    #                     neg_cls_ele_i1=neg_indx[pos_index][pos_cls]
    #                     pos_cls_ele_i2=pos_indx[batch_size_ft+pos_index][pos_cls]
    #                     neg_cls_ele_i2=neg_indx[batch_size_ft+pos_index][pos_cls]
    #
    #                     #print('cls_i1',pos_cls,pos_cls_ele_i1,neg_cls_ele_i1)
    #                     #print('cls_i2',pos_cls,pos_cls_ele_i2,neg_cls_ele_i2)
    #
    #                     #############################
    #                     #mask of image 1 (x) from batch X_B
    #                     #select positive classes masks' mean embeddings
    #                     mask_i1_pos=tf.gather(mask_i1,pos_cls+1,axis=-1)
    #                     ##mask_i1_pos=tf.gather(mask_i1,pos_cls,axis=-1)
    #                     pos_cls_avg_i1 = tf.boolean_mask(x_num_i1, mask_i1_pos)
    #                     pos_avg_vec_i1_p = tf.reshape(tf.reduce_mean(pos_cls_avg_i1,axis=0),(1,tmp_no_filters))
    #                     pos_avg_i1_nan=tf.is_nan(tf.reduce_sum(pos_avg_vec_i1_p))
    #                     ##print('pos_avg_vec_i1',pos_cls,mask_i1_pos,pos_cls_avg_i1,pos_avg_vec_i1_p)
    #                     #############################
    #
    #                     #############################
    #                     # make list of negative classes masks' mean embeddings from image 1 (x) mask
    #                     neg_mask1_list = []
    #                     for neg_cls_i1 in neg_cls_ref:
    #                         mask_i1_neg = tf.gather(mask_i1, neg_cls_i1, axis=-1)
    #                         neg_cls_avg_i1 = tf.boolean_mask(x_num_i1, mask_i1_neg)
    #                         neg_avg_vec_i1_p = tf.reshape(tf.reduce_mean(neg_cls_avg_i1, axis=0), (1, tmp_no_filters))
    #                         neg_avg_i1_nan=tf.is_nan(tf.reduce_sum(neg_avg_vec_i1_p))
    #                         neg_mask1_list.append(neg_avg_vec_i1_p)
    #                     #print('neg_mask1_list', neg_mask1_list)
    #                     #############################
    #
    #                     #############################
    #                     #mask of image 2 (x')  from batch X_B
    #                     #select positive classes masks' mean embeddings
    #                     mask_i2_pos=tf.gather(mask_i2,pos_cls+1,axis=-1)
    #                     ##mask_i2_pos=tf.gather(mask_i2,pos_cls,axis=-1)
    #                     pos_cls_avg_i2 = tf.boolean_mask(x_num_i2, mask_i2_pos)
    #                     pos_avg_vec_i2_p = tf.reshape(tf.reduce_mean(pos_cls_avg_i2,axis=0),(1,tmp_no_filters))
    #                     pos_avg_i2_nan=tf.is_nan(tf.reduce_sum(pos_avg_vec_i2_p))
    #                     #print('pos_avg_vec_i2',pos_cls,mask_i2_pos,pos_cls_avg_i2,pos_avg_vec_i2_p)
    #                     #############################
    #
    #                     #############################
    #                     # #select negative classes mask averages
    #                     # make list of negative classes masks' mean embeddings from image 2 (x') mask
    #                     neg_mask2_list = []
    #                     for neg_cls_i2 in neg_cls_ref:
    #                         mask_i2_neg = tf.gather(mask_i2, neg_cls_i2, axis=-1)
    #                         neg_cls_avg_i2 = tf.boolean_mask(x_num_i2, mask_i2_neg)
    #                         neg_avg_vec_i2_p = tf.reshape(tf.reduce_mean(neg_cls_avg_i2, axis=0), (1, tmp_no_filters))
    #                         neg_avg_i2_nan=tf.is_nan(tf.reduce_sum(neg_avg_vec_i2_p))
    #                         neg_mask2_list.append(neg_avg_vec_i2_p)
    #                     #print('neg_mask2_list', neg_mask2_list)
    #                     #############################
    #
    #                     #Loop over all the positive embeddings from f(x) of all classes
    #                     for n_pos_idx in range(0,no_of_pos_eles,1):
    #                         x_num_tmp_i1 = tf.gather(x_num_i1,pos_cls_ele_i1[n_pos_idx][0],axis=1)
    #                         #print('x_num_tmp_i1 j0',x_num_tmp_i1)
    #                         x_num_tmp_i1 = tf.gather(x_num_tmp_i1,pos_cls_ele_i1[n_pos_idx][1],axis=1)
    #                         #print('x_num_tmp_i1 j1',x_num_tmp_i1)
    #
    #                         x_n1_count=tf.math.count_nonzero(x_num_tmp_i1)
    #                         x_n_i1_flat = tf.cond(tf.equal(x_n1_count,0), lambda: x_zero_vec, lambda: tf.layers.flatten(inputs=x_num_tmp_i1))
    #                         #print('x_n_i1_flat',x_n_i1_flat,tf.layers.flatten(inputs=x_num_tmp_i1))
    #                         x_w3_n_i1=x_n_i1_flat
    #
    #                         x_num_tmp_i2 = tf.gather(x_num_i2,pos_cls_ele_i2[n_pos_idx][0],axis=1)
    #                         x_num_tmp_i2 = tf.gather(x_num_tmp_i2,pos_cls_ele_i2[n_pos_idx][1],axis=1)
    #                         #print('x_num_tmp_i2 j',x_num_tmp_i2)
    #
    #                         x_n2_count=tf.math.count_nonzero(x_num_tmp_i2)
    #                         x_n_i2_flat = tf.cond(tf.equal(x_n2_count,0), lambda: x_zero_vec, lambda: tf.layers.flatten(inputs=x_num_tmp_i2))
    #                         #print('x_n_i2_flat',x_n_i2_flat,tf.layers.flatten(inputs=x_num_tmp_i2))
    #                         x_w3_n_i2=x_n_i2_flat
    #
    #                         # Cosine loss for positive pair of pixel embeddings from f(x), f(x')
    #                         # Numerator loss terms of local loss
    #                         pos_avg_vec_i1=pos_avg_vec_i1_p
    #                         pos_avg_vec_i2=pos_avg_vec_i2_p
    #
    #                         log_or_n1 = tf.math.logical_or(tf.equal(x_n1_count,0),tf.equal(tf.math.count_nonzero(pos_avg_vec_i1),0))
    #                         log_or_n1_nan = tf.math.logical_or(tf.is_nan(tf.reduce_sum(x_w3_n_i1)),tf.is_nan(tf.reduce_sum(pos_avg_vec_i1)))
    #                         log_or_n1_net = tf.math.logical_or(log_or_n1,log_or_n1_nan)
    #                         num_i1_ss = tf.cond(log_or_n1_net, lambda: x_zero_re, lambda: self.cos_sim(x_w3_n_i1,pos_avg_vec_i1,temp_fac))
    #                         log_or_n2 = tf.math.logical_or(tf.equal(x_n2_count,0),tf.equal(tf.math.count_nonzero(pos_avg_vec_i2),0))
    #                         log_or_n2_nan = tf.math.logical_or(tf.is_nan(tf.reduce_sum(x_w3_n_i2)),tf.is_nan(tf.reduce_sum(pos_avg_vec_i2)))
    #                         log_or_n2_net = tf.math.logical_or(log_or_n2, log_or_n2_nan)
    #                         num_i2_ss = tf.cond(log_or_n2_net, lambda: x_zero_re, lambda: self.cos_sim(x_w3_n_i2,pos_avg_vec_i2,temp_fac))
    #
    #                         if(local_loss_exp_no==1):
    #                             log_or_i1_i2 = tf.math.logical_or(tf.equal(x_n1_count,0),tf.equal(tf.math.count_nonzero(pos_avg_vec_i2),0))
    #                             log_or_i1_i2_nan = tf.math.logical_or(tf.is_nan(tf.reduce_sum(x_w3_n_i1)),tf.is_nan(tf.reduce_sum(pos_avg_vec_i2)))
    #                             log_or_i1_i2_net = tf.math.logical_or(log_or_i1_i2, log_or_i1_i2_nan)
    #                             num_i1_i2_ss = tf.cond(log_or_i1_i2_net, lambda: x_zero_re, lambda: self.cos_sim(x_w3_n_i1,pos_avg_vec_i2,temp_fac))
    #
    #                             log_or_i2_i1 = tf.math.logical_or(tf.equal(x_n2_count,0),tf.equal(tf.math.count_nonzero(pos_avg_vec_i1),0))
    #                             log_or_i2_i1_nan = tf.math.logical_or(tf.is_nan(tf.reduce_sum(x_w3_n_i2)),tf.is_nan(tf.reduce_sum(pos_avg_vec_i1)))
    #                             log_or_i2_i1_net = tf.math.logical_or(log_or_i2_i1,log_or_i2_i1_nan)
    #                             num_i2_i1_ss = tf.cond(log_or_i2_i1_net, lambda: x_zero_re, lambda: self.cos_sim(x_w3_n_i2,pos_avg_vec_i1,temp_fac))
    #
    #                         # Denominator loss terms of local loss
    #                         den_i1_ss,den_i2_ss=0,0
    #                         den_i1_i2_ss,den_i2_i1_ss=0,0
    #
    #                         #############################
    #                         # compute loss for positive mean class pixels from mask of image 1 (x)
    #                         # negatives from mask of image 1 (x)
    #                         for neg_avg_i1_c1 in neg_mask1_list:
    #                             log_or_n1_d1 = tf.math.logical_or(tf.equal(x_n1_count, 0),tf.equal(tf.math.count_nonzero(neg_avg_i1_c1), 0))
    #                             log_or_n1_d1_nan = tf.math.logical_or(tf.is_nan(tf.reduce_sum(x_w3_n_i1)),tf.is_nan(tf.reduce_sum(neg_avg_i1_c1)))
    #                             log_or_n1_d1_net = tf.math.logical_or(log_or_n1_d1,log_or_n1_d1_nan)
    #                             den_i1_ss = den_i1_ss + tf.cond(log_or_n1_d1_net, lambda: x_zero_re,lambda: tf.exp(self.cos_sim(x_w3_n_i1, neg_avg_i1_c1, temp_fac)))
    #
    #                         # negatives from mask of image 2 (x')
    #                         if (local_loss_exp_no == 1):
    #                             for neg_avg_i1_c2 in neg_mask2_list:
    #                                 log_or_n1_d2 = tf.math.logical_or(tf.equal(x_n1_count, 0),tf.equal(tf.math.count_nonzero(neg_avg_i1_c2), 0))
    #                                 log_or_n1_d2_nan = tf.math.logical_or(tf.is_nan(tf.reduce_sum(x_w3_n_i1)),tf.is_nan(tf.reduce_sum(neg_avg_i1_c2)))
    #                                 log_or_n1_d2_net = tf.math.logical_or(log_or_n1_d2,log_or_n1_d2_nan)
    #                                 den_i1_ss = den_i1_ss + tf.cond(log_or_n1_d2_net, lambda: x_zero_re,lambda: tf.exp(self.cos_sim(x_w3_n_i1, neg_avg_i1_c2, temp_fac)))
    #                         #############################
    #
    #                         #############################
    #                         # compute loss for positive avg class pixels from mask of image 2 (x')
    #                         # negatives from mask of image 2 (x')
    #                         for neg_avg_i2_c2 in neg_mask2_list:
    #                             log_or_n2_d2 = tf.math.logical_or(tf.equal(x_n2_count, 0),tf.equal(tf.math.count_nonzero(neg_avg_i2_c2),0))
    #                             log_or_n2_d2_nan = tf.math.logical_or(tf.is_nan(tf.reduce_sum(x_w3_n_i2)),tf.is_nan(tf.reduce_sum(neg_avg_i2_c2)))
    #                             log_or_n2_d2_net = tf.math.logical_or(log_or_n2_d2,log_or_n2_d2_nan)
    #                             den_i2_ss = den_i2_ss + tf.cond(log_or_n2_d2_net, lambda: x_zero_re,lambda: tf.exp(self.cos_sim(x_w3_n_i2, neg_avg_i2_c2,temp_fac)))
    #
    #                         # negatives from mask of image 1 (x)
    #                         if (local_loss_exp_no == 1):
    #                             for neg_avg_i2_c1 in neg_mask1_list:
    #                                 log_or_n2_d1 = tf.math.logical_or(tf.equal(x_n2_count, 0), tf.equal(tf.math.count_nonzero(neg_avg_i2_c1), 0))
    #                                 log_or_n2_d1_nan = tf.math.logical_or(tf.is_nan(tf.reduce_sum(x_w3_n_i2)),tf.is_nan(tf.reduce_sum(neg_avg_i2_c1)))
    #                                 log_or_n2_d1_net = tf.math.logical_or(log_or_n2_d1,log_or_n2_d1_nan)
    #                                 den_i2_ss = den_i2_ss + tf.cond(log_or_n2_d1_net, lambda: x_zero_re,lambda: tf.exp(self.cos_sim(x_w3_n_i2, neg_avg_i2_c1,temp_fac)))
    #
    #                         log_num_i1_nan = tf.math.logical_or(tf.is_nan(tf.exp(num_i1_ss)),tf.is_nan(tf.exp(den_i1_ss)))
    #                         log_num_i1_nan = tf.squeeze(log_num_i1_nan)
    #                         log_num_i1_zero = tf.math.logical_or(tf.equal(tf.math.count_nonzero(num_i1_ss), 0),tf.equal(tf.math.count_nonzero(den_i1_ss), 0))
    #                         log_num_i1_net = tf.math.logical_or(log_num_i1_zero, log_num_i1_nan)
    #                         num_i1_loss = tf.cond(log_num_i1_net, lambda: x_zero_re, lambda: -tf.log(tf.math.reduce_sum(tf.exp(num_i1_ss))/(tf.math.reduce_sum(den_i1_ss))))
    #                         local_loss = local_loss + num_i1_loss
    #
    #                         log_num_i2_nan = tf.math.logical_or(tf.is_nan(tf.exp(num_i2_ss)),tf.is_nan(tf.exp(den_i2_ss)))
    #                         log_num_i2_nan = tf.squeeze(log_num_i2_nan)
    #                         log_num_i2_zero = tf.math.logical_or(tf.equal(tf.math.count_nonzero(num_i2_ss), 0),tf.equal(tf.math.count_nonzero(den_i2_ss), 0))
    #                         log_num_i2_net = tf.math.logical_or(log_num_i2_zero, log_num_i2_nan)
    #                         num_i2_loss = tf.cond(log_num_i2_net, lambda: x_zero_re, lambda: -tf.log(tf.math.reduce_sum(tf.exp(num_i2_ss))/(tf.math.reduce_sum(den_i2_ss))))
    #                         local_loss = local_loss + num_i2_loss
    #                         if (local_loss_exp_no == 1):
    #                             #local loss from feature map f(x) of image 1 (x)
    #                             #log_num_i1_i2 = tf.math.logical_or(tf.equal(tf.math.reduce_sum(tf.exp(num_i1_i2_ss)),0),tf.equal((tf.math.reduce_sum(tf.exp(num_i1_i2_ss))+tf.math.reduce_sum(den_i1_ss)),0))
    #                             log_num_i1_i2_nan = tf.math.logical_or(tf.is_nan(tf.exp(num_i1_i2_ss)),tf.is_nan(tf.exp(den_i1_ss)))
    #                             log_num_i1_i2_nan = tf.squeeze(log_num_i1_i2_nan)
    #                             log_num_i1_i2_zero = tf.math.logical_or(tf.equal(tf.math.count_nonzero(num_i1_i2_ss), 0),tf.equal(tf.math.count_nonzero(den_i1_ss), 0))
    #                             log_num_i1_i2_net = tf.math.logical_or(log_num_i1_i2_zero, log_num_i1_i2_nan)
    #                             #local_loss = local_loss + tf.cond(log_num_i1_i2_net, lambda: x_zero_re, lambda: -tf.log(tf.math.reduce_sum(tf.exp(num_i1_i2_ss))/(tf.math.reduce_sum(tf.exp(num_i1_i2_ss))+tf.math.reduce_sum(den_i1_ss))))
    #                             local_loss = local_loss + tf.cond(log_num_i1_i2_net, lambda: x_zero_re, lambda: -tf.log(tf.math.reduce_sum(tf.exp(num_i1_i2_ss))/(tf.math.reduce_sum(den_i1_ss))))
    #
    #                             # local loss from feature map f(x') of image 2 (x')
    #                             #log_num_i2_i1 = tf.math.logical_or(tf.equal(tf.math.reduce_sum(tf.exp(num_i2_i1_ss)),0),tf.equal((tf.math.reduce_sum(tf.exp(num_i2_i1_ss))+tf.math.reduce_sum(den_i2_ss)),0))
    #                             log_num_i2_i1_nan = tf.math.logical_or(tf.is_nan(tf.exp(num_i2_i1_ss)),tf.is_nan(tf.exp(den_i2_ss)))
    #                             log_num_i2_i1_nan = tf.squeeze(log_num_i2_i1_nan)
    #                             log_num_i2_i1_zero = tf.math.logical_or(tf.equal(tf.math.count_nonzero(num_i2_i1_ss), 0),tf.equal(tf.math.count_nonzero(den_i2_ss), 0))
    #                             log_num_i2_i1_net = tf.math.logical_or(log_num_i2_i1_zero,log_num_i2_i1_nan)
    #                             #local_loss = local_loss + tf.cond(log_num_i2_i1_net, lambda: x_zero_re, lambda: -tf.log(tf.math.reduce_sum(tf.exp(num_i2_i1_ss))/(tf.math.reduce_sum(tf.exp(num_i2_i1_ss))+tf.math.reduce_sum(den_i2_ss))))
    #                             local_loss = local_loss + tf.cond(log_num_i2_i1_net, lambda: x_zero_re, lambda: -tf.log(tf.math.reduce_sum(tf.exp(num_i2_i1_ss))/(tf.math.reduce_sum(den_i2_ss))))
    #
    #                 if (local_loss_exp_no == 1):
    #                     local_loss=local_loss/(2*no_of_pos_eles*(num_classes-1))
    #                 else:
    #                     local_loss=local_loss/(no_of_pos_eles*(num_classes-1))
    #
    #             net_local_loss=local_loss/batch_size_ft
    # cont_loss_cost=lamda_local * tf.reduce_mean(net_local_loss)
    #                 seg_cost=tf.reduce_mean(seg_cost)
    #                 net_cost= seg_cost + cont_loss_cost


class DirectionalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, pos_thresh=0.7, selected_num=8000):
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
        pos1 = (output_feat1 * output_feat2.detach()).sum(-1, keepdim=True) / self.temp
        pos2 = (output_feat1.detach() * output_feat2).sum(-1, keepdim=True) / self.temp

        # Flatten and sample memory bank candidates``
        b, c, h, w = output_ul1.size()
        output_ul1_flat = output_ul1.permute(0, 2, 3, 1).contiguous().view(b * h * w, c)
        output_ul2_flat = output_ul2.permute(0, 2, 3, 1).contiguous().view(b * h * w, c)

        selected_idx1 = torch.randperm(output_ul1_flat.size(0))[:self.selected_num]
        selected_idx2 = torch.randperm(output_ul2_flat.size(0))[:self.selected_num]

        output_ul_flatten_selected = torch.cat([
            output_ul1_flat[selected_idx1],
            output_ul2_flat[selected_idx2]
        ], dim=0)

        pseudo_label1_flat = pseudo_label1.view(-1)
        pseudo_label2_flat = pseudo_label2.view(-1)
        pseudo_label_flatten_selected = torch.cat([
            pseudo_label1_flat[selected_idx1],
            pseudo_label2_flat[selected_idx2]
        ], dim=0)

        self.feature_bank.append(output_ul_flatten_selected)
        self.pseudo_label_bank.append(pseudo_label_flatten_selected)

        if self.step_count > self.step_save:
            self.feature_bank = self.feature_bank[1:]
            self.pseudo_label_bank = self.pseudo_label_bank[1:]
        else:
            self.step_count += 1

        output_ul_all = torch.cat(self.feature_bank, dim=0)
        pseudo_label_all = torch.cat(self.pseudo_label_bank, dim=0)

        # --- LOSS 1 ---
        logits1_down = self._contrastive_loss(
            pos=pos1,
            anchor=output_feat1,
            memory=output_ul_all,
            anchor_labels=pseudo_label1,
            memory_labels=pseudo_label_all
        )
        logits1 = torch.exp(pos1 - logits1_down['max']).squeeze(-1) / (logits1_down['sum'] + eps)
        pos_mask_1 = ((pseudo_logits2 > self.pos_thresh_value) & (pseudo_logits1 < pseudo_logits2)).float()
        loss1 = -torch.log(logits1 + eps)
        loss1 = (loss1 * pos_mask_1).sum() / (pos_mask_1.sum() + 1e-12)

        # --- LOSS 2 ---
        logits2_down = self._contrastive_loss(
            pos=pos2,
            anchor=output_feat2,
            memory=output_ul_all,
            anchor_labels=pseudo_label2,
            memory_labels=pseudo_label_all
        )
        logits2 = torch.exp(pos2 - logits2_down['max']).squeeze(-1) / (logits2_down['sum'] + eps)
        pos_mask_2 = ((pseudo_logits1 > self.pos_thresh_value) & (pseudo_logits2 < pseudo_logits1)).float()
        loss2 = -torch.log(logits2 + eps)
        loss2 = (loss2 * pos_mask_2).sum() / (pos_mask_2.sum() + 1e-12)

        return loss1 + loss2

    def _contrastive_loss(self, pos, anchor, memory, anchor_labels, memory_labels):
        eps = 1e-8
        mask = (anchor_labels.unsqueeze(0) != memory_labels.unsqueeze(-1)).float()
        neg = (anchor @ memory.T) / self.temp
        neg = torch.cat([pos, neg], dim=1)
        mask = torch.cat([
            torch.ones(mask.size(0), 1).float().to(anchor.device),
            mask
        ], dim=1)
        neg_max = torch.max(neg, dim=1, keepdim=True)[0]
        weighted_neg = torch.exp(neg - neg_max) * mask
        neg_sum = weighted_neg.sum(dim=1)
        return {'sum': neg_sum, 'max': neg_max}

    # if self.mode == 'supervised':
    #                 total_loss, cur_losses, outputs = self.model(x_l=input_l, target_l=target_l, x_ul=input_ul,
    #                                                             curr_iter=batch_idx, target_ul=target_ul, epoch=epoch-1)
    #             else:
    #                 kargs = {'gpu': self.gpu, 'ul1': ul1, 'br1': br1, 'ul2': ul2, 'br2': br2, 'flip': flip}
    #                 total_loss, cur_losses, outputs = self.model(x_l=input_l, target_l=target_l, x_ul=input_ul,
    #                                                             curr_iter=batch_idx, target_ul=target_ul, epoch=epoch-1, **kargs)
    #                 target_ul = target_ul[:, 0]

    # in model.py
    # elif self.mode == 'semi':
    #
    #             enc = self.encoder(x_l)
    #             enc = self.classifier(enc)
    #             output_l = F.interpolate(enc, size=x_l.size()[2:], mode='bilinear', align_corners=True)
    #             loss_sup = self.sup_loss(output_l, target_l, ignore_index=self.ignore_index, temperature=1.0) * self.sup_loss_w
    #
    #             curr_losses = {'loss_sup': loss_sup}
    #             outputs = {'sup_pred': output_l}
    #             total_loss = loss_sup
    #
    #             if epoch < self.epoch_start_unsup:
    #                 return total_loss, curr_losses, outputs
    #
    #             # x_ul: [batch_size, 2, 3, H, W]
    #             x_ul1 = x_ul[:, 0, :, :, :]
    #             x_ul2 = x_ul[:, 1, :, :, :]
    #
    #             enc_ul1 = self.encoder(x_ul1)
    #             if self.downsample:
    #                 enc_ul1 = F.avg_pool2d(enc_ul1, kernel_size=2, stride=2)
    #             output_ul1 = self.project(enc_ul1) #[b, c, h, w]
    #             output_ul1 = F.normalize(output_ul1, 2, 1)
    #
    #             enc_ul2 = self.encoder(x_ul2)
    #             if self.downsample:
    #                 enc_ul2 = F.avg_pool2d(enc_ul2, kernel_size=2, stride=2)
    #             output_ul2 = self.project(enc_ul2) #[b, c, h, w]
    #             output_ul2 = F.normalize(output_ul2, 2, 1)
    #
    #             # compute pseudo label
    #             logits1 = self.classifier(enc_ul1) #[batch_size, num_classes, h, w]
    #             logits2 = self.classifier(enc_ul2)
    #             pseudo_logits_1 = F.softmax(logits1, 1).max(1)[0].detach() #[batch_size, h, w]
    #             pseudo_logits_2 = F.softmax(logits2, 1).max(1)[0].detach()
    #             pseudo_label1 = logits1.max(1)[1].detach() #[batch_size, h, w]
    #             pseudo_label2 = logits2.max(1)[1].detach()
    #
    #             # get overlap part
    #             output_feature_list1 = []
    #             output_feature_list2 = []
    #             pseudo_label_list1 = []
    #             pseudo_label_list2 = []
    #             pseudo_logits_list1 = []
    #             pseudo_logits_list2 = []
    #             for idx in range(x_ul1.size(0)):
    #                 output_ul1_idx = output_ul1[idx]
    #                 output_ul2_idx = output_ul2[idx]
    #                 pseudo_label1_idx = pseudo_label1[idx]
    #                 pseudo_label2_idx = pseudo_label2[idx]
    #                 pseudo_logits_1_idx = pseudo_logits_1[idx]
    #                 pseudo_logits_2_idx = pseudo_logits_2[idx]
    #                 if flip[0][idx] == True:
    #                     output_ul1_idx = torch.flip(output_ul1_idx, dims=(2,))
    #                     pseudo_label1_idx = torch.flip(pseudo_label1_idx, dims=(1,))
    #                     pseudo_logits_1_idx = torch.flip(pseudo_logits_1_idx, dims=(1,))
    #                 if flip[1][idx] == True:
    #                     output_ul2_idx = torch.flip(output_ul2_idx, dims=(2,))
    #                     pseudo_label2_idx = torch.flip(pseudo_label2_idx, dims=(1,))
    #                     pseudo_logits_2_idx = torch.flip(pseudo_logits_2_idx, dims=(1,))
    #                 output_feature_list1.append(output_ul1_idx[:, ul1[0][idx]//8:br1[0][idx]//8, ul1[1][idx]//8:br1[1][idx]//8].permute(1, 2, 0).contiguous().view(-1, output_ul1.size(1)))
    #                 output_feature_list2.append(output_ul2_idx[:, ul2[0][idx]//8:br2[0][idx]//8, ul2[1][idx]//8:br2[1][idx]//8].permute(1, 2, 0).contiguous().view(-1, output_ul2.size(1)))
    #                 pseudo_label_list1.append(pseudo_label1_idx[ul1[0][idx]//8:br1[0][idx]//8, ul1[1][idx]//8:br1[1][idx]//8].contiguous().view(-1))
    #                 pseudo_label_list2.append(pseudo_label2_idx[ul2[0][idx]//8:br2[0][idx]//8, ul2[1][idx]//8:br2[1][idx]//8].contiguous().view(-1))
    #                 pseudo_logits_list1.append(pseudo_logits_1_idx[ul1[0][idx]//8:br1[0][idx]//8, ul1[1][idx]//8:br1[1][idx]//8].contiguous().view(-1))
    #                 pseudo_logits_list2.append(pseudo_logits_2_idx[ul2[0][idx]//8:br2[0][idx]//8, ul2[1][idx]//8:br2[1][idx]//8].contiguous().view(-1))
    #             output_feat1 = torch.cat(output_feature_list1, 0) #[n, c]
    #             output_feat2 = torch.cat(output_feature_list2, 0) #[n, c]
    #             pseudo_label1_overlap = torch.cat(pseudo_label_list1, 0) #[n,]
    #             pseudo_label2_overlap = torch.cat(pseudo_label_list2, 0) #[n,]
    #             pseudo_logits1_overlap = torch.cat(pseudo_logits_list1, 0) #[n,]
    #             pseudo_logits2_overlap = torch.cat(pseudo_logits_list2, 0) #[n,]
    #             assert output_feat1.size(0) == output_feat2.size(0)
    #             assert pseudo_label1_overlap.size(0) == pseudo_label2_overlap.size(0)
    #             assert output_feat1.size(0) == pseudo_label1_overlap.size(0)
    #
    #             # concat across multi-gpus
    #             b, c, h, w = output_ul1.size()
    #             selected_num = self.selected_num
    #             output_ul1_flatten = output_ul1.permute(0, 2, 3, 1).contiguous().view(b*h*w, c)
    #             output_ul2_flatten = output_ul2.permute(0, 2, 3, 1).contiguous().view(b*h*w, c)
    #             selected_idx1 = np.random.choice(range(b*h*w), selected_num, replace=False)
    #             selected_idx2 = np.random.choice(range(b*h*w), selected_num, replace=False)
    #             output_ul1_flatten_selected = output_ul1_flatten[selected_idx1]
    #             output_ul2_flatten_selected = output_ul2_flatten[selected_idx2]
    #             output_ul_flatten_selected = torch.cat([output_ul1_flatten_selected, output_ul2_flatten_selected], 0) #[2*kk, c]
    #             output_ul_all = self.concat_all_gather(output_ul_flatten_selected) #[2*N, c]
    #
    #             pseudo_label1_flatten_selected = pseudo_label1.view(-1)[selected_idx1]
    #             pseudo_label2_flatten_selected = pseudo_label2.view(-1)[selected_idx2]
    #             pseudo_label_flatten_selected = torch.cat([pseudo_label1_flatten_selected, pseudo_label2_flatten_selected], 0) #[2*kk]
    #             pseudo_label_all = self.concat_all_gather(pseudo_label_flatten_selected) #[2*N]
    #
    #             self.feature_bank.append(output_ul_all)
    #             self.pseudo_label_bank.append(pseudo_label_all)
    #             if self.step_count > self.step_save:
    #                 self.feature_bank = self.feature_bank[1:]
    #                 self.pseudo_label_bank = self.pseudo_label_bank[1:]
    #             else:
    #                 self.step_count += 1
    #             output_ul_all = torch.cat(self.feature_bank, 0)
    #             pseudo_label_all = torch.cat(self.pseudo_label_bank, 0)
    #
    #             eps = 1e-8
    #             pos1 = (output_feat1 * output_feat2.detach()).sum(-1, keepdim=True) / self.temp #[n, 1]
    #             pos2 = (output_feat1.detach() * output_feat2).sum(-1, keepdim=True) / self.temp #[n, 1]
    #
    #             # compute loss1
    #             b = 8000
    #             def run1(pos, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap, neg_max1):
    #                 # print("gpu: {}, i_1: {}".format(gpu, i))
    #                 mask1_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label1_overlap.unsqueeze(-1)).float() #[n, b]
    #                 neg1_idx = (output_feat1 @ output_ul_idx.T) / self.temp #[n, b]
    #                 logits1_neg_idx = (torch.exp(neg1_idx - neg_max1) * mask1_idx).sum(-1) #[n, ]
    #                 return logits1_neg_idx
    #
    #             def run1_0(pos, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap):
    #                 # print("gpu: {}, i_1_0: {}".format(gpu, i))
    #                 mask1_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label1_overlap.unsqueeze(-1)).float() #[n, b]
    #                 neg1_idx = (output_feat1 @ output_ul_idx.T) / self.temp #[n, b]
    #                 neg1_idx = torch.cat([pos, neg1_idx], 1) #[n, 1+b]
    #                 mask1_idx = torch.cat([torch.ones(mask1_idx.size(0), 1).float().cuda(), mask1_idx], 1) #[n, 1+b]
    #                 neg_max1 = torch.max(neg1_idx, 1, keepdim=True)[0] #[n, 1]
    #                 logits1_neg_idx = (torch.exp(neg1_idx - neg_max1) * mask1_idx).sum(-1) #[n, ]
    #                 return logits1_neg_idx, neg_max1
    #
    #             N = output_ul_all.size(0)
    #             logits1_down = torch.zeros(pos1.size(0)).float().cuda()
    #             for i in range((N-1)//b + 1):
    #                 # print("gpu: {}, i: {}".format(gpu, i))
    #                 pseudo_label_idx = pseudo_label_all[i*b:(i+1)*b]
    #                 output_ul_idx = output_ul_all[i*b:(i+1)*b]
    #                 if i == 0:
    #                     logits1_neg_idx, neg_max1 = torch.utils.checkpoint.checkpoint(run1_0, pos1, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap)
    #                 else:
    #                     logits1_neg_idx = torch.utils.checkpoint.checkpoint(run1, pos1, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap, neg_max1)
    #                 logits1_down += logits1_neg_idx
    #
    #             logits1 = torch.exp(pos1 - neg_max1).squeeze(-1) / (logits1_down + eps)
    #
    #             pos_mask_1 = ((pseudo_logits2_overlap > self.pos_thresh_value) & (pseudo_logits1_overlap < pseudo_logits2_overlap)).float()
    #             loss1 = -torch.log(logits1 + eps)
    #             loss1 = (loss1 * pos_mask_1).sum() / (pos_mask_1.sum() + 1e-12)
    #
    #             # compute loss2
    #             def run2(pos, output_feat2, output_ul_idx, pseudo_label_idx, pseudo_label2_overlap, neg_max2):
    #                 # print("gpu: {}, i_2: {}".format(gpu, i))
    #                 mask2_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label2_overlap.unsqueeze(-1)).float() #[n, b]
    #                 neg2_idx = (output_feat2 @ output_ul_idx.T) / self.temp #[n, b]
    #                 logits2_neg_idx = (torch.exp(neg2_idx - neg_max2) * mask2_idx).sum(-1) #[n, ]
    #                 return logits2_neg_idx
    #
    #             def run2_0(pos, output_feat2, output_ul_idx, pseudo_label_idx, pseudo_label2_overlap):
    #                 # print("gpu: {}, i_2_0: {}".format(gpu, i))
    #                 mask2_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label2_overlap.unsqueeze(-1)).float() #[n, b]
    #                 neg2_idx = (output_feat2 @ output_ul_idx.T) / self.temp #[n, b]
    #                 neg2_idx = torch.cat([pos, neg2_idx], 1) #[n, 1+b]
    #                 mask2_idx = torch.cat([torch.ones(mask2_idx.size(0), 1).float().cuda(), mask2_idx], 1) #[n, 1+b]
    #                 neg_max2 = torch.max(neg2_idx, 1, keepdim=True)[0] #[n, 1]
    #                 logits2_neg_idx = (torch.exp(neg2_idx - neg_max2) * mask2_idx).sum(-1) #[n, ]
    #                 return logits2_neg_idx, neg_max2
    #
    #             N = output_ul_all.size(0)
    #             logits2_down = torch.zeros(pos2.size(0)).float().cuda()
    #             for i in range((N-1)//b + 1):
    #                 pseudo_label_idx = pseudo_label_all[i*b:(i+1)*b]
    #                 output_ul_idx = output_ul_all[i*b:(i+1)*b]
    #                 if i == 0:
    #                     logits2_neg_idx, neg_max2 = torch.utils.checkpoint.checkpoint(run2_0, pos2, output_feat2, output_ul_idx, pseudo_label_idx, pseudo_label2_overlap)
    #                 else:
    #                     logits2_neg_idx = torch.utils.checkpoint.checkpoint(run2, pos2, output_feat2, output_ul_idx, pseudo_label_idx, pseudo_label2_overlap, neg_max2)
    #                 logits2_down += logits2_neg_idx
    #
    #             logits2 = torch.exp(pos2 - neg_max2).squeeze(-1) / (logits2_down + eps)
    #
    #             pos_mask_2 = ((pseudo_logits1_overlap > self.pos_thresh_value) & (pseudo_logits2_overlap < pseudo_logits1_overlap)).float()
    #
    #             loss2 = -torch.log(logits2 + eps)
    #             loss2 = (loss2 * pos_mask_2).sum() / (pos_mask_2.sum() + 1e-12)
    #
    #             loss_unsup = self.weight_unsup * (loss1 + loss2)
    #             curr_losses['loss1'] = loss1
    #             curr_losses['loss2'] = loss2
    #             curr_losses['loss_unsup'] = loss_unsup
    #             total_loss = total_loss + loss_unsup
    #             return total_loss, curr_losses, outputs

    # def _update_losses(self, cur_losses):
    #         for key in cur_losses:
    #             loss = cur_losses[key]
    #             n = loss.numel()
    #             count = torch.tensor([n]).long().cuda()
    #             dist.all_reduce(loss), dist.all_reduce(count)
    #             n = count.item()
    #             mean = loss.sum() / n
    #             if self.gpu == 0:
    #                 getattr(self, key).update(mean.item())


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

    def forward(self, features, feats_t=None, labels=None, predict=None, cb_mask=None):
        # feat from student, feats_2 from teacher if have

        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (features.shape[2], features.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == features.shape[-1], '{} {}'.format(labels.shape, features.shape)
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