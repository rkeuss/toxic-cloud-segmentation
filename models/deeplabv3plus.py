# from paper chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/2012.00827
# Adapted from https://github.com/RK621/ThreeStageSelftraining_SemanticSegmentation/blob/main/architectures/deeplab3plus.py
# Adapted from Gongfan Fang's implementation of DeepLab v3+ found at:
# https://github.com/VainF/DeepLabV3Plus-Pytorch
# Also used Yude Wang's implementation as a reference:
# https://github.com/YudeWang/deeplabv3plus-pytorch

from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import resnet, ResNet101_Weights
    from torchvision.models._utils import IntermediateLayerGetter
    from torchvision.models.segmentation.deeplabv3 import ASPP, DeepLabV3, DeepLabHead

except ImportError:
    ResNet101_Weights = None
    IntermediateLayerGetter = None
    ASPP = None
    DeepLabV3 = None


def freeze_bn_module(m):
    """ Freeze the module `m` if it is a batch norm layer.

    :param m: a torch module
    :param mode: 'eval' or 'no_grad'
    """
    classname = type(m).__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        # As in Yude Wang's implementation
        # (https://github.com/YudeWang/deeplabv3plus-pytorch/blob/master/lib/net/deeplabv3plus.py)
        # we have two conv-bn-relu blocks, rather than 1 as in Gongfan Fang's implementation
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project(feature['low_level'])
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabV3Plus(nn.Module):
    def __init__(self, backbone, classifier):
        super(DeepLabV3Plus, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        classifier_out = self.classifier(features)
        logits = F.interpolate(classifier_out, size=input_shape, mode='bilinear', align_corners=False)
        return logits, features['out']  # return both final logits and high-level feature maps


def _deeplabv3plus(backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    if backbone_name == 'resnet101':
        weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = resnet.resnet101(weights=weights, replace_stride_with_dilation=replace_stride_with_dilation)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    inplanes = 2048
    low_level_planes = 256

    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3Plus(backbone, classifier)
    return model


class DeepLabv3Wrapper (nn.Module):
    BLOCK_SIZE = (1, 1)
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    def __init__(self, model, pretraining=None):
        super(DeepLabv3Wrapper, self).__init__()

        self.deeplab = model
        self.pretraining = pretraining


    def forward(self, x, feature_maps=False, use_dropout=False):
        logits, features = self.deeplab(x)
        if feature_maps:
            return logits, features
        return logits


    def freeze_batchnorm(self):
        self.deeplab.backbone.apply(freeze_bn_module)


    def _backbone_parameters(self):
        return list(self.deeplab.backbone.parameters())

    def _classifier_end_parameters(self):
        if isinstance(self.deeplab.classifier, DeepLabHead):
            return list(self.deeplab.classifier[-1].parameters())
        elif isinstance(self.deeplab.classifier, DeepLabHeadV3Plus):
            return list(self.deeplab.classifier.classifier[-1].parameters())
        else:
            raise TypeError('Oh dear, seem to have encountered unknown classifier head type {}'.format(
                type(self.deeplab.classifier)
            ))


    def pretrained_parameters(self):
        if self.pretraining is None:
            return []
        elif self.pretraining == 'imagenet':
            return self._backbone_parameters()
        elif self.pretraining == 'coco':
            new_ids = [id(p) for p in self._classifier_end_parameters()]
            return [p for p in self.parameters() if id(p) not in new_ids]
        else:
            raise ValueError('Unknown pretraining {}'.format(self.pretraining))

    def new_parameters(self):
        if self.pretraining is None:
            return list(self.parameters())
        elif self.pretraining == 'imagenet':
            backbone_ids = [id(p) for p in self._backbone_parameters()]
            return [p for p in self.parameters() if id(p) not in backbone_ids]
        elif self.pretraining == 'coco':
            return self._classifier_end_parameters()
        else:
            raise ValueError('Unknown pretraining {}'.format(self.pretraining))



def resnet101_deeplabv3plus_imagenet(num_classes, pretrained=True):
    deeplab = _deeplabv3plus('resnet101', num_classes, 8, pretrained)
    return DeepLabv3Wrapper(deeplab)