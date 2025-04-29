import torchvision.models as models

def get_resnet101(pretrained=True):
    return models.resnet101(pretrained=pretrained)
