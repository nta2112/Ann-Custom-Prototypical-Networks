import torch.nn as nn
import torchvision.models as models


def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ProtoNet(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    def __init__(self, backbone='cnn', x_dim=1, hid_dim=64, z_dim=64):
        super(ProtoNet, self).__init__()
        if backbone == 'resnet50':
            print("==> Initializing ResNet50 backbone (pretrained=True)")
            # Using pretrained weights
            self.encoder = models.resnet50(pretrained=True)
            # If grayscale (Omniglot), adapt the first conv layer
            if x_dim == 1:
                self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            # Remove the classification head (fc layer)
            # Output will be 2048 from AdaptiveAvgPool2d
            self.encoder.fc = nn.Identity()
        else:
            self.encoder = nn.Sequential(
                conv_block(x_dim, hid_dim),
                conv_block(hid_dim, hid_dim),
                conv_block(hid_dim, hid_dim),
                conv_block(hid_dim, z_dim),
            )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
