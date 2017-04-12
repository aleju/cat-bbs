"""File that contains the CNN models."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import model_urls

class Model(nn.Module):
    """Simple CNN model to convert an input image to (1+9) heatmaps."""
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 128, 3, dilation=2, padding=2)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, 128, 3, dilation=4, padding=4)
        self.bn6 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128, 1+9, 3, padding=1)

    def forward(self, x):
        """Forward input images through the network to generate heatmaps."""
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.sigmoid(self.conv7(x))
        return x

class Model2(nn.Module):
    """CNN using a ResNet18 as its base to convert an image to (9+1) heatmaps.
    """
    def __init__(self):
        super(Model2, self).__init__()

        # fine tuning the ResNet helped significantly with the accuracy
        base_model = MyResNet(BasicBlock, [2, 2, 2, 2])
        base_model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        # code needed to deactivate fine tuning of resnet
        #for param in base_model.parameters():
        #    param.requires_grad = False
        self.base_model = base_model
        self.drop0 = nn.Dropout2d(0.05)

        self.conv1 = nn.Conv2d(512, 256, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.drop1 = nn.Dropout2d(0.05)

        self.conv2 = nn.Conv2d(256, 128, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.drop2 = nn.Dropout2d(0.05)

        self.conv3 = nn.Conv2d(128, 1+9, 3, padding=1, bias=False)

    def forward(self, x):
        """Forward input images through the network to generate heatmaps."""
        x = self.drop0(self.base_model(x))
        x = self.drop1(F.relu(self.bn1(self.conv1(x)), inplace=True))
        x = self.drop2(F.relu(self.bn2(self.conv2(x)), inplace=True))
        x = F.sigmoid(self.conv3(x))
        return x

    # code that is needed if the ResNet is not supposes to be fine tuned
    """
    def parameters(self, memo=None):
        if memo is None:
            memo = set()
        for p in self._parameters.values():
            if p is not None and p not in memo:
                memo.add(p)
                yield p
        for module in self.children():
            if module != self.base_model:
                for p in module.parameters(memo):
                    yield p
    """

# -------------------------------------------------------
# Code below is stuff mostly copied from the PyTorch repository
# only minor changes are done so that
# (a) the global average pooling and FC layer at the end are dropped,
#     i.e. resnet is used to convert images to 2D feature maps
# (b) some stride=2 is reduced to stride=1
#     to generate larger outputs (height and width).
# (c) dilation is added to ResNet's later layers to make up for the removed
#     stride (a trous trick), also to make it look at the image at a more
#     coarse level
# -------------------------------------------------------

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    # here with dilation
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1+(dilation-1)*(3-1), dilation=dilation, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class MyResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(MyResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # note the increasing dilation
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilation=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        # these layers will not be used
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, 1, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            # here with dilation
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # deactivated layers
        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x
