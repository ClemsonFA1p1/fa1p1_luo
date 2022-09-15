import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from configs import g_conf



__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class DNNBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(DNNBlock, self).__init__()

        self.fc1 = nn.Linear(1024, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(512,512)
        self.bn3 = nn.BatchNorm1d(512)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.fc1(x)
        #out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu(out)
       
        return out



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                              bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2)

        # TODO: THis is a super hardcoding ..., in order to fit my image size on resnet
        if block.__name__ == 'Bottleneck':
            self.fc = nn.Linear(6144, num_classes)
        else:
            self.fc = nn.Linear(1536, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
       
        x0 = self.maxpool(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, [x0, x1, x2, x3, x4]  # output, intermediate

    def get_layers_features(self, x):
        # Just get the intermediate layers directly.

        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x5 = self.avgpool(x4)
        x = x5.view(x.size(0), -1)
        x = self.fc(x)

        all_layers = [x0, x1, x2, x3, x4, x5, x]
        return all_layers
        
class MoCoCNN(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(MoCoCNN, self).__init__()
        self.fc1 = nn.Linear(2048, 52800)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                              bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        #self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 2, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2)

        # TODO: THis is a super hardcoding ..., in order to fit my image size on resnet
        if block.__name__ == 'Bottleneck':
            self.fc = nn.Linear(6144, num_classes)
        else:
            self.fc = nn.Linear(1536, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.fc1(x)
       
        x = torch.reshape(x, (x.shape[0],3, 88, 200))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
       
        x0 = self.maxpool(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, [x0, x1, x2, x3, x4]  # output, intermediate

    def get_layers_features(self, x):
        # Just get the intermediate layers directly.

        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x5 = self.avgpool(x4)
        x = x5.view(x.size(0), -1)
        x = self.fc(x)

        all_layers = [x0, x1, x2, x3, x4, x5, x]
        return all_layers

class DNN(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(DNN, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(1024, 512)
        #self.fc3 = nn.Linear(512, 512)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(3, stride=2)
        self.layer1 = self._make_layer(block, 2, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool1d(1, stride=1)

        # TODO: THis is a super hardcoding ..., in order to fit my image size on resnet
        if block.__name__ == 'Bottleneck':
            self.fc = nn.Linear(6144, num_classes)
        else:
            self.fc = nn.Linear(1536, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        #if stride != 1 or self.inplanes != planes * block.expansion:
        #    downsample = nn.Sequential(
        #        nn.Conv2d(self.inplanes, planes * block.expansion,
        #                  kernel_size=1, stride=stride, bias=False),
        #        nn.BatchNorm2d(planes * block.expansion),
        #    )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        #for i in range(1, blocks):
        #    layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.fc1(x)
      
        #x = self.relu(x)
        #x = self.fc2(x)
        #x = self.relu(x)
        #x = self.fc3(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        
       
        # #x0 = self.maxpool(x)
        x0 = x
      
       
        #x1 = self.layer1(x0)
        
        #x2 = self.layer2(x0)
        
        #x3 = self.layer3(x0)
        
        #x4 = self.layer4(x0)
        x1=x
        x2=x
        x3=x
        x4=x
        # #x = self.avgpool(x1)
        # x = x1
        # x = x.view(x.size(0), -1)
        
        #x = self.fc(x)
        
        
        return x, [x0, x1, x2, x3, x4]  # output, intermediate

    def get_layers_features(self, x):
        # Just get the intermediate layers directly.

        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x5 = self.avgpool(x4)
        x = x5.view(x.size(0), -1)
        x = self.fc(x)

        all_layers = [x0, x1, x2, x3, x4, x5, x]
        return all_layers

class Layers_5_DNN(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(Layers_5_DNN, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 512)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(3, stride=2)
        self.layer1 = self._make_layer(block, 2, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool1d(1, stride=1)

        # TODO: THis is a super hardcoding ..., in order to fit my image size on resnet
        if block.__name__ == 'Bottleneck':
            self.fc = nn.Linear(6144, num_classes)
        else:
            self.fc = nn.Linear(1536, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        #if stride != 1 or self.inplanes != planes * block.expansion:
        #    downsample = nn.Sequential(
        #        nn.Conv2d(self.inplanes, planes * block.expansion,
        #                  kernel_size=1, stride=stride, bias=False),
        #        nn.BatchNorm2d(planes * block.expansion),
        #    )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        #for i in range(1, blocks):
        #    layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        
       
        # #x0 = self.maxpool(x)
        x0 = x
      
       
        #x1 = self.layer1(x0)
        
        #x2 = self.layer2(x0)
        
        #x3 = self.layer3(x0)
        
        #x4 = self.layer4(x0)
        x1=x
        x2=x
        x3=x
        x4=x
        # #x = self.avgpool(x1)
        # x = x1
        # x = x.view(x.size(0), -1)
        
        #x = self.fc(x)
        

        return x, [x0, x1, x2, x3, x4]  # output, intermediate

    def get_layers_features(self, x):
        # Just get the intermediate layers directly.

        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x5 = self.avgpool(x4)
        x = x5.view(x.size(0), -1)
        x = self.fc(x)

        all_layers = [x0, x1, x2, x3, x4, x5, x]
        return all_layers

class Layers_3_DNN(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(Layers_3_DNN, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 512)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(3, stride=2)
        self.layer1 = self._make_layer(block, 2, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool1d(1, stride=1)

        # TODO: THis is a super hardcoding ..., in order to fit my image size on resnet
        if block.__name__ == 'Bottleneck':
            self.fc = nn.Linear(6144, num_classes)
        else:
            self.fc = nn.Linear(1536, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        #if stride != 1 or self.inplanes != planes * block.expansion:
        #    downsample = nn.Sequential(
        #        nn.Conv2d(self.inplanes, planes * block.expansion,
        #                  kernel_size=1, stride=stride, bias=False),
        #        nn.BatchNorm2d(planes * block.expansion),
        #    )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        #for i in range(1, blocks):
        #    layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        
       
        # #x0 = self.maxpool(x)
        x0 = x
      
       
        #x1 = self.layer1(x0)
        
        #x2 = self.layer2(x0)
        
        #x3 = self.layer3(x0)
        
        #x4 = self.layer4(x0)
        x1=x
        x2=x
        x3=x
        x4=x
        # #x = self.avgpool(x1)
        # x = x1
        # x = x.view(x.size(0), -1)
        
        #x = self.fc(x)
        

        return x, [x0, x1, x2, x3, x4]  # output, intermediate

    def get_layers_features(self, x):
        # Just get the intermediate layers directly.

        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x5 = self.avgpool(x4)
        x = x5.view(x.size(0), -1)
        x = self.fc(x)

        all_layers = [x0, x1, x2, x3, x4, x5, x]
        return all_layers
        
class Layers_6_DNN(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(Layers_6_DNN, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 512)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(3, stride=2)
        self.layer1 = self._make_layer(block, 2, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool1d(1, stride=1)

        # TODO: THis is a super hardcoding ..., in order to fit my image size on resnet
        if block.__name__ == 'Bottleneck':
            self.fc = nn.Linear(6144, num_classes)
        else:
            self.fc = nn.Linear(1536, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        #if stride != 1 or self.inplanes != planes * block.expansion:
        #    downsample = nn.Sequential(
        #        nn.Conv2d(self.inplanes, planes * block.expansion,
        #                  kernel_size=1, stride=stride, bias=False),
        #        nn.BatchNorm2d(planes * block.expansion),
        #    )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        #for i in range(1, blocks):
        #    layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        
        
       
        # #x0 = self.maxpool(x)
        x0 = x
      
       
        #x1 = self.layer1(x0)
        
        #x2 = self.layer2(x0)
        
        #x3 = self.layer3(x0)
        
        #x4 = self.layer4(x0)
        x1=x
        x2=x
        x3=x
        x4=x
        # #x = self.avgpool(x1)
        # x = x1
        # x = x.view(x.size(0), -1)
        
        #x = self.fc(x)
        

        return x, [x0, x1, x2, x3, x4]  # output, intermediate

    def get_layers_features(self, x):
        # Just get the intermediate layers directly.

        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x5 = self.avgpool(x4)
        x = x5.view(x.size(0), -1)
        x = self.fc(x)

        all_layers = [x0, x1, x2, x3, x4, x5, x]
        return all_layers
        
class Layers_7_DNN(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(Layers_7_DNN, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)
        self.fc0 = nn.Linear(2048, 2048)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 512)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(3, stride=2)
        self.layer1 = self._make_layer(block, 2, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool1d(1, stride=1)

        # TODO: THis is a super hardcoding ..., in order to fit my image size on resnet
        if block.__name__ == 'Bottleneck':
            self.fc = nn.Linear(6144, num_classes)
        else:
            self.fc = nn.Linear(1536, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        #if stride != 1 or self.inplanes != planes * block.expansion:
        #    downsample = nn.Sequential(
        #        nn.Conv2d(self.inplanes, planes * block.expansion,
        #                  kernel_size=1, stride=stride, bias=False),
        #        nn.BatchNorm2d(planes * block.expansion),
        #    )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        #for i in range(1, blocks):
        #    layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc0(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        
        
       
        # #x0 = self.maxpool(x)
        x0 = x
      
       
        #x1 = self.layer1(x0)
        
        #x2 = self.layer2(x0)
        
        #x3 = self.layer3(x0)
        
        #x4 = self.layer4(x0)
        x1=x
        x2=x
        x3=x
        x4=x
        # #x = self.avgpool(x1)
        # x = x1
        # x = x.view(x.size(0), -1)
        
        #x = self.fc(x)
        

        return x, [x0, x1, x2, x3, x4]  # output, intermediate

    def get_layers_features(self, x):
        # Just get the intermediate layers directly.

        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x5 = self.avgpool(x4)
        x = x5.view(x.size(0), -1)
        x = self.fc(x)

        all_layers = [x0, x1, x2, x3, x4, x5, x]
        return all_layers

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:

        model_dict = model_zoo.load_url(model_urls['resnet18'])
        # remove the fc layers
        del model_dict['fc.weight']
        del model_dict['fc.bias']
        state = model.state_dict()
        state.update(model_dict)
        model.load_state_dict(state)
    return model



def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    
    if g_conf.USE_MOCO:
        model = Layers_7_DNN(DNNBlock, [3, 4, 6, 3], **kwargs)
        #model = MoCoCNN(BasicBlock, [3, 4, 6, 3],  **kwargs)
    
    else:
        model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    
    if pretrained:

        model_dict = model_zoo.load_url(model_urls['resnet34'])
        # remove the fc layers
        del model_dict['fc.weight']
        del model_dict['fc.bias']
        state = model.state_dict()
        state.update(model_dict)
        model.load_state_dict(state)

   
         
    #   checkpoint = torch.load('moco_v3/checkpoint_0000.pth.tar')
    #   checkpoint_dict = checkpoint['state_dict']

    #   for k in list (checkpoint_dict.keys()):
    #     if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
    #         checkpoint_dict[k[len("module.encoder_q."):]] = checkpoint_dict[k]

    #         del checkpoint_dict[k]

    #   model.load_state_dict(checkpoint_dict, strict=False)
    #   print("loaded")

    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_dict = model_zoo.load_url(model_urls['resnet50'])
        # remove the fc layers
        del model_dict['fc.weight']
        del model_dict['fc.bias']
        state = model.state_dict()
        state.update(model_dict)
        model.load_state_dict(state)

    if g_conf.USE_MOCO:
        load_moco()
    return model



def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

def load_moco():
    
    checkpoint = torch.load('moco_v3/checkpoint_0000.pth.tar')
    checkpoint_dict = checkpoint['state_dict']

    for k in list (checkpoint_dict.keys()):
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            checkpoint_dict[k[len("module.encoder_q."):]] = checkpoint_dict[k]

        del checkpoint_dict[k]

    model.load_state_dict(checkpoint_dict, strict=False)
    
    return model