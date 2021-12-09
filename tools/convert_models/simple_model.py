import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import torch
from torch import nn

import numpy as np
import cv2
import torch.nn.functional as F
from core.models.resnet.iresnet import conv1x1, conv3x3

class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

class SimpleModel(nn.Module):
    fc_scale = 7 * 7
    def __init__(self,
                 dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(SimpleModel, self).__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        block = IBasicBlock
        layers = [3, 13, 30, 3]
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.features(x)

        #x = F.relu(x)
        return x

def main():
    rng = np.random.RandomState(1234)
    torch.manual_seed(1234)
    simple_model = SimpleModel()
    data = rng.normal(0, 1, size=(1, 3, 112, 112))
    data = cv2.imread("cvt_23.jpg")
    data = data - 128.0
    data = data * 0.078125
    data = np.transpose(data, (2, 0, 1))
    data= np.expand_dims(data, 0)
    #print("data:", data[0, 0, :5, :5])
    weight_path = "/home/users/han.tang/workspace/model_result/1206_mega_all_augment_agbt_bz200_sr10_circle/backbone-76520.pth"
    state_dict = torch.load(weight_path)
    simple_model.load_state_dict(state_dict)
    data = torch.tensor(data, dtype=torch.float32)
    simple_model.eval()
    out = simple_model(data)
    print("output:", out[0, :5])
    #print(simple_model.prelu.weight)
    state_dict = simple_model.state_dict()
    save_dict = dict()
    for p, v in state_dict.items():
        if p.find("tracked") > 0:
            continue
        save_dict[p] = v.cpu().numpy()
    save_dict["output"] = out.detach().cpu().numpy()
    np.savez("simple_dict.npz", **save_dict)

if __name__ == "__main__":
    main()
