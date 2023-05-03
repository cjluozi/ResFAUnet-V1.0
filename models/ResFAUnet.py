import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class tiqu4_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(tiqu4_block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.block(x)
        out= F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        out= F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out= F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out= F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        return out

class tiqu3_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(tiqu3_block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.block(x)
        out= F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        out= F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out= F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        return out

class tiqu2_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(tiqu2_block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.block(x)
        out= F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        out= F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        return out

class tiqu1_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(tiqu1_block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.block(x)
        out= F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return out

class tiqu0_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(tiqu0_block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        return out

def final_block(in_channels, out_channels):
    block = torch.nn.Sequential(
        nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels, stride=1, padding=0),
    )
    return block

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self,in_channel,out_channel, block, num_block, groups=1, width_per_group=64):
        super().__init__()

        self.in_channels = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, self.in_channels, kernel_size = 7, stride = 2, padding = 3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(1024 + 2048, 1024)
        self.dconv_up2 = double_conv(512 + 1024, 512)
        self.dconv_up1 = double_conv(256 + 512, 256)

        self.dconv_last=nn.Sequential(
            nn.Conv2d(256+64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64,out_channel,1)
        )
        self.att4 = Attention_block(F_g=2048, F_l=1024, F_int=1024)
        self.att3 = Attention_block(F_g=1024, F_l=512, F_int=512)
        self.att2 = Attention_block(F_g=512, F_l=256, F_int=256)
        self.att1 = Attention_block(F_g=256, F_l=64, F_int=64)
        self.tiqu4_block = tiqu4_block(1024+2048,out_channel)
        self.tiqu3_block = tiqu3_block(512+1024,out_channel)
        self.tiqu2_block = tiqu2_block(256+512,out_channel)
        self.tiqu1_block = tiqu1_block(256+64,out_channel)
        self.tiqu0_block = tiqu0_block(out_channel,out_channel)
        self.final_layer = final_block(5*out_channel, out_channel)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channels,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channels = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channels,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)


    def forward(self, x):
        conv1 = self.conv1(x)
        temp=self.maxpool(conv1)
        conv2 = self.conv2_x(temp)
        conv3 = self.conv3_x(conv2)
        conv4 = self.conv4_x(conv3)
        bottle = self.conv5_x(conv4)
        x = self.upsample(bottle)
        conv4 = self.att4(g=x, x=conv4)
        x = torch.cat([x, conv4], dim=1)
        tiqu_block4 = self.tiqu4_block(x)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        conv3 = self.att3(g=x, x=conv3)
        x = torch.cat([x, conv3], dim=1)
        tiqu_block3 = self.tiqu3_block(x)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        conv2 = self.att2(g=x, x=conv2)
        x = torch.cat([x, conv2], dim=1)
        tiqu_block2 = self.tiqu2_block(x)
        x = self.dconv_up1(x)
        x=self.upsample(x)
        conv1 = self.att1(g=x, x=conv1)
        x=torch.cat([x,conv1],dim=1)
        tiqu_block1 = self.tiqu1_block(x)
        out=self.dconv_last(x)
        tiqu_block0 = self.tiqu0_block(out)
        decode_cat = torch.cat((tiqu_block4, tiqu_block3, tiqu_block2, tiqu_block1, tiqu_block0), dim=1)
        out = self.final_layer(decode_cat)
        return out

    def load_pretrained_weights(self):
        model_dict=self.state_dict()
        resnext101_weights = models.resnext101_32x8d(True).state_dict()
        count_res = 0
        count_my = 0
        reskeys = list(resnext101_weights.keys())
        mykeys = list(model_dict.keys())

        corresp_map = []
        while (True):
            reskey = reskeys[count_res]
            mykey = mykeys[count_my]
            if "fc" in reskey:
                break
            while reskey.split(".")[-1] not in mykey:
                count_my += 1
                mykey = mykeys[count_my]
            corresp_map.append([reskey, mykey])
            count_res += 1
            count_my += 1
        for k_res, k_my in corresp_map:
            model_dict[k_my]=resnext101_weights[k_res]
        try:
            self.load_state_dict(model_dict)
            print("Loaded resnext101_32x8d weights in mynet !")
        except:
            print("Loaded resnext101_32x8d weights in mynet !")
            raise



def ResNext101FPNAttentionUnet(in_channel,out_channel,pretrain=True):
    groups = 32
    width_per_group = 8
    model=ResNet(in_channel,out_channel,Bottleneck, [3, 4, 23, 3], groups=groups, width_per_group=width_per_group)
    if pretrain:
        model.load_pretrained_weights()
    return model


