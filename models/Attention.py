import torch
import torch.nn as nn



class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channel, _, _ = x.size()
        y = x.view(batch, channel, -1).mean(dim=2)  # Global Average Pooling
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch, channel, 1, 1)  # Reshape to [batch, channel, 1, 1]
        return x * y

class LightweightAttention(nn.Module):
    def __init__(self, channel):
        super(LightweightAttention, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(channel // 8, channel, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise attention
        attn = self.conv1(x)
        attn = self.conv2(attn)
        attn = self.sigmoid(attn)
        return x * attn
class SelfAttention(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(channel, channel // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(channel, channel // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(channel, channel, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.reduction = reduction  # Reduction factor for downsampling
        self.downsample = nn.MaxPool2d(kernel_size=reduction)

    def forward(self, x):
        batch, channel, height, width = x.size()
        x_down = self.downsample(x)  # Downsample input
        query = self.query_conv(x_down).view(batch, -1, x_down.size(2) * x_down.size(3)).permute(0, 2, 1)
        key = self.key_conv(x_down).view(batch, -1, x_down.size(2) * x_down.size(3))
        value = self.value_conv(x_down).view(batch, -1, x_down.size(2) * x_down.size(3)).permute(0, 2, 1)

        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(attention, value).permute(0, 2, 1).view(batch, channel, x_down.size(2), x_down.size(3))
        out = nn.functional.interpolate(out, size=(height, width), mode='bilinear', align_corners=False)  # Upsample
        return out + x  # Residual connection


class SparseAttention(nn.Module):
    def __init__(self, channel, reduction=16, downsample_factor=4):
        super(SparseAttention, self).__init__()
        self.query_conv = nn.Conv2d(channel, channel // reduction, kernel_size=1)
        self.key_conv = nn.Conv2d(channel, channel // reduction, kernel_size=1)
        self.value_conv = nn.Conv2d(channel, channel, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.downsample = nn.MaxPool2d(kernel_size=downsample_factor)

    def forward(self, x):
        batch, channel, height, width = x.size()
        x_down = self.downsample(x)  # Downsample input
        query = self.query_conv(x_down).view(batch, -1, x_down.size(2) * x_down.size(3)).permute(0, 2, 1)
        key = self.key_conv(x_down).view(batch, -1, x_down.size(2) * x_down.size(3))
        value = self.value_conv(x_down).view(batch, -1, x_down.size(2) * x_down.size(3)).permute(0, 2, 1)

        attention = torch.bmm(query, key)
        attention = attention.masked_fill(attention < attention.mean(), 0)
        attention = self.softmax(attention)

        out = torch.bmm(attention, value).permute(0, 2, 1).view(batch, channel, x_down.size(2), x_down.size(3))
        out = nn.functional.interpolate(out, size=(height, width), mode='bilinear', align_corners=False)  # Upsample
        return out + x  # Residual connection