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


class UNetWithAttention(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, base_filters=64, attention_type='SE'):
        super(UNetWithAttention, self).__init__()

        self.attention_type = attention_type

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        def up_block(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
            )

        self.encoder1 = conv_block(in_channels, base_filters)
        self.encoder2 = conv_block(base_filters, base_filters * 2)
        self.encoder3 = conv_block(base_filters * 2, base_filters * 4)
        self.encoder4 = conv_block(base_filters * 4, base_filters * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = conv_block(base_filters * 8, base_filters * 16)

        self.upconv4 = up_block(base_filters * 16, base_filters * 8)
        self.decoder4 = conv_block(base_filters * 16, base_filters * 8)

        self.upconv3 = up_block(base_filters * 8, base_filters * 4)
        self.decoder3 = conv_block(base_filters * 8, base_filters * 4)

        self.upconv2 = up_block(base_filters * 4, base_filters * 2)
        self.decoder2 = conv_block(base_filters * 4, base_filters * 2)

        self.upconv1 = up_block(base_filters * 2, base_filters)
        self.decoder1 = conv_block(base_filters * 2, base_filters)

        self.final_conv = nn.Conv2d(base_filters, num_classes, kernel_size=1)

        # Attention modules
        if attention_type == 'SE':
            self.attention4 = SqueezeExcitation(base_filters * 8)
            self.attention3 = SqueezeExcitation(base_filters * 4)
            self.attention2 = SqueezeExcitation(base_filters * 2)
            self.attention1 = SqueezeExcitation(base_filters)
        elif attention_type == 'LightweightAttention':
            self.attention4 = LightweightAttention(base_filters * 8)
            self.attention3 = LightweightAttention(base_filters * 4)
            self.attention2 = LightweightAttention(base_filters * 2)
            self.attention1 = LightweightAttention(base_filters)
        elif attention_type == 'SelfAttention':
            self.attention4 = SelfAttention(base_filters * 8)
            self.attention3 = SelfAttention(base_filters * 4)
            self.attention2 = SelfAttention(base_filters * 2)
            self.attention1 = SelfAttention(base_filters)
        elif attention_type == 'SparseAttention':
            self.attention4 = SparseAttention(base_filters * 8)
            self.attention3 = SparseAttention(base_filters * 4)
            self.attention2 = SparseAttention(base_filters * 2)
            self.attention1 = SparseAttention(base_filters)
        else:
            raise ValueError("Unsupported attention type. Choose either 'SE' or 'LightweightAttention'.")

    def crop_and_concat(self, encoder_feature, decoder_feature):
        _, _, H, W = decoder_feature.size()
        encoder_feature = encoder_feature[:, :, :H, :W]
        return torch.cat([encoder_feature, decoder_feature], dim=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        p1 = self.pool(e1)

        e2 = self.encoder2(p1)
        p2 = self.pool(e2)

        e3 = self.encoder3(p2)
        p3 = self.pool(e3)

        e4 = self.encoder4(p3)
        p4 = self.pool(e4)

        b = self.bottleneck(p4)

        up4 = self.upconv4(b)
        e4 = self.attention4(e4)
        d4 = self.decoder4(self.crop_and_concat(e4, up4))

        up3 = self.upconv3(d4)
        e3 = self.attention3(e3)
        d3 = self.decoder3(self.crop_and_concat(e3, up3))

        up2 = self.upconv2(d3)
        e2 = self.attention2(e2)
        d2 = self.decoder2(self.crop_and_concat(e2, up2))

        up1 = self.upconv1(d2)
        e1 = self.attention1(e1)
        d1 = self.decoder1(self.crop_and_concat(e1, up1))

        out = self.final_conv(d1)
        return torch.softmax(out, dim=1)

# Example usage
if __name__ == "__main__":
    model = UNetWithAttention(in_channels=3, num_classes=2, attention_type='SelfAttention').to('cuda')
    x = torch.randn(1, 3, 256, 256,device='cuda')  # Example input
    output = model(x)
    print(output.shape)
