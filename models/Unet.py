import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, base_filters=64, input_size=(572, 572)):
        super(UNet, self).__init__()

        self.input_size = input_size

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

    def crop_and_concat(self, encoder_feature, decoder_feature):
        """
        Crop encoder_feature to match the size of decoder_feature and concatenate along dimension 1.
        """
        _, _, H, W = decoder_feature.size()
        encoder_feature = encoder_feature[:, :, :H, :W]  # Crop to match size
        return torch.cat([encoder_feature, decoder_feature], dim=1)

    def forward(self, x):
        # Encoder path
        e1 = self.encoder1(x)
        p1 = self.pool(e1)

        e2 = self.encoder2(p1)
        p2 = self.pool(e2)

        e3 = self.encoder3(p2)
        p3 = self.pool(e3)

        e4 = self.encoder4(p3)
        p4 = self.pool(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder path
        up4 = self.upconv4(b)
        d4 = self.decoder4(self.crop_and_concat(e4, up4))

        up3 = self.upconv3(d4)
        d3 = self.decoder3(self.crop_and_concat(e3, up3))

        up2 = self.upconv2(d3)
        d2 = self.decoder2(self.crop_and_concat(e2, up2))

        up1 = self.upconv1(d2)
        d1 = self.decoder1(self.crop_and_concat(e1, up1))

        # Final output
        out = self.final_conv(d1)
        return out

# Example usage
if __name__ == "__main__":
    num_classes = 3  # Example number of classes
    model = UNet(in_channels=3, num_classes=num_classes, base_filters=64, input_size=(256, 256))
    x = torch.randn(8, 3, *model.input_size)  # Example input
    output = model(x)
    print(output.shape)
