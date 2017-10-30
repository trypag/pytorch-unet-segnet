import torch
import torch.nn as nn
import torch.nn.functional as F


class _Encoder(nn.Module):
    """Encoder layer encodes the features along the contracting path (left side),
    drop_rate parameter is used with respect to the paper and the official
    caffe model.

    Args:
        n_in_feat (int): number of input features
        n_out_feat (int): number of output features
        drop_rate (float): dropout rate at the end of the block
    """
    def __init__(self, n_in_feat, n_out_feat, drop_rate=0):
        super(_Encoder, self).__init__()

        layers = [nn.Conv2d(n_in_feat, n_out_feat, 3),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(n_out_feat, n_out_feat, 3),
                  nn.ReLU(inplace=True)]

        if drop_rate > 0:
            layers += [nn.Dropout(drop_rate)]

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)


class _Decoder(nn.Module):
    """Decoder layer decodes the features by performing deconvolutions and
    concatenating the resulting features with cropped features from the
    corresponding encoder (skip-connections). Encoder features are cropped
    because convolution operations does not allow to recover the same
    resolution in the expansive path (cf input image size > output
    segmentation map size).

    Args:
        n_in_feat (int): number of input features
        n_out_feat (int): number of output features
    """
    def __init__(self, n_in_feat, n_out_feat):
        super(_Decoder, self).__init__()

        self.encoder = _Encoder(n_in_feat, n_out_feat)
        self.decoder = nn.ConvTranspose2d(n_in_feat, n_out_feat, 2, 2)

    def forward(self, x, feat_encoder):
        feat_decoder = F.relu(self.decoder(x), True)

        # eval offset to allow cropping of the encoder's features
        crop_size = feat_decoder.size(-1)
        offset = (feat_encoder.size(-1) - crop_size) // 2
        crop = feat_encoder[:, :, offset:offset + crop_size,
                            offset:offset + crop_size]
        return self.encoder(torch.cat([feat_decoder, crop], 1))


class UNet(nn.Module):
    """U-Net: Convolutional Networks for Biomedical Image Segmentation
       based on https://arxiv.org/abs/1505.04597

    Args:
        num_classes (int): number of classes to segment
        n_init_features (int): number of input features in the fist convolution
        drop_rate (float): dropout rate of the last two encoders
        filter_config (list of 5 ints): number of output features at each level
    """
    def __init__(self, num_classes, n_init_features=1, drop_rate=0.5,
                 filter_config=(64, 128, 256, 512, 1024)):
        super(UNet, self).__init__()

        self.encoder1 = _Encoder(n_init_features, filter_config[0])
        self.encoder2 = _Encoder(filter_config[0], filter_config[1])
        self.encoder3 = _Encoder(filter_config[1], filter_config[2])
        self.encoder4 = _Encoder(filter_config[2], filter_config[3], drop_rate)
        self.encoder5 = _Encoder(filter_config[3], filter_config[4], drop_rate)

        self.decoder1 = _Decoder(filter_config[4], filter_config[3])
        self.decoder2 = _Decoder(filter_config[3], filter_config[2])
        self.decoder3 = _Decoder(filter_config[2], filter_config[1])
        self.decoder4 = _Decoder(filter_config[1], filter_config[0])

        # final classifier (equivalent to a fully connected layer)
        self.classifier = nn.Conv2d(64, num_classes, 1)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight)

    def forward(self, x):
        feat_encoder_1 = self.encoder1(x)
        feat_encoder_2 = self.encoder2(F.max_pool2d(feat_encoder_1, 2))
        feat_encoder_3 = self.encoder3(F.max_pool2d(feat_encoder_2, 2))
        feat_encoder_4 = self.encoder4(F.max_pool2d(feat_encoder_3, 2))
        feat_encoder_5 = self.encoder5(F.max_pool2d(feat_encoder_4, 2))

        feat_decoder = self.decoder1(feat_encoder_5, feat_encoder_4)
        feat_decoder = self.decoder2(feat_decoder, feat_encoder_3)
        feat_decoder = self.decoder3(feat_decoder, feat_encoder_2)
        feat_decoder = self.decoder4(feat_decoder, feat_encoder_1)

        return self.classifier(feat_decoder)
