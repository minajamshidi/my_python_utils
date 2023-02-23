
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_model_summary import summary


# region functions
def conv2d_layer(chan1, chan2, activation=nn.ReLU(inplace=True), batchnorm=False):
    if batchnorm:
        return nn.Sequential(
            nn.Conv2d(in_channels=chan1, out_channels=chan2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(chan2),
            activation)
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels=chan1, out_channels=chan2, kernel_size=3, stride=1, padding=1),
            activation)


# endregion


# region conv nets

class AlexNet(nn.Module):
    def __init__(self, dim_image):
        super(AlexNet, self).__init__()
        n_layer = 5
        n_chan1 = 96
        n_chan2 = 256
        n_chan3 = 384
        n_chan4 = 384
        n_chan5 = 256
        n_neurons1 = 4096

        self.activation = nn.ReLU(inplace=True)
        self.drop_out = nn.Dropout(0.5)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=n_chan1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan1),
            self.activation,
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan1, out_channels=n_chan2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan2),
            self.activation,
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan2, out_channels=n_chan3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan3),
            self.activation,
            nn.MaxPool2d(2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan3, out_channels=n_chan4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan4),
            self.activation,
            nn.MaxPool2d(2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan4, out_channels=n_chan5, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan5),
            self.activation,
            nn.MaxPool2d(2)
        )

        n_pix_final = int(dim_image / 2**n_layer)
        self.fc1 = nn.Sequential(
            nn.Linear(n_chan5*n_pix_final**2, n_neurons1),
            self.activation
        )
        self.fc2 = nn.Sequential(
            nn.Linear(n_neurons1, n_neurons1),
            self.activation
        )
        self.fc3 = nn.Sequential(
            nn.Linear(n_neurons1, 2)
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.layer5(y)
        y = y.view(x.shape[0], -1)
        y = self.drop_out(y)
        y = self.fc1(y)
        y = self.drop_out(y)
        y = self.fc2(y)
        y = self.drop_out(y)
        y = self.fc3(y)
        return y


class VGG16(nn.Module):
    def __init__(self, dim_image):
        super(VGG16, self).__init__()
        n_maxpool2 = 5
        n_chan1 = 64
        n_chan2 = 128
        n_chan3 = 256
        n_chan4 = 512

        self.activation = nn.ReLU(inplace=True)
        self.drop_out = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.layer1_3_chan1_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=n_chan1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan1),
            self.activation,
        )  # dim_image

        self.layer2_chan1_chan1_2 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan1, out_channels=n_chan1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan1),
            self.activation,
            self.maxpool
        )  # dim_image / 2

        self.layer3_chan1_chan2_1 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan1, out_channels=n_chan2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan2),
            self.activation,
        )  # dim_image / 2

        self.layer4_chan2_chan2_2 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan2, out_channels=n_chan2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan2),
            self.activation,
            self.maxpool
        )  # dim_image / 2^2

        self.layer5_chan2_chan3_1 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan2, out_channels=n_chan3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan3),
            self.activation,
        )  # dim_image / 2^2

        self.layer6_chan3_chan3_2 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan3, out_channels=n_chan3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan3),
            self.activation,
        )  # dim_image / 2^2

        self.layer7_chan3_chan3_3 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan3, out_channels=n_chan3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan3),
            self.activation,
            self.maxpool
        )  # dim_image / 2^3

        self.layer8_chan3_chan4_1 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan3, out_channels=n_chan4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan4),
            self.activation,
        )  # dim_image / 2^3

        self.layer9_chan4_chan4_2 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan4, out_channels=n_chan4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan4),
            self.activation,
        )  # dim_image / 2^3

        self.layer10_chan4_chan4_3 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan4, out_channels=n_chan4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan4),
            self.activation,
            self.maxpool
        )  # dim_image / 2^4

        self.layer11_chan4_chan4_4 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan4, out_channels=n_chan4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan4),
            self.activation,
        )  # dim_image / 2^4

        self.layer12_chan4_chan4_5 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan4, out_channels=n_chan4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan4),
            self.activation,
        )  # dim_image / 2^4

        self.layer13_chan4_chan4_6 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan4, out_channels=n_chan4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan4),
            self.activation,
            self.maxpool
        )  # dim_image / 2^5

        # layer_AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(None)

        n_pix_final = int(dim_image / 2 ** n_maxpool2)
        n_chan_final = n_chan4

        self.fc1 = nn.Sequential(
            nn.Linear(n_chan_final * n_pix_final ** 2, n_chan4),
            # nn.BatchNorm1d(n_chan4),
            self.activation
        )
        self.fc2 = nn.Sequential(
            nn.Linear(n_chan4, 10),
            self.activation
        )
        self.fc3 = nn.Sequential(
            nn.Linear(10, 2)
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        y = self.layer1_3_chan1_1(x)
        y = self.layer2_chan1_chan1_2(y)
        y = self.layer3_chan1_chan2_1(y)
        y = self.layer4_chan2_chan2_2(y)
        y = self.layer5_chan2_chan3_1(y)
        y = self.layer6_chan3_chan3_2(y)
        y = self.layer7_chan3_chan3_3(y)
        y = self.layer8_chan3_chan4_1(y)
        y = self.layer9_chan4_chan4_2(y)
        y = self.layer10_chan4_chan4_3(y)
        y = self.layer11_chan4_chan4_4(y)
        y = self.layer12_chan4_chan4_5(y)
        y = self.layer13_chan4_chan4_6(y)
        y = y.view(x.shape[0], -1)
        y = self.drop_out(y)
        y = self.fc1(y)
        y = self.drop_out(y)
        y = self.fc2(y)
        y = self.drop_out(y)
        y = self.fc3(y)
        return y

# endregion


# region auto-encoders

class MyConvAutoEncoder1(torch.nn.Module):
    def __init__(self, img_chan):
        super().__init__()
        n_chan1 = 32
        n_chan2 = 8

        self.activation = nn.ReLU()

        # Encoder ----------
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=img_chan, out_channels=n_chan1, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_chan1),
            self.activation,
            nn.MaxPool2d(2)
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan1, out_channels=n_chan2, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_chan2),
            self.activation,
            nn.MaxPool2d(2)
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan2, out_channels=n_chan2, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_chan2),
            self.activation,
            # nn.MaxPool2d(2)
        )

        # Decoder ----------
        self.decoder1 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan2, out_channels=n_chan2, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_chan2),
            self.activation,
            nn.Upsample(scale_factor=2)
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan2, out_channels=n_chan1, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_chan1),
            self.activation,
            nn.Upsample(scale_factor=2)
        )

        self.decoder3 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan1, out_channels=img_chan, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(img_chan),
            self.activation,
            # nn.Upsample(scale_factor=2)
        )

    def forward(self, x):
        y = self.encoder1(x)
        y = self.encoder2(y)
        y = self.encoder3(y)
        y = self.decoder1(y)
        y = self.decoder2(y)
        y = self.decoder3(y)
        return y


class MyDenseAutoEncoder1(torch.nn.Module):
    def __init__(self, nfeatures1):
        super().__init__()
        nfeatures2 = 128
        self.encoder1 = nn.Sequential(nn.Linear(nfeatures1, nfeatures2),
                                      nn.ReLU())
        self.encoder2 = nn.Sequential(nn.Linear(nfeatures2, nfeatures2),
                                      nn.ReLU())
        self.decoder1 = nn.Sequential(nn.Linear(nfeatures2, nfeatures2),
                                      nn.ReLU())
        self.decoder2 = nn.Sequential(nn.Linear(nfeatures2, nfeatures1),
                                      nn.ReLU())

    def forward(self, x):
        y = self.encoder1(x)
        y = self.encoder2(y)
        y = self.decoder1(y)
        y = self.decoder2(y)
        return y


# endregion


# region UNETS
class UNET1(nn.Module):

    def __init__(self, n_chan_input, n_levels, start_power, exponent=2, kernel_encoder=3, kernel_decoder=3, kernel_upconv=3):
        super().__init__()
        self.n_levels = n_levels
        chans1 = [n_chan_input] + [exponent**power for power in range(start_power, start_power+self.n_levels)]
        chans2 = chans1[::-1][:-1]

        # maxpool and upsample
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)
        # encoder
        self.conv_encode = torch.nn.ModuleList([])
        for l in range(self.n_levels):
            self.conv_encode += torch.nn.ModuleList([nn.Conv2d(in_channels=chans1[l], out_channels=chans1[l+1],
                                                               kernel_size=kernel_encoder, stride=1, padding=1)])
            self.conv_encode += torch.nn.ModuleList([nn.Conv2d(in_channels=chans1[l+1], out_channels=chans1[l+1],
                                                               kernel_size=kernel_encoder, stride=1, padding=1)])


        self.conv_decode = torch.nn.ModuleList([])
        for l in range(self.n_levels-1):
            self.conv_decode += torch.nn.ModuleList([nn.Conv2d(in_channels=chans2[l], out_channels=chans2[l + 1],
                                                               kernel_size=kernel_upconv, stride=1, padding=1)])
            self.conv_decode += torch.nn.ModuleList([nn.Conv2d(in_channels=chans2[l], out_channels=chans2[l+1],
                                                               kernel_size=kernel_decoder, stride=1, padding=1)])
            self.conv_decode += torch.nn.ModuleList([nn.Conv2d(in_channels=chans2[l+1], out_channels=chans2[l+1],
                                                               kernel_size=kernel_decoder, stride=1, padding=1)])

        self.conv_final = nn.Conv2d(in_channels=chans2[-1], out_channels=1,
                                                               kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        y = x
        encoder_fea = [None for i in range(self.n_levels-1)]

        # encode
        for l in range(self.n_levels):
            y = self.maxpool(y) if l > 0 else y
            y = self.conv_encode[2*l](y)
            y = self.relu(y)
            y = F.dropout(y, p=0.1)
            y = self.conv_encode[2*l+1](y)
            y = self.relu(y)
            y = F.dropout(y, p=0.1)
            if l < self.n_levels-1 : encoder_fea[l] = y

        encoder_fea = encoder_fea[::-1]

        # decode
        for l in range(self.n_levels-1):
            y = self.upsample(y)
            y = self.conv_decode[3*l](y)
            y = self.relu(y)
            y = F.dropout(y, p=0.1)
            y = torch.concat((encoder_fea[l], y), dim=1)
            y = self.conv_decode[3*l+1](y)
            y = self.relu(y)
            y = F.dropout(y, p=0.3)
            y = self.conv_decode[3*l+2](y)
            y = F.dropout(y, p=0.2)
            y = self.relu(y)

        y = self.conv_final(y)
        y = torch.sigmoid(y)

        return y





# endregion

