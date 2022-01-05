import torch
import torch.nn as nn
import torch.nn.functional as F


# paper: https://arxiv.org/pdf/1907.05062.pdf

def conv3x3x3(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Fire_Encoder(nn.Module):
    def __init__(self, in_channel, start_channel):
        self.in_channel = in_channel
        self.start_channel = start_channel
        super(Fire_Encoder, self).__init__()
        self.encoder0 = self.encoder(self.in_channel, self.start_channel, bias=True)

        self.encoder1 = self.encoder(self.start_channel, self.start_channel * 2, stride=2, bias=True)

        self.encoder2 = self.encoder(self.start_channel * 2, self.start_channel * 4, stride=2, bias=True)

        self.residual_block1 = ResidualBlock(self.start_channel * 4, self.start_channel * 4)

        self.residual_block2 = ResidualBlock(self.start_channel * 4, self.start_channel * 4)

        self.residual_block3 = ResidualBlock(self.start_channel * 4, self.start_channel * 4)

        self.residual_block4 = ResidualBlock(self.start_channel * 4, self.start_channel * 4)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU())
        layer  # .to("cuda")
        return layer

    def forward(self, x):
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.residual_block1(e2)
        e4 = self.residual_block1(e3)
        e5 = self.residual_block1(e4)
        e6 = self.residual_block1(e5)

        return e6


class Synthesizer_Decoder(nn.Module):
    def __init__(self, start_channel):
        # self.in_channel = in_channel
        self.start_channel = start_channel

        ## Declarations #####
        super(Synthesizer_Decoder, self).__init__()
        self.rb1 = ResidualBlock(self.start_channel * 4, self.start_channel * 4, 1)
        self.rb2 = ResidualBlock(self.start_channel * 4, self.start_channel * 4, 1)
        self.rb3 = ResidualBlock(self.start_channel * 4, self.start_channel * 4, 1)
        self.rb4 = ResidualBlock(self.start_channel * 4, self.start_channel * 4, 1)

        self.up1 = self.decoder(self.start_channel * 4, self.start_channel * 2)
        self.up2 = self.decoder(self.start_channel * 2, self.start_channel * 1)

        self.cb = self.convblock(self.start_channel * 1, self.start_channel // 2)
        self.out = self.onecrossoneblock(self.start_channel // 2, 1)

    # Decoder upsampler block start #
    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.LeakyReLU())
        return layer

    def convblock(self, in_channels, out_channels, kernel_size=3,
                  bias=False, batchnorm=False):
        layer = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias, padding=0), )
        return layer

    def onecrossoneblock(self, in_channels, out_channels=1, kernel_size=1,
                         bias=False, batchnorm=False):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias, padding=1),
        )
        return layer

    def forward(self, x):
        rout = self.rb1(x)
        rout = self.rb2(rout)
        rout = self.rb3(rout)
        rout = self.rb4(rout)
        dec1 = self.up1(rout)
        dec2 = self.up2(dec1)
        cbo = self.cb(dec2)
        output = self.out(cbo)
        return output


#  grid sampler code

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, is_affine=False, theta=None, mode='bilinear', affine_image_size=(2, 1, 128, 128, 128)):
        super().__init__()

        self.mode = mode
        self.isaffine = is_affine
        self.theta = theta
        self.affine_image_size = affine_image_size
        # create sampling grid
        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict

        if (self.isaffine):
            grid = F.affine_grid(self.theta, self.affine_image_size, align_corners=False)
            # grid = grid.permute(0, 4, 1, 2, 3)
            self.register_buffer('grid', grid)
        else:
            vectors = [torch.arange(0, s) for s in size]
            grids = torch.meshgrid(vectors)
            grid = torch.stack(grids)
            grid = torch.unsqueeze(grid, 0)
            grid = grid.type(torch.FloatTensor)
            self.register_buffer('grid', grid)

    def forward(self, src, flow=None):
        if (self.isaffine):
            grid = F.affine_grid(self.theta, self.affine_image_size)

            warped_image = F.grid_sample(src, grid)

            # warped_image = warped_image.permute(0, 4, 1, 2, 3)
            return warped_image
        else:
            # new locations
            new_locs = self.grid + flow
            shape = flow.shape[2:]

            # need to normalize grid values to [-1, 1] for resampler
            for i in range(len(shape)):
                new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

            # move channels dim to last position
            # also not sure why, but the channels need to be reversed
            if len(shape) == 2:
                new_locs = new_locs.permute(0, 2, 3, 1)
                new_locs = new_locs[..., [1, 0]]
            elif len(shape) == 3:
                new_locs = new_locs.permute(0, 2, 3, 4, 1)
                new_locs = new_locs[..., [2, 1, 0]]

            return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class Transformation_Deformable_Network(nn.Module):
    def __init__(self, start_channel):
        # self.in_channel = in_channel
        self.start_channel = start_channel
        super(Transformation_Deformable_Network, self).__init__()

        self.convblock1 = self.convblock(self.start_channel * 32, 8)
        self.convblock2 = self.convblock(self.start_channel * 32, 8)

        self.rb1 = ResidualBlock(16, 16, 1)

        ## Harcoded to get the output channels to 3 as deformable field has 3 fields ##
        self.convblock3 = self.convblock(16, 3)
        self.lkrelublock1 = self.leakyrelublock()
        self.lkrelublock2 = self.leakyrelublock()
        self.lkrelublock3 = self.leakyrelublock()

        self.inb1 = self.instancenormblock(3)
        self.inb2 = self.instancenormblock(3)

        self.tb1 = self.tanhblock()

        return;

    def convblock(self, in_channels, out_channels, kernel_size=3, bias=False, batchnorm=False):
        layer = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias, padding=1), )
        return layer

    def leakyrelublock(self):
        layer = nn.LeakyReLU()
        return layer

    def instancenormblock(self, out_channels):
        layer = nn.InstanceNorm3d(out_channels)
        return layer

    def tanhblock(self):
        layer = nn.Tanh()
        return layer

    def forward(self, gx, gy):
        cb1 = self.convblock1(gx)
        cb1 = self.lkrelublock1(cb1)
        cb2 = self.convblock2(gy)
        cb2 = self.lkrelublock2(cb2)

        cat_in = torch.cat((cb1, cb2), 1)

        rb = self.rb1(cat_in)
        print(rb.shape)
        ib1 = self.inb1(rb)
        print(ib1.shape)
        lk = self.lkrelublock3(ib1)
        cb3 = self.convblock3(lk)
        ib2 = self.inb2(cb3)
        tanhb1 = self.tb1(ib2)
        return tanhb1


def rmse_loss(input, target):
    y_true_f = input.view(-1)
    y_pred_f = target.view(-1)
    diff = y_true_f - y_pred_f
    mse = torch.mul(diff, diff).mean()
    rmse = torch.sqrt(mse)
    return rmse


def smoothloss(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
    dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
    dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])
    return (torch.mean(dx * dx) + torch.mean(dy * dy) + torch.mean(dz * dz)) / 3.0
