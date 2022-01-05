# !/usr/bin/python
# coding=utf-8
import glob
import os
import math
from argparse import ArgumentParser
import torch
import random
import numpy as np
import torch.utils.data as Data
import warnings
from Models import Fire_Encoder, Synthesizer_Decoder, SpatialTransformer, rmse_loss, smoothloss, \
    Transformation_Deformable_Network
from Functions import Dataset

torch.manual_seed(1701)
random.seed(1701)
np.random.seed(1701)
warnings.simplefilter("ignore", UserWarning)

parser = ArgumentParser()
parser.add_argument("--lr1", type=float,
                    dest="lr1", default=5e-5, help="learning rate for transformer network")
parser.add_argument("--lr2", type=float,
                    dest="lr2", default=1e-4, help="learning rate for encoder/decoder network")
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=200,
                    help="number of total iterations")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--datapath_f", type=str,
                    dest="datapath_f", default='../Dataset',
                    help="data path for training images t1 fixed")
parser.add_argument("--datapath_m", type=str,
                    dest="datapath_m", default='../Dataset',
                    help="data path for training images t1 moving")

opt = parser.parse_args()

lr1 = opt.lr1
lr2 = opt.lr2
iteration = opt.iteration
start_channel = opt.start_channel
datapath_f = opt.datapath_f
datapath_m = opt.datapath_m


def train_model():
    torch.cuda.empty_cache()  # for emptying the memory occupied by cuda

    encoder_A = Fire_Encoder(in_channel=1, start_channel=16).to("cuda")
    encoder_B = Fire_Encoder(in_channel=1, start_channel=16).to("cuda")
    decoder_A_B = Synthesizer_Decoder(start_channel=16).to("cuda")
    decoder_B_A = Synthesizer_Decoder(start_channel=16).to("cuda")
    transformer_A_B = Transformation_Deformable_Network(2).to("cuda")
    transformer_B_A = Transformation_Deformable_Network(2).to("cuda")
    spatial_transformer_deformable_A_B_32 = SpatialTransformer(size=(32, 32, 32), is_affine=False).to("cuda")
    spatial_transformer_deformable_B_A_32 = SpatialTransformer(size=(32, 32, 32), is_affine=False).to("cuda")
    spatial_transformer_deformable_A_B_128 = SpatialTransformer(size=(128, 128, 128), is_affine=False).to("cuda")
    spatial_transformer_deformable_B_A_128 = SpatialTransformer(size=(128, 128, 128), is_affine=False).to("cuda")

    for param in spatial_transformer_deformable_A_B_32.parameters():
        param.requires_grad = False
        param.volatile = True

    for param in spatial_transformer_deformable_B_A_32.parameters():
        param.requires_grad = False
        param.volatile = True

    for param in spatial_transformer_deformable_A_B_128.parameters():
        param.requires_grad = False
        param.volatile = True

    for param in spatial_transformer_deformable_B_A_128.parameters():
        param.requires_grad = False
        param.volatile = True

    file_names_f = sorted(glob.glob(os.path.join(datapath_f, "*.nii.gz")))
    file_names_m = sorted(glob.glob(os.path.join(datapath_m, "*.nii.gz")))

    optimizer_synthesis = torch.optim.Adam(list(encoder_A.parameters()) + list(encoder_B.parameters()) +
                                           list(decoder_A_B.parameters()) + list(decoder_B_A.parameters())
                                           , lr=lr2)
    optimizer_transformation = torch.optim.Adam(list(transformer_A_B.parameters()) + list(transformer_B_A.parameters())
                                                , lr=lr1)

    model_dir = '../Model'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    dataset = Dataset(file_names_f, file_names_m, iteration, True)
    train_set, val_set = Data.random_split(dataset,
                                           [round(len(dataset) * 0.6), len(dataset) - round(len(dataset) * 0.6)])
    training_generator = Data.DataLoader(train_set, batch_size=2, shuffle=True)
    validation_generator = Data.DataLoader(val_set, batch_size=2, shuffle=False)

    epochs = 1000
    best_loss = math.inf

    def forward_pass(X, Y):
        XA = X.cuda().float()
        XB = Y.cuda().float()

        GX_A_64 = encoder_A(XA)
        GX_B_64 = encoder_B(XB)

        # Transforming images

        # Get the deformation field
        phi_A_B = transformer_A_B(GX_A_64, GX_B_64)
        phi_B_A = transformer_B_A(GX_B_64, GX_A_64)

        # Apply field to encoders

        GX_T_A_64 = spatial_transformer_deformable_A_B_32(GX_A_64, phi_A_B)
        GX_T_B_64 = spatial_transformer_deformable_B_A_32(GX_B_64, phi_B_A)

        # Registered images that actually deformed according to the other image through deformation field
        Xhat_T_B_64 = decoder_A_B(GX_T_A_64)
        Xhat_T_A_64 = decoder_B_A(GX_T_B_64)

        # Synthesis images that are identical to its counterpart without having deformation field applied
        Xhat_B_64 = decoder_A_B(GX_A_64)
        Xhat_A_64 = decoder_B_A(GX_B_64)

        # synthesis loss, syn stands for synthesis
        loss_syn_acc = rmse_loss(Xhat_T_B_64, XB) + rmse_loss(Xhat_T_A_64, XA)
        loss_syn_fea = rmse_loss(GX_A_64, GX_T_B_64) + rmse_loss(GX_B_64, GX_T_A_64)
        cyc_1 = decoder_B_A(encoder_B(Xhat_B_64))
        cyc_2 = decoder_A_B(encoder_A(Xhat_A_64))
        loss_syn_cyc = rmse_loss(cyc_1, XA) + rmse_loss(cyc_2, XB)
        del cyc_1, cyc_2

        align_1 = encoder_B(Xhat_B_64)
        align_2 = encoder_A(Xhat_A_64)
        loss_syn_align = rmse_loss(GX_A_64, align_1) + rmse_loss(GX_B_64, align_2)
        del align_1, align_2

        loss_syn = loss_syn_acc + loss_syn_fea + loss_syn_cyc + loss_syn_align

        # registration loss

        phi_A_B_128 = torch.nn.functional.interpolate(input=phi_A_B, scale_factor=4, align_corners=True,
                                                      mode='trilinear')
        phi_B_A_128 = torch.nn.functional.interpolate(input=phi_B_A, scale_factor=4, align_corners=True,
                                                      mode='trilinear')

        acc_1 = decoder_A_B((encoder_A(spatial_transformer_deformable_A_B_128(XA, phi_A_B_128))))
        acc_2 = decoder_B_A((encoder_B(spatial_transformer_deformable_B_A_128(XB, phi_B_A_128))))
        loss_reg_acc = rmse_loss(acc_1, XB) + rmse_loss(acc_2, XA)
        del acc_1, acc_2
        ic_1 = spatial_transformer_deformable_A_B_128(spatial_transformer_deformable_A_B_128(XA, phi_A_B_128),
                                                      phi_B_A_128)
        ic_2 = spatial_transformer_deformable_B_A_128(spatial_transformer_deformable_B_A_128(XB, phi_B_A_128),
                                                      phi_A_B_128)
        loss_reg_ic = rmse_loss(XA, ic_1) + rmse_loss(XB, ic_2)
        del ic_1, ic_2

        loss_reg = loss_reg_acc + loss_reg_ic

        # regularization
        r_lambda = 0.05  # 64 / (10* 128)
        # r_syn = not calculating as this training loop only does deformable regularization
        # r_reg =
        r_smooth = smoothloss(phi_A_B) + smoothloss(phi_B_A)
        regularization = r_lambda * r_smooth

        total_loss = loss_syn + loss_reg + regularization

        return total_loss, loss_syn, loss_reg, r_smooth

    for epoch in range(epochs):
        print("Epoch no is:", epoch, "\n")

        encoder_A.train(), encoder_B.train(), decoder_A_B.train(), decoder_B_A.train(), transformer_A_B.train(), \
        transformer_B_A.train(), spatial_transformer_deformable_A_B_32.train(), spatial_transformer_deformable_B_A_32.train(),
        spatial_transformer_deformable_A_B_128.train(), spatial_transformer_deformable_B_A_128.train()

        for X, Y in training_generator:
            torch.cuda.empty_cache()

            optimizer_transformation.zero_grad()
            total_loss, loss_syn, loss_reg, r_smooth = forward_pass(X, Y)
            total_loss.backward()
            optimizer_transformation.step()

            optimizer_synthesis.zero_grad()
            total_loss, loss_syn, loss_reg, r_smooth = forward_pass(X, Y)
            total_loss.backward()
            optimizer_synthesis.step()

            torch.cuda.empty_cache()

        if epoch % 4 == 0:
            modelname = model_dir + '/' + "fire_" + str(epoch) + '.pth'
            torch.save({"encoder_A": encoder_A.state_dict(), "encoder_B": encoder_B.state_dict(),
                        "decoder_A_B": decoder_A_B.state_dict(), "decoder_B_A": decoder_B_A.state_dict(),
                        "transformer_A_B": transformer_A_B.state_dict(),
                        "transformer_B_A": transformer_B_A.state_dict(),
                        "spatial_transformer_deformable_A_B_32": spatial_transformer_deformable_A_B_32.state_dict(),
                        "spatial_transformer_deformable_B_A_32": spatial_transformer_deformable_B_A_32.state_dict(),
                        "spatial_transformer_deformable_A_B_128": spatial_transformer_deformable_A_B_128.state_dict(),
                        "spatial_transformer_deformable_B_A_128": spatial_transformer_deformable_B_A_128.state_dict()},
                       modelname)
            print("Losses: {}, {} and {}".format(loss_syn, loss_reg, r_smooth))
            print("Total loss: {}".format(total_loss))
            print("Saving model checkpoints")
            print("======= =============== ===========")
            print()

        encoder_A.eval(), encoder_B.eval(), decoder_A_B.eval(), decoder_B_A.eval(), transformer_A_B.eval(), \
        transformer_B_A.eval(), spatial_transformer_deformable_A_B_32.eval(), spatial_transformer_deformable_B_A_32.eval(),
        spatial_transformer_deformable_A_B_128.eval(), spatial_transformer_deformable_B_A_128.eval()
        losses = []
        with torch.no_grad():
            for X, Y in validation_generator:
                torch.cuda.empty_cache()
                total_loss, loss_syn, loss_reg, r_smooth = forward_pass(X, Y)
                losses.append(total_loss.item())
                torch.cuda.empty_cache()

        if best_loss > np.median(losses):
            best_loss = np.median(losses)
            modelname = model_dir + '/' + "fire_best_" + str(epoch) + '.pth'
            torch.save({"encoder_A": encoder_A.state_dict(), "encoder_B": encoder_B.state_dict(),
                        "decoder_A_B": decoder_A_B.state_dict(), "decoder_B_A": decoder_B_A.state_dict(),
                        "transformer_A_B": transformer_A_B.state_dict(),
                        "transformer_B_A": transformer_B_A.state_dict(),
                        "spatial_transformer_deformable_A_B_32": spatial_transformer_deformable_A_B_32.state_dict(),
                        "spatial_transformer_deformable_B_A_32": spatial_transformer_deformable_B_A_32.state_dict(),
                        "spatial_transformer_deformable_A_B_128": spatial_transformer_deformable_A_B_128.state_dict(),
                        "spatial_transformer_deformable_B_A_128": spatial_transformer_deformable_B_A_128.state_dict()},
                       modelname)
            print("Saving best model checkpoints")
            print("======= =============== ===========")
            print()


train_model()
