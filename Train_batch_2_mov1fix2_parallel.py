# !/usr/bin/python
# coding=utf-8
import glob
import os
from argparse import ArgumentParser
import GPUtil
import torch
import torch.utils.data as Data

from Functions import Dataset
from Models_parallel import Fire_Encoder, Synthesizer_Decoder, SpatialTransformer, rmse_loss, smoothloss, \
    Transformation_Deformable_Network

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=200,
                    help="number of total iterations")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=4000,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath", default='../Datasets',
                    help="data path for training images")
opt = parser.parse_args()

lr = opt.lr
iteration = opt.iteration
start_channel = opt.start_channel
n_checkpoint = opt.checkpoint

datapath = "/nfs1/bajaj/IRDL/Datasets/Train/T2_fixed_T1_moving/"


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

    file_names_t1 = sorted(glob.glob(os.path.join(datapath + "/T1/", "*.nii.gz")))
    file_names_t2 = sorted(glob.glob(os.path.join(datapath + "/T2/", "*.nii.gz")))
    print(file_names_t1)

    optimizer = torch.optim.Adam(list(encoder_A.parameters()) + list(encoder_B.parameters()) +
                                 list(decoder_A_B.parameters()) + list(decoder_B_A.parameters()) +
                                 list(transformer_A_B.parameters()) + list(transformer_B_A.parameters())
                                 , lr=lr)

    model_dir = "/nfs1/bajaj/IRDL/FIRE/Model/MovingFixed/Batch2/"

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    training_generator = Data.DataLoader(Dataset(file_names_t1, file_names_t2, iteration, True), batch_size=1,
                                         shuffle=False)

    epochs = 400

    for epoch in range(epochs):
        step = 0
        print("Epoch no is:", epoch, "\n")
        for X, Y in training_generator:
            torch.cuda.empty_cache()

            XA = X.cuda().float()
            XB = Y.cuda().float()
            XA_64 = torch.nn.functional.interpolate(input=XA, scale_factor=0.5, align_corners=True,
                                                    mode='trilinear')
            XB_64 = torch.nn.functional.interpolate(input=XB, scale_factor=0.5, align_corners=True,
                                                    mode='trilinear')

            print("X shape: {}".format(XA_64.shape))
            print("Y shape: {}".format(XB_64.shape))
            GPUtil.showUtilization()

            GX_A_64 = encoder_A(XA)
            GX_B_64 = encoder_B(XB)

            # Transforming images

            # Get the deformation field
            phi_A_B = transformer_A_B(GX_A_64, GX_B_64)
            phi_B_A = transformer_B_A(GX_B_64, GX_A_64)

            print("phi_A_B shape: {}".format(phi_A_B.shape))
            print("GX_A shape: {}".format(GX_A_64.shape))
            GPUtil.showUtilization()

            print("phi_B_A shape: {}".format(phi_B_A.shape))
            print("GX_B shape: {}".format(GX_B_64.shape))
            GPUtil.showUtilization()

            # Apply field to encoders

            GX_T_A_64 = spatial_transformer_deformable_A_B_32(GX_A_64, phi_A_B)
            GX_T_B_64 = spatial_transformer_deformable_B_A_32(GX_B_64, phi_B_A)

            print("GX_T_A shape: {}".format(GX_T_A_64.shape))
            print("GX_T_B shape: {}".format(GX_T_B_64.shape))
            GPUtil.showUtilization()

            # Registered images that actually deformed according to the other image through deformation field
            Xhat_T_A_64 = decoder_A_B(GX_T_A_64)
            Xhat_T_B_64 = decoder_B_A(GX_T_B_64)
            GPUtil.showUtilization()

            print("Xhat_T_A shape: {}".format(Xhat_T_A_64.shape))
            print("Xhat_T_B shape: {}".format(Xhat_T_B_64.shape))

            # Synthesis images that are identical to its counterpart without having deformation field applied
            Xhat_B_64 = decoder_A_B(GX_A_64)
            Xhat_A_64 = decoder_B_A(GX_B_64)
            GPUtil.showUtilization()

            print("Xhat_B shape: {}".format(Xhat_B_64.shape))
            print("Xhat_A shape: {}".format(Xhat_A_64.shape))

            # synthesis loss, syn stands for synthesis
            loss_syn_acc = rmse_loss(Xhat_T_A_64, XB) + rmse_loss(Xhat_T_B_64, XA)
            loss_syn_fea = rmse_loss(GX_A_64, GX_T_A_64) + rmse_loss(GX_B_64, GX_T_B_64)
            GPUtil.showUtilization()
            cyc_1 = decoder_B_A(encoder_B(Xhat_B_64))
            cyc_2 = decoder_A_B(encoder_A(Xhat_A_64))
            print("cyc_1 shape: {}".format(cyc_1.shape))
            print("cyc_2 shape: {}".format(cyc_2.shape))
            GPUtil.showUtilization()
            loss_syn_cyc = rmse_loss(cyc_1, XA) + rmse_loss(cyc_2, XB)
            print("loss_syn_cyc: {}".format(loss_syn_cyc))
            del cyc_1, cyc_2

            align_1 = encoder_B(Xhat_B_64)
            align_2 = encoder_A(Xhat_A_64)
            loss_syn_align = rmse_loss(GX_A_64, align_1) + rmse_loss(GX_B_64, align_2)
            print("loss_syn_align: {}".format(loss_syn_align))
            GPUtil.showUtilization()
            del align_1, align_2

            loss_syn = loss_syn_acc + loss_syn_fea + loss_syn_cyc + loss_syn_align
            print("loss_syn: {}".format(loss_syn))

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

            # all losses
            total_loss = loss_syn + loss_reg + regularization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            del XA, XB, GX_A_64, GX_B_64, phi_A_B, phi_B_A, GX_T_A_64, GX_T_B_64, Xhat_B_64, \
                Xhat_A_64, phi_A_B_128, phi_B_A_128, Xhat_T_B_64, Xhat_T_A_64, XA_64, XB_64
            torch.cuda.empty_cache()

            if epoch % 4 == 0:
                modelname = model_dir + '/' + "fire_" + str(epoch) + '.pth'
                torch.save({"encoder_A": encoder_A.state_dict(), "encoder_B": encoder_B.state_dict(),
                            "decoder_A_B": decoder_A_B.state_dict(), "decoder_B_A": decoder_B_A.state_dict(),
                            "transformer_A_B": transformer_A_B.state_dict(),
                            "transformer_B_A": transformer_B_A.state_dict(),
                            "spatial_transformer_deformable_A_B_128": spatial_transformer_deformable_A_B_128.state_dict(),
                            "spatial_transformer_deformable_B_A_128": spatial_transformer_deformable_B_A_128.state_dict()},
                           modelname)
                print("Losses: {}, {} and {}".format(loss_syn, loss_reg, r_smooth))
                print("Total loss: {}".format(total_loss))
                print("Saving model checkpoints")
                print("======= =============== ===========")
                print()
            step += 1


train_model()
