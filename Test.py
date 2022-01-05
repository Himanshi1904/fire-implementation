import os
from argparse import ArgumentParser
import glob
import numpy as np
import torch
import SimpleITK as sitk
import nibabel as nb
import torch.utils.data as Data
from torch.autograd import Variable
from Functions import Dataset
from Models import Fire_Encoder, Synthesizer_Decoder, Transformation_Deformable_Network, SpatialTransformer

parser = ArgumentParser()
parser.add_argument("--modelpath", type=str,
                    dest="modelpath", default='../Model/model.pth',
                    help="frequency of saving models")
parser.add_argument("--savepath", type=str,
                    dest="savepath", default='../Result',
                    help="path for saving images")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--data_path_fixed", type=str,
                    dest="data_path_fixed", default='../Dataset/Fixed/',
                    help="fixed image")
parser.add_argument("--data_path_moving", type=str,
                    dest="data_path_moving", default='../Dataset/Moving/',
                    help="moving image")
opt = parser.parse_args()
data_path_fixed = opt.data_path_fixed
data_path_moving = opt.data_path_moving
start_channel = opt.start_channel
modelpath = opt.modelpath
savepath = opt.savepath

if not os.path.isdir(savepath):
    os.mkdir(savepath)


def test_model(epoch=1, batch_size=1):
    encoder_A_inference = Fire_Encoder(in_channel=1, start_channel=16).to("cuda")
    encoder_B_inference = Fire_Encoder(in_channel=1, start_channel=16).to("cuda")
    decoder_A_B_inference = Synthesizer_Decoder(start_channel=16).to("cuda")
    decoder_B_A_inference = Synthesizer_Decoder(start_channel=16).to("cuda")
    transformer_A_B_inference = Transformation_Deformable_Network(2).to("cuda")
    transformer_B_A_inference = Transformation_Deformable_Network(2).to("cuda")
    spatial_transformer_deformable_A_B_32_inference = SpatialTransformer(size=(32, 32, 32), is_affine=False).to("cuda")
    spatial_transformer_deformable_B_A_32_inference = SpatialTransformer(size=(32, 32, 32), is_affine=False).to("cuda")

    checkpoint = torch.load(modelpath)
    encoder_A_inference.load_state_dict(checkpoint['encoder_A'])
    encoder_B_inference.load_state_dict(checkpoint['encoder_B'])
    decoder_A_B_inference.load_state_dict(checkpoint['decoder_A_B'])
    decoder_B_A_inference.load_state_dict(checkpoint['decoder_B_A'])
    transformer_A_B_inference.load_state_dict(checkpoint['transformer_A_B'])
    transformer_B_A_inference.load_state_dict(checkpoint['transformer_B_A'])

    file_names_fixed = sorted(glob.glob(os.path.join(data_path_fixed, "*.nii.gz")))
    file_names_moving = sorted(glob.glob(os.path.join(data_path_moving, "*.nii.gz")))
    test_generator = Data.DataLoader(Dataset(file_names_fixed, file_names_moving, True), batch_size=1, shuffle=False)

    counter = 0
    for X, Y in test_generator:
        XA = X.cuda().float()  # fixed
        XB = Y.cuda().float()  # moving

        GX_A = encoder_A_inference(XA)
        GX_B = encoder_B_inference(XB)

        # Transforming images

        # Get the deformation field
        phi_A_B = transformer_A_B_inference(GX_A, GX_B)
        phi_B_A = transformer_B_A_inference(GX_B, GX_A)

        # dvf
        # dvf_A = spatial_transformer_deformable_A_B_32_inference(XA, phi_A_B)

        # Apply field to encoders

        GX_T_A = spatial_transformer_deformable_A_B_32_inference(GX_A, phi_A_B)
        GX_T_B = spatial_transformer_deformable_B_A_32_inference(GX_B, phi_B_A)

        # Registered images that actually deformed according to the other image through deformation field
        Xhat_T_B = decoder_A_B_inference(GX_T_A)  # (identical to the target image XB)
        Xhat_T_A = decoder_B_A_inference(GX_T_B)

        # Synthesis images that are identical to its counterpart without having deformation field applied
        Xhat_B = decoder_A_B_inference(GX_A)  # (B image registered to A).  (t2 registered to t1)
        Xhat_A = decoder_B_A_inference(GX_B)  # (A image registered to B) (t1 registered to t2)

        warped_A_tensor = Xhat_A.detach().to("cpu")
        warped_B_tensor = Xhat_B.detach().to("cpu")
        warped_A_np = warped_A_tensor.numpy()
        warped_B_np = warped_B_tensor.numpy()

        warped_B_nb = nb.Nifti1Image(warped_B_np[0, 0, :, :, :], np.eye(4))
        warped_A_nb = nb.Nifti1Image(warped_A_np[0, 0, :, :, :], np.eye(4))

        nb.save(warped_B_nb, savepath + os.path.basename(
            file_names_fixed[counter])[:-7] + '_F_' + os.path.basename(file_names_moving[counter])[
                                                      :-7] + '_M' + '.nii.gz')
        nb.save(warped_A_nb,
                savepath + os.path.basename(
                    file_names_moving[counter])[:-7] + '_F_' + os.path.basename(file_names_fixed[counter])[
                                                              :-7] + '_M' + '.nii.gz')

        counter = counter + 1
        del XA, XB, phi_A_B, phi_B_A, Xhat_T_B, Xhat_T_A, GX_T_A, GX_T_B, warped_A_nb, warped_A_tensor, warped_B_tensor, warped_B_np, warped_A_np
        torch.cuda.empty_cache()


test_model()
