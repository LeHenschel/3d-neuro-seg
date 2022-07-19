import argparse
import os
import os.path as op
import sys
import logging
import time
import numpy as np
import nibabel as nib
import torch
import glob

# from data_utils.preprocessing.preprocess_utils import load_paths

from model.QuadNet import assign_nets_to_coords
from model.RCVNet import RCVNet

from utils.fastsurfer.load_neuroimaging_data import load_and_conform_image
import FastInfantSurfer.data_util.data_utils as du

from tqdm import tqdm
from os import makedirs
from itertools import product

HELPTEXT = """
Script to generate aparc.DKTatlas+aseg.deep.mgz using 3D Deep Networks. \n 
Mirrors FastSurfer's eval method. \n

Dependencies:
    Torch 
    Torchvision
    Skimage
    Numpy
    Matplotlib
    h5py
    scipy
    Python 3.5
    Nibabel (to read and write neuroimaging data, http://nipy.org/nibabel/)

Author: Saikat Roy

Modified by David Kügler
"""


def options_parse():
    """
    Command line option parser
    """
    parser = argparse.ArgumentParser(description=HELPTEXT, epilog='$Id: 3d_segmentation, v 0.1$')

    # 1. Directory information (where to read from, where to write to)
    parser.add_argument('--i_dir', '--input_directory', dest='input', help='path to directory of input volume(s).')
    parser.add_argument('--csv_file', '--csv_file', help="CSV-file with directories to process", default=None)
    parser.add_argument('--o_dir', '--output_directory', dest='output',
                        help='path to output directory. Will be created if it does not already exist')

    # 2. Options for the MRI volumes (name of in and output, order of interpolation if not conformed)
    parser.add_argument('--in_name', '--input_name', dest='iname', help='name of file to process. Default: orig.mgz',
                        default='orig.mgz')
    parser.add_argument('--out_name', '--output_name', dest='oname', default='aparc.DKTatlas+aseg.deep.mgz',
                        help='name under which segmentation will be saved. Default: aparc.DKTatlas+aseg.deep.mgz. '
                             'If a separate subfolder is desired (e.g. FS conform, add it to the name: '
                             'mri/aparc.DKTatlas+aseg.deep.mgz)')
    parser.add_argument('--order', dest='order', type=int, default=1,
                        help="order of interpolation (0=nearest,1=linear(default),2=quadratic,3=cubic)")

    # 3. Options for log-file and search-tag
    parser.add_argument('--t', '--tag', dest='search_tag', default="*",
                        help='Search tag to process only certain subjects. If a single image should be analyzed, '
                             'set the tag with its id. Default: processes all.')
    parser.add_argument('--log', dest='logfile', help='name of log-file. Default: deep-seg.log',
                        default='deep-seg.log')

    # 4. Pre-trained weights
    parser.add_argument('--base_pretrained_path', help="path to pre-trained weights of base network",
                        default='./checkpoints/unensembled_model')
    parser.add_argument('--use_fv', action='store_true',
                        help="adds the full volume prediction on the base model. Works on both overlapped eval and"
                             "quadnet. Can improve results at the cost of using more memory. Default: False")
    parser.add_argument('--quadnet_path', help="path to the QuadNet experiment folder with quadnet weights.",
                        default='./checkpoints/ensembled_model')

    # 5. Options for model parameters setup (only change if model training was changed)
    parser.add_argument('--num_filters', type=int, default=64,
                        help='Filter dimensions for DenseNet (all layers same). Default=64')
    parser.add_argument('--num_classes', type=int, default=79,
                        help='Number of classes to predict in network, including background. Default=79')
    parser.add_argument('--num_channels', type=int, default=1,
                        help='Number of input channels in 3d volume. Default=1')
    parser.add_argument('--kernel_height', type=int, default=5, help='Height of Kernel (Default 5)')
    parser.add_argument('--kernel_width', type=int, default=5, help='Width of Kernel (Default 5)')
    parser.add_argument('--kernel_depth', type=int, default=5, help='Depth of Kernel (Default 5)')
    parser.add_argument('--stride', type=int, default=1, help="Stride during convolution (Default 1)")
    parser.add_argument('--stride_pool', type=int, default=2, help="Stride during pooling (Default 2)")
    parser.add_argument('--pool', type=int, default=2, help='Size of pooling filter (Default 2)')

    # 6. Clean up and GPU/CPU options (disable cuda, change batchsize)
    parser.add_argument('--clean', dest='cleanup', help="Flag to clean up segmentation", action='store_true')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for inference. Default: 8")
    parser.add_argument('--simple_run', action='store_true', default=False,
                        help='Simplified run: only analyse one given image specified by --in_name (output: --out_name). '
                             'Need to specify absolute path to both --in_name and --out_name if this option is chosen.')
    parser.add_argument('--cuda_device', type=str, default='cuda:0', dest='model_device')
    parser.add_argument('--aggregate_device', type=str, default='cuda_device', dest='agg_device')

    # 7. Options for full volume, overlapping and quadnet eval
    parser.add_argument('--eval_type', choices=['full', 'overlap', 'quadnet'], help="Flag to select evaluation type",
                        default='overlap')
    parser.add_argument('--overlap_side', type=int, help="The side length of the overlapped window for evaluation",
                        default=128)
    parser.add_argument('--overlap_stride', type=int, help="The stride of the overlapped window for evaluation",
                        default=64)

    sel_option = parser.parse_args()
    if sel_option.input is None and sel_option.csv_file is None and not sel_option.simple_run:
        parser.print_help(sys.stderr)
        sys.exit('----------------------------\nERROR: Please specify data directory or input volume\n')

    if sel_option.output is None and not sel_option.simple_run:
        parser.print_help(sys.stderr)
        sys.exit('----------------------------\nERROR: Please specify data output directory '
                 '(can be same as input directory)\n')

    if sel_option.agg_device == 'cuda_device':
        sel_option.agg_device = sel_option.model_device

    return sel_option


@torch.no_grad()
def eval_epoch_fv(model, volume, agg_device=None):
    if agg_device is None:
        agg_device = volume.device
    model.eval()
    # volume = volume.to(torch.device(options.model_device))
    out_vol = model(volume)
    return out_vol.to(agg_device)


@torch.no_grad()
def eval_epoch_patched(model, volume, p_len=128, p_step=32, inp_len=256, n_class=79, agg_device=None):
    if agg_device is None:
        agg_device = volume.device

    model.eval()

    patch_s = [p for p in range(0, inp_len - p_len + 1, p_step)]
    patch_list = list(product(patch_s, patch_s, patch_s))
    logger.info("Running overlapped evaluation for patch len = {} and patch step = {}".format(p_len, p_step))
    logger.info("Total patches: {}".format(len(patch_list)))

    out = torch.zeros((1, n_class, inp_len, inp_len, inp_len), device=agg_device)

    for idx, (x, y, z) in enumerate(patch_list):
        outs_patch = model(volume[..., x:x + p_len, y:y + p_len, z:z + p_len])
        outs_patch = torch.nn.functional.softmax(outs_patch, dim=1)
        out[..., x:x + p_len, y:y + p_len, z:z + p_len] += outs_patch.to(agg_device)  # .squeeze(0).cpu()

    out = torch.nn.functional.softmax(out, dim=1)
    return out


@torch.no_grad()
def eval_epoch_quadnet(model, volume, p_len=128, p_step=64, agg_device=None):
    inp_len = volume.shape[2]  # Assuming equal dimensions

    # n_patches = inp_len / ((p_len) if p_len >= p_step else p_len - p_step)

    patch_s = [p for p in range(0, inp_len - p_len + 1, p_step)]
    patch_list = list(product(patch_s, patch_s, patch_s))

    # net_coords is a dict of tuple(int, int, int) identifying the network and the assigned patches (from patch_list), also preserving index
    net_coords = assign_nets_to_coords(
        input_shape=(inp_len, inp_len, inp_len),
        patch_size=(p_len, p_len, p_len),
        coords_list=patch_list
    )
    if agg_device is None:
        agg_device = volume.device

    # print(net_coords)
    model.eval()

    logger.info("Running overlapped evaluation for patch len = {} and patch step = {}".format(p_len, p_step))
    logger.info("Total patches: {}".format(len(patch_list)))

    out = torch.zeros((1, options.num_classes, inp_len, inp_len, inp_len), device=agg_device)


    for idx, k in enumerate(net_coords.keys()):

        part_model_save_path = os.path.join(options.quadnet_path, f"ensembled_model_{idx + 1}")
        model.load_state_dict(torch.load(part_model_save_path))

        quad_data = net_coords[k]  # list of (index, patch_coords)
        model_coordinates = [i for _, i in quad_data]  # remove the index from quad_data

        # print(k)
        for x, y, z in model_coordinates:
            outs_patch = model(volume[..., x:x + p_len, y:y + p_len, z:z + p_len])
            outs_patch = torch.nn.functional.softmax(outs_patch, dim=1)
            out[..., x:x + p_len, y:y + p_len, z:z + p_len] += outs_patch.to(agg_device)  # .squeeze(0).cpu()
            del outs_patch

    out = torch.nn.functional.softmax(out, dim=1)
    return out


def run_network(model, img_filename, save_as, params):

    start_total = time.time()
    logger.info("Reading volume {}".format(img_filename))

    # TODO INSERT ASSERT FOR USING BOTH EVALUATION TYPE AS OVERLAP AND QUADNET TOGETHER.

    with torch.no_grad():

        try:
            # orig = nib.load(join(p, orig_post_fix))
            header_info, affine_info, orig_data = load_and_conform_image(img_filename, interpol=1,
                                                                         logger=logger)
            # Convert to Torch float Tensor and b,c,0,1,2 formatting
            orig_data = np.clip(orig_data / 255.0, a_min=0.0, a_max=1.0)
            orig_data = torch.from_numpy(orig_data).unsqueeze(0).unsqueeze(0).float()
            orig_data = orig_data.to(torch.device(options.model_device))

        except:
            logger.info("Could not load/conform volume {} data.".format(img_filename))
            return None
            # exit(1)

        pred_prob = torch.zeros((1, params["num_classes"]) + orig_data.shape[2:],
                                device=torch.device(options.agg_device))

        if options.eval_type == 'full' or options.use_fv:
            logger.info("Running full volume evaluation on base model at {}".format(options.base_pretrained_path))
            start = time.time()
            pred_prob += eval_epoch_fv(model, orig_data).to(torch.device(options.agg_device))
            logger.info("Full volume evaluation in {:0.4f} seconds".format(time.time() - start))

        if options.eval_type == 'overlap':
            logger.info("Running overlapped evaluation on base model at {}".format(options.base_pretrained_path))
            start = time.time()
            pred_prob += eval_epoch_patched(model, orig_data, p_len=options.overlap_side, n_class=params['num_classes'],
                                            p_step=options.overlap_stride, agg_device=torch.device(options.agg_device))
            logger.info("Overlapping evaluation in {:0.4f} seconds".format(time.time() - start))

        if options.eval_type == 'quadnet':
            logger.info("Running overlapping evaluation on QuadNet models...")
            start = time.time()
            pred_prob += eval_epoch_quadnet(model, orig_data, p_len=options.overlap_side,
                                            p_step=options.overlap_stride, agg_device=torch.device(options.agg_device))
            logger.info("Overlapping QuadNet evaluation in {:0.4f} seconds".format(time.time() - start))

        _, pred_prob = torch.max(pred_prob, 1)
        pred_prob = pred_prob.squeeze(0).squeeze(0)  # .cpu().numpy()
        lut = du.read_classes_from_lut(params["lut"])
        torch_labels = torch.from_numpy(lut["ID"].values)
        pred_prob = du.map_label2aparc_aseg(pred_prob, torch_labels)

        # Saving image
        header_info.set_data_dtype(np.int16)
        mapped_aseg_img = nib.MGHImage(pred_prob, affine_info, header_info)
        mapped_aseg_img.to_filename(save_as)
        logger.info("Saving Segmentation to {}".format(save_as))
        logger.info("Total processing time: {:0.4f} seconds.".format(time.time() - start_total))


def main(invol, save_file_name):
    params = {'in_channels': 1,
              'out_channels': 16,
              'total_conv_per_layer': 6,
              'kernel_size': (3, 3, 3),
              'input_shape': (256, 256, 256),
              'patch_size': (128, 128, 128),
              'num_classes': 34,
              'out': False,
              'input': True,
              'gen_random': False,
              'gpu_map': {},
              'num_blocks': 5,
              'sub_model_name': 'vnet',
              'training': False
              'lut': "/autofs/vast/lzgroup/Users/LeonieHenschel/FastInfantSurfer/configs/FastInfantSurfer_LUT.tsv"
              }

    model = RCVNet(params).to(torch.device(options.model_device))
    logger.info("Loading checkpoint {}".format(options.base_pretrained_path))
    model_state = torch.load(options.base_pretrained_path, map_location=options.model_device)
    model.load_state_dict(model_state["model_state"])

    run_network(model, invol, save_file_name, params)


def curl(url, target):
    import requests
    response = requests.get(url)
    with open(target, 'wb') as f:
        f.write(response.content)


if __name__ == "__main__":

    # Command Line options and error checking done here
    options = options_parse()

    # Set up the logger
    logger = logging.getLogger("eval")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))

    downloadurl = "https://b2share.fz-juelich.de/api/files/3c163ecc-1652-4ea5-ae48-1ee7e02d726f"

    if not os.path.exists(options.base_pretrained_path):
        basepath = os.path.dirname(options.base_pretrained_path)
        if not os.path.exists(basepath):
            makedirs(basepath)
        curl(downloadurl + "/unensembled_model", options.base_pretrained_path)

    if options.eval_type == "quadnet":
        if not os.path.exists(options.quadnet_path):
            makedirs(options)
        for i in range(8):
            target_file = f"/ensembled_model_{i + 1}"
            if not os.path.exists(options.quadnet_path + target_file):
                curl(downloadurl + target_file, options.quadnet_path + target_file)

    if options.simple_run:

        # Check if output subject directory exists and create it otherwise
        sub_dir, out = op.split(options.oname)

        if not op.exists(sub_dir):
            makedirs(sub_dir)

        main(options.iname, options.oname)

    else:

        # Prepare subject list to be processed
        if options.csv_file is not None:
            with open(options.csv_file, "r") as s_dirs:
                subject_directories = [line.strip() for line in s_dirs.readlines()]

        else:
            search_path = op.join(options.input, options.search_tag)
            subject_directories = glob.glob(search_path)

        # Report number of subjects to be processed and loop over them
        data_set_size = len(subject_directories)
        logger.info("Total Dataset Size is {}".format(data_set_size))

        for current_subject in tqdm(subject_directories):

            subject = current_subject.split("/")[-1]

            # Define volume to process, log-file and name under which final prediction will be saved
            if options.csv_file:

                dataset = current_subject.split("/")[-2]
                invol = op.join(current_subject, options.iname)
                logfile = op.join(options.output, dataset, subject, options.logfile)
                save_file_name = op.join(options.output, dataset, subject, options.oname)

            else:

                invol = op.join(current_subject, options.iname)
                logfile = op.join(options.output, subject, options.logfile)
                save_file_name = op.join(options.output, subject, options.oname)

            logger.info("Running Fast Surfer on {}".format(subject))

            # Check if output subject directory exists and create it otherwise
            sub_dir, out = op.split(save_file_name)

            if not op.exists(sub_dir):
                makedirs(sub_dir)

            # Prepare the log-file (logging to File in subject directory)
            fh = logging.FileHandler(logfile, mode='w')
            logger.addHandler(fh)

            # Run network
            main(invol, save_file_name)

            logger.removeHandler(fh)
            fh.close()

        sys.exit(0)
