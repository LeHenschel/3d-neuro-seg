import torch
import torch.nn as nn


def make_rand_coords(input_size=(256, 256, 256), patch_size=(64, 64, 64)):
    return [get_dims(input_size[0] - patch_size[0]), \
            get_dims(input_size[1] - patch_size[1]), \
            get_dims(input_size[2] - patch_size[2])]


def get_dims(upper):
    # Random value in the range [0, upper)
    return int(upper * torch.rand(1))


def multi_gpu_check(gpu_map, l_name, *args):
    """
    Can move computations to other GPUs if specified. The names of the layers and corresponding GPUs can be specified
    in GPU map.

    :param gpu_map:
    :param l_name:
    :param args:
    :return:
    """
    args = list(args)
    if l_name in gpu_map.keys():
        # print(l_name)
        for idx, l in enumerate(args):
            args[idx] = l.to(torch.device(gpu_map[l_name]))
            # print(args[idx].device, gpu_map[l_name])

    if len(args) == 1:
        return args[0]
    return args


class RCVNet(nn.Module):
    """
    Random Cropping VNet. Model is designed to extract patches randomly during feedforward pass unless specifically
    prevented by setting a random patch coordinate manually. Can also move operations for individual layers to different
    GPUs if specified in params
    
    Standard VNet Architecture
    """

    def __init__(self, params):
        """
        Standard VNet Architecture
        """
        super(RCVNet, self).__init__()

        self.coords = None
        self.input_shape = params['input_shape']
        self.patch_size = params['patch_size']
        self.gen_random = params['gen_random']
        self.num_blocks = params["num_blocks"] - 1

        # Choose sub model
        if params['sub_model_name'] == 'vnet':
            from ThreeDNeuroSeg.model.VNet import EncoderBlock, BottleNeck, DecoderBlock
        elif params['sub_model_name'] == 'vnet_2d_3d':
            from .VNet_2D_3D import EncoderBlock, BottleNeck, DecoderBlock
        elif params['sub_model_name'] == 'vnet_asym':
            from .VNetAsym import EncoderBlock, BottleNeck, DecoderBlock
        elif params['sub_model_name'] == 'vnet_sym':
            from .VNetSym import EncoderBlock, BottleNeck, DecoderBlock
        elif params['sub_model_name'] == 'vnet_denseadd':
            from .VNetDenseAdd import EncoderBlock, BottleNeck, DecoderBlock
        elif params['sub_model_name'] == 'vnet_exclusion':
            from .VNetExclusion import EncoderBlock, BottleNeck, DecoderBlock
        elif params['sub_model_name'] == 'vnet_se':
            from .VNetSE import EncoderBlock, BottleNeck, DecoderBlock
        else:
            raise ValueError(f"{params['sub_model_name']} does not exist.")

        # Start model creation
        # in_channels: 16, out_channels: 16
        # Parameters for the Descending Arm
        self.encoderbase = nn.ModuleList()
        increment = params["total_conv_per_layer"] // 3
        for i in range(params["num_blocks"] - 1):
            params['conv_per_layer'] = min(increment + i * increment, params["total_conv_per_layer"])
            self.encoderbase.append(EncoderBlock(params))  # 1, 16
            params['input'] = False
            params['in_channels'] = params['out_channels'] * 2  # 32, 64, 128, 256
            params['out_channels'] = params['out_channels'] * 2  # 32, 64, 128, 256

        self.bottleneck_block = BottleNeck(params)

        # Parameters for Ascending Arm
        self.decoderbase = nn.ModuleList()
        for i in range(params["num_blocks"] - 1):
            # 256 + 128 = 384, 128 + 64 = 192, 64 + 32 = 96
            params['in_channels'] = params['out_channels'] + int(params['out_channels'] / 2)
            params['out'] = (params["num_blocks"] - 2) == i
            self.decoderbase.append(DecoderBlock(params))
            params['out_channels'] = int(params['out_channels'] / 2)  # 128, 64, 32
            params['conv_per_layer'] = max(params['total_conv_per_layer'] - increment * i, increment)

        # Logits
        params['out'] = False
        self.output_block = nn.Conv3d(in_channels=params['out_channels'] * 2, out_channels=params['num_classes'],
                                      kernel_size=(1, 1, 1), stride=1, padding=0)

        self.gpu_map = params['gpu_map']

    def set_coords(self):
        # Generate Random Coordinates if needed
        if self.gen_random:  # For usage by QuadNet
            self.coords = make_rand_coords(self.input_shape, self.patch_size)

    def crop_vol_to_patch(self, img):
        assert self.coords is not None
        return img[..., self.coords[0]:self.coords[0] + self.patch_size[0],
               self.coords[1]:self.coords[1] + self.patch_size[1],
               self.coords[2]:self.coords[2] + self.patch_size[2]]

    def assure_non_empty(self, img, eps=128):
        crop = self.crop_vol_to_patch(img)
        while torch.sum(crop) < eps:
            self.set_coords()
            self.crop_vol_to_patch(img)
        return crop

    def forward(self, x, sf=None, affine=None):
        """
        Standard VNet Architecture
        """
        if self.training:
            self.set_coords()
            x = self.crop_vol_to_patch(x)

        # Running Encoder side of network
        encode, skip, decode = [x], [], []
        for i in range(self.num_blocks):
            input = multi_gpu_check(self.gpu_map, 'encoder_block_' + str(i + 1), encode[i])
            skip_encoder, encoder_output = self.encoderbase[i](input)
            encode.append(encoder_output)
            skip.append(skip_encoder)

        # Running bottleneck
        input = multi_gpu_check(self.gpu_map, 'bottleneck_block', encode[-1])
        decode.append(self.bottleneck_block(input))

        # Run decoder
        for i in range(self.num_blocks):
            skip_in, decode_in = multi_gpu_check(self.gpu_map, 'decoder_block_' + str(self.num_blocks - 1),
                                                 skip[-i - 1], decode[i])
            decode.append(self.decoderbase[i](skip_in, decode_in))

        # Run logits
        decode_in = multi_gpu_check(self.gpu_map, 'output_block', decode[-1])
        out = self.output_block(decode_in)

        return out


class RCVNetAttention(RCVNet):

    def __init__(self, params):
        super(RCVNet, self).__init__()

        from model.VNetAttention import EncoderBlock as AttEncoderBlock, BottleNeck as AttBottleNeck, \
            DecoderBlock as AttDecoderBlock

        self.coords = None
        self.input_shape = params['input_shape']
        self.patch_size = params['patch_size']
        self.gen_random = params['gen_random']

        self.down_input_lower = nn.Sequential(
            nn.Conv3d(in_channels=params['in_channels'], out_channels=4 * params['out_channels'],
                      kernel_size=(4, 4, 4), padding=0, stride=4),
            nn.GroupNorm(num_groups=4, num_channels=4 * params['out_channels']),
            nn.PReLU()
        )
        # in_channels: 16, out_channels: 16
        self.encoder_block_1 = AttEncoderBlock(params)

        params['input'] = False
        params['create_layer_1'] = True
        params['in_channels'] = params['out_channels'] * 2  # 32
        params['out_channels'] = params['out_channels'] * 2  # 32
        self.encoder_block_2 = AttEncoderBlock(params)

        params['create_layer_2'] = True
        params['in_channels'] = params['out_channels'] * 2  # 64
        params['out_channels'] = params['out_channels'] * 2  # 64
        self.encoder_block_3 = AttEncoderBlock(params)

        params['in_channels'] = params['out_channels'] * 2  # 128
        params['out_channels'] = params['out_channels'] * 2  # 128
        self.encoder_block_4 = AttEncoderBlock(params)

        params['in_channels'] = params['out_channels'] * 2  # 256
        params['out_channels'] = int(params['out_channels'] * 2)  # 256
        self.bottleneck_block = AttBottleNeck(params)

        enc_channels = 128
        params['in_channels'] = params['out_channels'] + enc_channels  # 256 + 128
        params['F_g'], params['F_l'], params['F_int'] = (256, 128, 128)
        params['out_channels'] = params['out_channels']  # 256
        self.decoder_block_4 = AttDecoderBlock(params)

        enc_channels = int(enc_channels / 2)
        params['in_channels'] = int(params['out_channels'] / 2) + enc_channels  # 128 + 64
        params['out_channels'] = int(params['out_channels'] / 2)  # 128
        params['F_g'], params['F_l'], params['F_int'] = (128, 64, 64)
        self.decoder_block_3 = AttDecoderBlock(params)

        enc_channels = int(enc_channels / 2)
        params['in_channels'] = int(params['out_channels'] / 2) + enc_channels  # 64 + 32
        params['out_channels'] = int(params['out_channels'] / 2)  # 64
        params['F_g'], params['F_l'], params['F_int'] = (64, 32, 32)
        params['create_layer_2'] = False
        self.decoder_block_2 = AttDecoderBlock(params)

        enc_channels = int(enc_channels / 2)
        params['in_channels'] = int(params['out_channels'] / 2) + enc_channels  # 32 + 16
        params['out_channels'] = int(params['out_channels'] / 2)  # 32
        params['F_g'], params['F_l'], params['F_int'] = (32, 16, 16)
        params['create_layer_1'] = False
        params['out'] = True
        self.decoder_block_1 = AttDecoderBlock(params)
        params['out'] = False

        self.output_block = nn.Conv3d(in_channels=params['out_channels'], out_channels=params['num_classes'],
                                      kernel_size=(1, 1, 1), stride=1, padding=0)

        self.gpu_map = params['gpu_map']


if __name__ == "__main__":
    # TEST CODE [RUN THIS TO VERIFY MODELS]
    params = {'in_channels': 1,
              'out_channels': 16,
              'total_conv_per_layer': 6,
              'kernel_size': (3, 3, 3),
              'input_shape': (64, 64, 64),
              'patch_size': (64, 64, 64),
              'num_classes': 34,
              'out': False,
              'input': True,
              # 'F_g': None,
              # 'F_l': None,
              # 'F_int': None
              'gen_random': True,
              'gpu_map': {},
              'num_blocks': 5,
              'sub_model_name': "vnet",
              'training': True
              }

    m = RCVNet(params=params).cuda()
    # m.eval()
    # m = CompetitiveEncoderBlockInput(params=params).cuda()
    try:
        from torchsummary import summary

        # print([l for l in m.named_children()])
        summary(m, input_size=(1, 64, 64, 64))
    except ImportError:
        pass
    #
    # print([l for l in m.decoder_block_1.parameters()])
    # print([l.device() for _, l in m.named_children()])
