from .skip import skip
from .skip_ import skip_
from .texture_nets import get_texture_nets
from .resnet import ResNet
from .unet import UNet

import torch.nn as nn

def get_net(input_depth, NET_TYPE, pad, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
    if NET_TYPE == 'ResNet':
        # TODO
        net = ResNet(input_depth, 3, 10, 16, 1, nn.BatchNorm2d, False)
    elif NET_TYPE in ['skip','skip_']:
        if NET_TYPE == 'skip':
            get_arch = skip
        else NET_TYPE == 'skip_':
            get_arch = skip_
        net = get_arch(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

    elif NET_TYPE == 'texture_nets':
        net = get_texture_nets(inp=input_depth, ratios = [32, 16, 8, 4, 2, 1], fill_noise=False,pad=pad)

    elif NET_TYPE =='UNet':
        net = UNet(num_input_channels=input_depth, num_output_channels=3, 
                   feature_scale=4, more_layers=0, concat_x=False,
                   upsample_mode=upsample_mode, pad=pad, norm_layer=nn.BatchNorm2d, need_sigmoid=True, need_bias=True)
    elif NET_TYPE == 'identity':
        assert input_depth == 3
        net = nn.Sequential()
    else:
        assert False

    return net

def multi_sequential_parameters(seq_list):
    params = []
    for s in seq_list:
        params.extend(list(s.parameters()))
    return params

def parameters_till(net,scale,end=False):
    d = multi_sequential_parameters(net.opwise_modules['down'])
    up_coarse_to_fine = reversed(net.opwise_modules['up'])
    skip_coarse_to_fine = reversed(net.opwise_modules['skip'])
    u = multi_sequential_parameters(net.opwise_modules['up'][:scale+1])
    s = multi_sequential_parameters(net.opwise_modules['skip'][:scale+1])
    e = []
    if end:
        e = multi_sequential_parameters(net.opwise_modules['end'])
    p = d + u + s + e
    return p