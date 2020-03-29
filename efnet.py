import torch
import torch.nn.functional as F
from utils import conv_params, linear_params, bnparams, bnstats, \
    flatten_params, flatten_stats


def efnet(depth, width, num_classes, num_channels=1):
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = 1
    # widths = torch.Tensor([16, 32, 64]).mul(width).int()

    def gen_block_params():
        return {
            'conv0': conv_params(32, 32*2, 4),
            'conv1': conv_params(32*2, 32*4, 4),
            'conv2': conv_params(32*4, 10, 4),
            'bn0': bnparams(32*2),
            'bn1': bnparams(32*4),
        }

    def gen_group_params():
        return {'block0': gen_block_params()}

    def gen_group_stats():
        return {'block0': {'bn0': bnstats(32*2),
                           'bn1': bnstats(32*4)}}

    params = {
        'conv0': conv_params(num_channels, 32, 3),
        'group0': gen_group_params(),
    }

    stats = {
        'group0': gen_group_stats(),
    }

    flat_params = flatten_params(params)
    flat_stats = flatten_stats(stats)

    def activation(x, params, stats, base, mode):
        return F.relu(F.batch_norm(x, weight=params[base + '.weight'],
                                   bias=params[base + '.bias'],
                                   running_mean=stats[base + '.running_mean'],
                                   running_var=stats[base + '.running_var'],
                                   training=mode, momentum=0.1, eps=1e-5), inplace=True)

    def block(x, params, stats, base, mode, stride):
        #o1 = activation(x, params, stats, base + '.bn0', mode)
        y1 = F.conv2d(x, params[base + '.conv0'], stride=stride, padding=1)
        o1 = activation(y1, params, stats, base + '.bn0', mode)
        o2d = F.dropout(o1, p=0.3, training=mode)
        y2 = F.conv2d(o2d, params[base + '.conv1'], stride=stride, padding=1)
        o2 = activation(y2, params, stats, base + '.bn1', mode)
        o3d = F.dropout(o2, p=0.3, training=mode)
        y3 = F.conv2d(o3d, params[base + '.conv2'], stride=1, padding=0)
        return y3


    def group(o, params, stats, base, mode, stride):
        for i in range(n):
            o = block(o, params, stats, '%s.block%d' % (base, i), mode, stride if i == 0 else 1)
        return o

    def f(input, params, stats, mode):
        assert input.get_device() == params['conv0'].get_device()
        #print(input.size(),)
        x = F.conv2d(input, params['conv0'], stride=2, padding=1)
        #print(x.size(),)
        o = group(x, params, stats, 'group0', mode, 2)
        o = o.view(o.size(0), -1)
        #print(o.size())
        return o

    return f, flat_params, flat_stats
