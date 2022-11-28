import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import math
import time
import numpy as np

'''https://discuss.pytorch.org/t/locally-connected-layers/26979'''
class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, padding, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        scale = math.sqrt(2 / (kernel_size * kernel_size * in_channels))
        weight = torch.zeros(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size ** 2)
        '''weight initial'''
        weight[:, :, :, 0, 0, :] = (2 * torch.rand(1, out_channels, in_channels, kernel_size ** 2) - 1) * scale
        # weight[:, :, :, 0, 0, :] = torch.full_like(weight[:, :, :, 0, 0, :], 0.0001)
        '''the weight and bias are encapsulated with nn.Parameter()'''
        self.weight = nn.Parameter(
            weight
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = padding
        self.activation = nn.LeakyReLU(np.float32(0.1))

    '''forward propagation of local_layer'''
    def forward(self, x):
        m = nn.ZeroPad2d(self.padding)
        x = m(x)     # padding
        _, c, h, w = x.size()          # shape of input
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        out = self.activation(out)
        return out

'''local_layer test case'''
if __name__ == "__main__":
    batch_size = 2
    in_channels = 2
    h, w = 3, 3
    x = torch.randn(batch_size, in_channels, h, w, requires_grad=True)   # generating network input

    out_channels = 3
    output_size = 2
    kernel_size = 2
    stride = 1
    padding = 0
    conv = LocallyConnected2d(
        in_channels, out_channels, output_size, kernel_size, stride, padding, bias=True)   # local_layer instance

    start_time = time.time()    # timing
    out = conv(x)     # forward propagation
    stop_time = time.time()
    print(stop_time - start_time)

    out.sum().backward()    # backward propagation
    print(time.time() - stop_time)