'''darknet: yolo_v1_backbone'''
import torch
import torch.nn as nn
from local_layer_new import LocallyConnected2d        # import local_layer
import math
import time
import yaml

'''conv -> BN -> activation'''
class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(CBL, self).__init__()
        self.activation=nn.LeakyReLU(negative_slope=0.1)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=kernel_size // 2, bias=False, dtype=torch.float32)
        self.BN = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, dtype=torch.float32)
    '''forward propagation'''
    def forward(self, input):
        output = self.conv(input)
        output = self.BN(output)
        output = self.activation(output)
        return output

'''1*1 conv -> 3*3 conv'''
class one2three(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(one2three, self).__init__()
        self.layer = nn.Sequential(
            CBL(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            CBL(in_channels=out_channels, out_channels=in_channels, kernel_size=3, stride=1),
        )
    def forward(self, input):
        output = self.layer(input)
        return output

'''def backbone of yolov1'''
class yolo_v1(nn.Module):
    def __init__(self, net_cfg):
        super(yolo_v1, self).__init__()
        layer_curr = []
        for item in net_cfg:
            if item[0] == 'CBL':
                layer_curr = self.make_CBL(layer_curr, item)
            elif item[0] == 'maxpool':
                layer_curr = self.make_maxpool(layer_curr, item)
            elif item[0] == 'one2three':
                layer_curr = self.make_one2three(layer_curr, item)
            elif item[0] == 'local':
                layer_curr = self.make_local(layer_curr, item)
            elif item[0] == 'dropout':
                layer_curr = self.make_dropout(layer_curr, item)
            elif item[0] == 'Flatten':
                layer_curr.append(nn.Flatten())
            elif item[0] == 'linear':
                layer_curr = self.make_linear(layer_curr, item)
        self.layers = nn.Sequential(*layer_curr)

        '''weight initial'''
        for m in self.modules():
            # is nn.Conv2d() instance ?
            if isinstance(m, nn.Conv2d):
                scale = math.sqrt(2 / (m.kernel_size[0] * m.kernel_size[1] * m.in_channels))
                m.weight.data = scale * m.weight.data.normal_(0, 1)
            # is nn.Linear() instance ?
            if isinstance(m, nn.Linear):
                scale = math.sqrt(2 / m.in_features)
                m.weight.data = scale * m.weight.data.uniform_(-1, 1)
                m.bias.data = torch.full_like(m.bias.data, 0.)

    def make_CBL(self, layers, CBL_cfg):
        for item in CBL_cfg[1:]:
            layers.append(CBL(in_channels=item[0], out_channels=item[1], kernel_size=item[2], stride=item[3]))
        return layers

    def make_one2three(self, layers, one2three_cfg):
        for i in range(one2three_cfg[-1][-1]):
            layers.append(one2three(in_channels=one2three_cfg[-1][0], out_channels=one2three_cfg[-1][1]))
        return layers

    def make_maxpool(self, layers, maxpool_cfg):
        layers.append(nn.MaxPool2d(kernel_size=maxpool_cfg[-1][0], stride=maxpool_cfg[-1][1]))
        return layers

    def make_local(self, layers, local_cfg):
        layers.append(LocallyConnected2d(in_channels=local_cfg[-1][0], out_channels=local_cfg[-1][1],
                                         output_size=local_cfg[-1][2], kernel_size=local_cfg[-1][3],
                                         stride=local_cfg[-1][4], padding=local_cfg[-1][5], bias=local_cfg[-1][6]))
        return layers

    def make_dropout(self, layers, dropout_cfg):
        layers.append(nn.Dropout(p=dropout_cfg[-1][0]))
        return layers

    def make_linear(self, layers, linear_cfg):
        layers.append(nn.Linear(in_features=linear_cfg[-1][0], out_features=linear_cfg[-1][1], dtype=torch.float32))
        return layers

    # rewrite forward propagation
    def forward(self, input):
        '''apply forward propagation to each layer'''
        output = self.layers(input)
        return output

'''backbone test case'''
if __name__ == "__main__":
    model_para = yaml.load(open('yolov1.yaml', 'r', encoding='utf-8'), yaml.FullLoader)
    my_yolo = yolo_v1(model_para['net_cfg'])     # make yolov1 network instance
    # print(my_yolo)
    input = torch.randn((2, 3, 448, 448), requires_grad=True)       # def network input: (B, C, H, W) format
    start_time = time.time()     # timing
    output = my_yolo(input).sum()     # forward propagation
    stop_time_for = time.time()
    output.backward()       # backward propagation
    stop_time_back = time.time()
    '''打印网络定时信息'''
    print('forward_time: {0:.6f}\nbackward_time: {1:.6f}'.format(stop_time_for - start_time,
                                                                 stop_time_back - stop_time_for))
    # # visualization
    # torch.onnx.export(
    #     my_yolo,
    #     input,
    #     'model.onnx',
    #     export_params=True,
    #     opset_version=8,
    #     training=torch.onnx.TrainingMode.TRAINING
    # )
