'''导入相关模块'''
import torch
import torch.nn as nn
import numpy as np
import dataset    # 导入数据集
from torch.utils.data import DataLoader
import backbone    # 导入网络结构类
import detection_layer   # 导入损失函数
from d2l import torch as d2l
import yaml
import sys
import logging
import random
import time

model_cfg_dir = 'yolov1.yaml'
'''定义随机数种子'''
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    model_para = yaml.load(open(model_cfg_dir, 'r', encoding='utf-8'), yaml.FullLoader)
    net_cfg = model_para['net_cfg']        # network config
    '''随机数种子'''
    seed = model_para['seed']
    setup_seed(seed)
    # define devices
    use_gpu = torch.cuda.is_available()       # if cuda ?
    gpu_nums = model_para['gpu_nums']
    devices = [d2l.try_gpu(i) for i in range(gpu_nums)]
    # model parameters
    train_epoch, batch_size, sub_batch = model_para['train_epoch'], model_para['batch_size'], model_para['sub_batch']
    '''定义优化器参数'''
    momentum, learning_rate, decay = model_para['momentum'], model_para['learning_rate'], model_para['decay']
    '''W：图像高，H：图像宽'''
    W, H, C = model_para['W'], model_para['H'], model_para['C']
    '''数据增强'''
    jitter, exposure, saturation, hue = model_para['jitter'], model_para['exposure'], \
                                        model_para['saturation'], model_para['hue']
    '''定义学习率变化的步长和步幅'''
    steps, scales = model_para['steps'], model_para['scales']
    '''损失函数的相关参数'''
    side, num_Bbox, classes = model_para['side'], model_para['num_Bbox'], model_para['classes']
    noobject_scale, object_scale, class_scale, coord_scale = model_para['noobject_scale'], model_para['object_scale'], \
                                                             model_para['class_scale'], model_para['coord_scale']
    # models
    '''初始化模型并放入相应的设备'''
    my_yolo = backbone.yolo_v1(net_cfg)
    my_yolo = my_yolo.to(devices[0])
    # dataset and loader
    '''定义数据集和数据集加载方法'''
    train_data = dataset.yolo_v1_dataset(root='', filename='train.txt', H=H, W=W, jitter=jitter, exposure=exposure,
                                         saturation=saturation, hue=hue, side=side, classes=classes, category='train')
    train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size,
                                   shuffle=True, num_workers=0, drop_last=False)

    # optmizer and loss function
    '''定义SGD优化器'''
    opt = torch.optim.SGD(my_yolo.parameters(), lr=learning_rate / batch_size,
                          momentum=momentum, weight_decay=decay * batch_size)
    criterion = detection_layer.Loss(side=side, B=num_Bbox, classes=classes,
                                     noobject_scale=noobject_scale, object_scale=object_scale,
                                     class_scale=class_scale, coord_scale=coord_scale, use_gpu=use_gpu)
    # logger
    '''打印信息'''
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # learning_rate lambda
    '''定义变化学习率的函数'''
    def f(cnt_i):
        scale = 1
        for index, elem in enumerate(steps):
            if elem > cnt_i:
                return scale
            scale = scale * scales[index]
        return scale
    # learning_rate schedule
    '''模型学习率变化表，每个循环进入函数f更新新的学习率'''
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=opt, lr_lambda=f)
    # writer = SummaryWriter("dataloader")
    '''初始化网络损失'''
    step = 0
    ave_loss = -1
    for cnt_epoch in range(train_epoch):
        my_yolo.train()  # 声明模型训练
        load_time0 = time.time()
        '''从数据加载方法中得到一个batch的图像数据'''
        for imgs, labels in train_data_loader:
            load_time1 = time.time()
            module_time0 = time.time()
            step += 1
            curr_batch = imgs.shape[0]
            division = curr_batch // sub_batch
            imgs, labels = imgs.to(devices[0]), labels.to(devices[0])
            sum_loss = 0
            opt.zero_grad()     # 梯度清零
            '''将图像数据分batch_size // sub_batch次传入网络'''
            for cnt_sub_batch in range(division):
                # forward propagation
                output = my_yolo(imgs[sub_batch * cnt_sub_batch : sub_batch * (cnt_sub_batch + 1)])
                # computing model loss
                loss = criterion(output, labels[sub_batch * cnt_sub_batch : sub_batch * (cnt_sub_batch + 1)])
                # backward propagation
                '''反向传播，最终在权重上叠加一个batch的grad'''
                loss.backward()
                sum_loss += loss.item()
            # 处理drop_last = False
            if curr_batch % sub_batch != 0:
                # forward propagation
                output = my_yolo(imgs[division * sub_batch:])
                # computing model loss
                loss = criterion(output, labels[division * sub_batch:])
                # backward propagation
                '''反向传播，最终在权重上叠加一个batch的grad'''
                loss.backward()
                sum_loss += loss.item()
            if use_gpu:
                torch.cuda.empty_cache()
            '''优化器步进'''
            opt.step()
            '''学习率步进'''
            scheduler.step()
            sum_loss /= curr_batch
            if ave_loss < 0:
                ave_loss = sum_loss
            '''当前batch的loss'''
            ave_loss = ave_loss * 0.9 + sum_loss * 0.1
            module_time1 = time.time()
            '''print training information'''
            logging.info('{0}: {1:.6f}, {2:.6f} ave, {3:6f} rate, {4:.6f} seconds(module), {5:.6f} seconds(load), {6} images'.format(
                step, sum_loss, ave_loss, opt.state_dict()['param_groups'][0]['lr'] * batch_size, module_time1 - module_time0,
                load_time1 - load_time0, step * curr_batch))
            # save model weights
            if (step % 5000 == 0):
                torch.save(my_yolo.state_dict(), 'models/yolo_v1_' + str(step) + '.pth')
            load_time0 = time.time()
    '''保存最终权重'''
    torch.save(my_yolo.state_dict(), 'models/yolo_final.pth')
