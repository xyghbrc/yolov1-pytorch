import torch
import torch.nn as nn
import cv2
from backbone import yolo_v1
import numpy as np
from dataset import yolo_v1_dataset
from torch.utils.data import DataLoader
from d2l import torch as d2l
import yaml
# from detection_layer import compute_iou
import time
import os

model_cfg_dir = 'yolov1.yaml'

'''color to get'''
colors = torch.Tensor([[255,0,255], [0,0,255], [0,255,255], [0,255,0], [255,255,0], [255,0,0]])

'''定义绘制检测框颜色的函数'''
def get_color(c, x, max):
    ratio = (x / max) * 5
    i = torch.floor(ratio).long()
    j = torch.ceil(ratio).long()
    ratio = ratio - i
    r = (1 - ratio) * colors[i, c] + ratio * colors[j, c]
    return r

'''计算两个bbox iou的函数'''
def compute_iou(truth, out):
    # intersection
    l1 = truth[:, 0] - 0.5 * truth[:, 2]
    l2 = out[:, 0] - 0.5 * out[:, 2]
    left = torch.where(l1 > l2, l1, l2)
    r1 = truth[:, 0] + 0.5 * truth[:, 2]
    r2 = out[:, 0] + 0.5 * out[:, 2]
    right = torch.where(r1 < r2, r1, r2)

    t1 = truth[:, 1] - 0.5 * truth[:, 3]
    t2 = out[:, 1] - 0.5 * out[:, 3]
    top = torch.where(t1 > t2, t1, t2)
    b1 = truth[:, 1] + 0.5 * truth[:, 3]
    b2 = out[:, 1] + 0.5 * out[:, 3]
    bottom = torch.where(b1 < b2, b1, b2)
    w = right - left
    h = bottom - top
    box_intersection = w * h
    box_union = truth[:, 2] * truth[:, 3] + out[:, 2] * out[:, 3] - box_intersection
    iou = box_intersection / box_union
    iou = torch.where((w < 0) | (h < 0), 0, iou)
    return iou

def decoder(predictions, h, w, pr_thresh, classes, side, B, devices):
    # define output
    prob = torch.zeros((classes, side * side * B), dtype=torch.float32, device=devices[0])
    confi_coor = torch.zeros((1 + 4, side * side * B), dtype=torch.float32, device=devices[0])
    # generate indices
    side_index = torch.arange(side * side, device=devices[0])
    row = torch.div(torch.repeat_interleave(side_index, B), side, rounding_mode='trunc')
    col = torch.fmod(torch.repeat_interleave(side_index, B), side)
    index = torch.arange(side * side * B, device=devices[0])
    p_index = index + side * side * classes
    box_index = side * side * (classes + B) + index * 4
    confi_coor[0, :] = predictions[:, p_index]  # confidence
    confi_coor[1, :] = (predictions[:, box_index] + col) / side * w  # x
    confi_coor[2, :] = (predictions[:, box_index + 1] + row) / side * h  # y
    confi_coor[3, :] = torch.pow(predictions[:, box_index + 2], 2) * w # w
    confi_coor[4, :] = torch.pow(predictions[:, box_index + 3], 2) * h # h
    class_index = torch.repeat_interleave(side_index * classes, B)
    for cnt_cla in range(classes):
        '''置信度乘以类别概率作为检测框最终置信度'''
        prob[cnt_cla, :] = confi_coor[0, :] * predictions[:, class_index + cnt_cla]
    prob = torch.where(prob > pr_thresh, prob, 0)
    return prob, confi_coor

def do_nms_sort(box_prob, confi_coor, total, classes, iou_thresh, devices):
    val, ind = torch.sort(box_prob[:, 0:total], dim=1, descending=True)   # get indices reflect to confi_coor
    '''do nms'''
    # generate idx
    k = torch.repeat_interleave(torch.arange(classes), total)
    i = torch.arange(total).repeat(classes)
    j = i + 1
    exist_val = torch.where((val[k, i] != 0) & (j < total))
    k_sel, i_sel, j_sel = k[exist_val], i[exist_val], j[exist_val]
    # get pairs of boxes
    box_curr = confi_coor[1: 5, ind[k_sel, i_sel]].transpose(1, 0)
    box_compare = confi_coor[1 : 5, ind[k_sel, j_sel]].transpose(1, 0)
    zeros_idx = torch.where(compute_iou(box_curr, box_compare) > iou_thresh)
    box_prob[k_sel[zeros_idx], ind[k_sel, j_sel][zeros_idx]] = 0
    return box_prob

if __name__ == "__main__":
    model_para = yaml.load(open(model_cfg_dir, 'r', encoding='utf-8'), yaml.FullLoader)
    # read labels
    VOC_NAMES = model_para['VOC_NAMES']
    '''detection model parameters'''
    gpu_nums = model_para['gpu_nums']  # gpu数量
    batch_size = model_para['batch_size_predict']
    '''H：图像高，W：图像宽，classes：类别数，side：grid cell个数，B：每个cell对应的bbox数'''
    net_cfg = model_para['net_cfg']
    classes, side, B = model_para['classes'], model_para['side'], model_para['num_Bbox']
    pr_thresh, iou_thresh, filter_thresh = model_para['pr_thresh'], model_para['iou_thresh'], model_para['filter_thresh']
    # define model and to devices
    devices = [d2l.try_gpu(i) for i in range(gpu_nums)]
    my_yolo_v1 = yolo_v1(net_cfg)
    my_yolo_v1 = my_yolo_v1.to(devices[0])
    # my_yolo_v1 = nn.DataParallel(module=my_yolo_v1, device_ids=devices)
    # loading weights
    my_yolo_v1.load_state_dict(torch.load('models/yolo_v1_40000.pth', map_location=devices[0]))
    dataset = yolo_v1_dataset('', 'train.txt', category='test')
    img_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=False)

    my_yolo_v1.eval()
    for img_bgr, img_rgb, img_name in img_loader:
        img_bgr = np.array(img_bgr[0])
        oh, ow, oc = img_bgr.shape
        # model forward
        start_time = time.time()
        output = my_yolo_v1(img_rgb)
        prob, confi_coor = decoder(predictions=output, h=oh, w=ow, pr_thresh=pr_thresh, classes=classes, side=side, B=B, devices=devices)
        '''Do NMS'''
        prob = do_nms_sort(prob, confi_coor, side * side * B, classes, iou_thresh, devices)
        '''write and save/show picture'''
        filter_idx = torch.where(prob > filter_thresh)
        x, y, w, h = confi_coor[1, filter_idx[1]], confi_coor[2, filter_idx[1]], confi_coor[3, filter_idx[1]], confi_coor[4, filter_idx[1]]
        x1, y1, x2, y2 = (x - w / 2).int(), (y - h / 2).int(), (x + w / 2).int(), (y + h / 2).int()
        x1, y1, x2, y2 = x1.clamp(0, ow - 1), y1.clamp(0, oh - 1), x2.clamp(0, ow - 1), y2.clamp(0, oh - 1)
        '''draw rectangle and label'''
        # define color
        offset = filter_idx[0] * 123457 % classes
        red, green, blue = get_color(2, offset, classes).int(), get_color(1, offset, classes).int(), get_color(0, offset, classes).int()
        for cnt, _ in enumerate(x):
            draw_color = [blue[cnt].item(), green[cnt].item(), red[cnt].item()]
            draw_x1, draw_y1, draw_x2, draw_y2 = x1[cnt].item(), y1[cnt].item(), x2[cnt].item(), y2[cnt].item()
            cv2.rectangle(img_bgr, (draw_x1, draw_y1), (draw_x2, draw_y2), draw_color, 2)
            name_prob = VOC_NAMES[filter_idx[0][cnt]] + ' ' + str(round(prob[filter_idx[0][cnt], filter_idx[1][cnt]].item(), 2))
            delta_x = len(name_prob) * 10
            delta_y = 17
            bounding = np.array([[draw_x1, draw_y1 - delta_y], [draw_x1 + delta_x, draw_y1 - delta_y], [draw_x1 + delta_x, draw_y1], [draw_x1, draw_y1]])
            cv2.fillPoly(img_bgr, [bounding], draw_color)
            cv2.putText(img_bgr, name_prob, (draw_x1 + 5, draw_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 0], 2)
        print("\r", 'Predicted in: {0:.6f} s '.format(time.time() - start_time), end = "", flush=True)
        # show image
        cv2.imshow('w1', img_bgr)
        cv2.waitKey()
        # # write into dir
        # cv2.imwrite(os.path.join('test_result', os.path.split(img_name[0])[-1]), img_bgr)