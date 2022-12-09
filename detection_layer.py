import math
import torch
import torch.nn as nn
import time

'''计算两个bbox iou的函数'''
def compute_iou(truth, out):
    # intersection
    l1 = truth[0] - 0.5 * truth[2]
    l2 = out[0] - 0.5 * out[2]
    left = l1 if l1 > l2 else l2
    r1 = truth[0] + 0.5 * truth[2]
    r2 = out[0] + 0.5 * out[2]
    right = r1 if r1 < r2 else r2

    t1 = truth[1] - 0.5 * truth[3]
    t2 = out[1] - 0.5 * out[3]
    top = t1 if t1 > t2 else t2
    b1 = truth[1] + 0.5 * truth[3]
    b2 = out[1] + 0.5 * out[3]
    bottom = b1 if b1 < b2 else b2
    w = right - left
    h = bottom - top
    if w < 0 or h < 0:
        return 0
    box_intersection = w * h
    box_union = truth[2] * truth[3] + out[2] * out[3] - box_intersection
    return box_intersection / box_union

'''计算两个bbox之间距离的rmse'''
def compute_rmse(truth, out):
    return math.sqrt(torch.pow(truth[0] - out[0], 2) + torch.pow(truth[1] - out[1], 2) +
                     torch.pow(truth[2] - out[2], 2) + torch.pow(truth[3] - out[3], 2))

class Loss(nn.Module):
    # parameter init
    '''noobject_scale, object_scale, class_scale, coord_scale：不同损失的权重缩放'''
    def __init__(self, side, B, classes, noobject_scale, object_scale, class_scale, coord_scale, use_gpu):
        super(Loss, self).__init__()
        self.side = side
        self.B = B
        self.classes = classes
        self.noobject_scale = 0.5 * noobject_scale
        self.object_scale = 0.5 * object_scale
        self.class_scale = 0.5 * class_scale
        self.coord_scale = 0.5 * coord_scale
        self.use_gpu = use_gpu

    # detection layer forward
    # input: 8 * 1715 (980 + 147 + 588)
    # labels: 8 * 1225 (49 * 25)
    def forward(self, input, labels):
        curr_batch = input.shape[0]
        locations = self.side * self.side
        delta = torch.zeros(curr_batch, locations * (self.classes + 5 * self.B))
        for cnt_batch in range(curr_batch):   # [0,8)
            for cnt_locations in range(locations):   # [0,49)
                best_index, best_iou, best_rmse = -1, 0, 20
                if_obj = labels[cnt_batch, cnt_locations * (1 + self.classes + 4)]
                '''only conf delta'''
                delta[cnt_batch,
                locations * self.classes + cnt_locations * self.B :
                locations * self.classes + cnt_locations * self.B + self.B] = self.noobject_scale * \
                    torch.pow(0 - input[cnt_batch,
                                  locations * self.classes + cnt_locations * self.B :
                                  locations * self.classes + cnt_locations * self.B + self.B],
                              2)
                if not if_obj:  continue
                '''conf delta + classes delta + coor delta'''
                # classes delta
                delta[cnt_batch,
                 cnt_locations * self.classes : cnt_locations * self.classes + self.classes] = self.class_scale * \
                    torch.pow(labels[cnt_batch,
                 1 + cnt_locations * (self.classes + 5) : 1 + cnt_locations * (self.classes + 5) + self.classes] - input[cnt_batch,
                                  cnt_locations * self.classes : cnt_locations * self.classes + self.classes],
                              2)

                # choose best bbox
                truth = labels[cnt_batch,
                        1 + self.classes + cnt_locations * (1 + self.classes + 4):
                        1 + self.classes + cnt_locations * (1 + self.classes + 4) + 4].clone()
                # truth[0: 2] = truth[0: 2] / self.side
                truth[0: 2] = truth[0: 2] / self.side
                # compute iou and rmse
                for cnt_B in range(self.B):
                    out = input[cnt_batch,
                    locations * self.classes + locations * self.B + cnt_locations * self.B * 4 + cnt_B * 4 :
                    locations * self.classes + locations * self.B + cnt_locations * self.B * 4 + cnt_B * 4 + 4].detach().clone()
                    # out[0 : 2] = out[0 : 2] / self.side
                    out[0 : 2] = out[0 : 2] / self.side
                    out[2 : 4] = out[2 : 4] * out[2 : 4]
                    curr_iou = compute_iou(truth, out)
                    curr_rmse = compute_rmse(truth, out)
                    # update the best bbox index
                    if best_iou > 0 or curr_iou > 0:
                        if curr_iou > best_iou:
                            best_iou = curr_iou
                            best_index = cnt_B
                    else:
                        if curr_rmse < best_rmse:
                            best_rmse = curr_rmse
                            best_index = cnt_B
                # conf delta
                best_out = input[cnt_batch,
                    locations * self.classes + locations * self.B + cnt_locations * self.B * 4 + best_index * 4 :
                    locations * self.classes + locations * self.B + cnt_locations * self.B * 4 + best_index * 4 + 4].detach().clone()
                # best_out[0: 2] = best_out[0: 2] / self.side
                best_out[0: 2] = best_out[0: 2] / self.side
                best_out[2: 4] = best_out[2: 4] * best_out[2: 4]

                if self.use_gpu:
                    iou = torch.Tensor([compute_iou(truth, best_out)]).detach().cuda()
                else:
                    iou = torch.Tensor([compute_iou(truth, best_out)]).detach()
                delta[cnt_batch,
                locations * self.classes + cnt_locations * self.B + best_index] = self.object_scale * \
                    torch.pow(iou - input[cnt_batch,
                                  locations * self.classes + cnt_locations * self.B + best_index],
                              2)

                # coor delta
                delta[cnt_batch,   # x
                    locations * self.classes + locations * self.B + cnt_locations * self.B * 4 + best_index * 4] = self.coord_scale * \
                    torch.pow(labels[cnt_batch,
                    1 + self.classes + cnt_locations * (1 + self.classes + 4)] - \
                    input[cnt_batch,
                    locations * self.classes + locations * self.B + cnt_locations * self.B * 4 + best_index * 4], 2)
                delta[cnt_batch,  # y
                    locations * self.classes + locations * self.B + cnt_locations * self.B * 4 + best_index * 4 + 1] = self.coord_scale * \
                    torch.pow(labels[cnt_batch,
                    1 + self.classes + cnt_locations * (1 + self.classes + 4) + 1] - \
                    input[cnt_batch,
                    locations * self.classes + locations * self.B + cnt_locations * self.B * 4 + best_index * 4 + 1], 2)
                delta[cnt_batch,  # w
                    locations * self.classes + locations * self.B + cnt_locations * self.B * 4 + best_index * 4 + 2] = self.coord_scale * \
                    torch.pow(torch.sqrt(labels[cnt_batch,
                    1 + self.classes + cnt_locations * (1 + self.classes + 4) + 2]) - \
                    input[cnt_batch,
                    locations * self.classes + locations * self.B + cnt_locations * self.B * 4 + best_index * 4 + 2], 2)
                delta[cnt_batch,  # h
                    locations * self.classes + locations * self.B + cnt_locations * self.B * 4 + best_index * 4 + 3] = self.coord_scale * \
                    torch.pow(torch.sqrt(labels[cnt_batch,
                    1 + self.classes + cnt_locations * (1 + self.classes + 4) + 3]) - \
                    input[cnt_batch,
                    locations * self.classes + locations * self.B + cnt_locations * self.B * 4 + best_index * 4 + 3], 2)
        '''delta用于网络的反向传播'''
        return torch.nansum(delta)

'''program test'''
if __name__ == "__main__":
    my_loss = Loss(side = 7, B = 3, classes = 20,
                   noobject_scale = 0.5, object_scale = 1, class_scale = 1, coord_scale = 5, use_gpu = 0)
    input = torch.randn(8, 1715)
    input.requires_grad_(requires_grad = True)
    print(torch.sum(input))
    labels = torch.randn(8, 1225)
    start_time = time.time()
    loss = my_loss(input, labels)
    stop_time = time.time()
    loss.backward()
    print(stop_time - start_time)