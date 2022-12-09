'''导入构建数据集的相关模块'''
import torch
import torch.utils.data as data
import cv2
import os
import random
import numpy as np
import time

class yolo_v1_dataset(data.Dataset):
    '''root：训练或检测文件的根目录，filename：文件名称，H：图像高，W：图像宽，jitter：图像裁切的随机抖动'''
    '''exposure, saturation, hue：图像曝光度，饱和度，色调'''
    def __init__(self, root, filename, H=448, W=448, jitter=0.1, exposure=1.5, saturation=1.5, hue=0.1, side=7,
                 classes=20, category='train'):
        self.H = H
        self.W = W
        self.jitter = jitter
        self.exposure = exposure
        self.saturation = saturation
        self.hue = hue
        self.root = root
        self.side = side
        self.classes = classes
        self.category = category
        list_file = os.path.join(root, filename)
        with open(list_file) as f:
            self.read_lines  = f.readlines()
        self.img_names = []
        self.txt_names = []
        for line in self.read_lines:
            temp = line.strip()
            self.img_names.append(temp)
            self.txt_names.append(temp.replace('images', 'labels').replace('JPEGImages', 'labels')
                             .replace('jpg', 'txt').replace('JPG', 'txt').replace('JPEG', 'txt'))

    def __getitem__(self, index):
        img_bgr = cv2.imread(self.img_names[index])
        # bgr2rgb
        '''因为网络输入的是rgb格式，所以要将图像格式转换过来'''
        img_rgb = self.BGR2RGB(img_bgr)
        # convert pixel value to [0, 1]
        img_rgb = np.float32(img_rgb / 255.)
        # img_rgb = torch.Tensor(img_rgb)
        oh, ow, _ = img_rgb.shape
        if self.category == 'test':
            # reshape to: (C, H, W)
            img_rgb = self.resize_image(img_rgb, oh, ow).transpose(2, 0, 1)
            return img_bgr, torch.Tensor(img_rgb), self.img_names[index]
        elif self.category == 'train':
            '''从txt中读入图像名称和对应的labels标签'''
            with open(self.txt_names[index]) as f_txt:
                ground_truth = f_txt.readlines()
            boxes = []
            for line in ground_truth:
                split_line = line.strip().split()
                c = int(split_line[0])
                x = np.float32(split_line[1])
                y = np.float32(split_line[2])
                w = np.float32(split_line[3])
                h = np.float32(split_line[4])
                left = x - w / 2
                right = x + w / 2
                top = y - h / 2
                bottom = y + h / 2
                boxes.append([x, y, w, h, left, right, top, bottom, c])
            boxes = torch.Tensor(boxes)  # convert to tensor
            '''Data pre-process'''
            # random crop
            dw = int(ow*self.jitter)
            dh = int(oh*self.jitter)
            '''随机裁切的上下左右四个参数'''
            pleft  = int(random.uniform(-dw, dw))
            pright = int(random.uniform(-dw, dw))
            ptop = int(random.uniform(-dh, dh))
            pbot = int(random.uniform(-dh, dh))
            swidth = ow - pleft - pright
            sheight = oh - ptop - pbot
            '''裁切图像'''
            img_crop = self.rand_crop(img_rgb, oh, ow, pleft, ptop, swidth, sheight)
            # resize image
            '''将图像缩放至448*448'''
            img_resized = self.resize_image(img_crop, sheight, swidth)
            # random flip
            '''以0.5的概率随机左右翻转图像'''
            flip_rand_num = random.randint(0,1)
            if flip_rand_num == 1:
                img_resized = self.flip_image(img_resized)
            # distort image
            '''对图像做hsv变换'''
            img_resized = self.distort_image(img_resized)
            '''boxes adjust'''
            sx = np.float32(swidth / ow)
            sy = np.float32(sheight / oh)
            dx = np.float32(pleft / ow / sx)
            dy = np.float32(ptop / oh / sy)
            num_boxes = boxes.size(0)   # nums of boxes
            boxes = self.adj_boxes(boxes, num_boxes, flip_rand_num, dx, dy, np.float32(1 / sx), np.float32(1 / sy))
            '''boxes encode'''
            truth = self.encoder(boxes, num_boxes)
            # reshape to: (C, H, W)
            img_resized = img_resized.transpose(2, 0, 1)
            return torch.Tensor(img_resized), truth

    def __len__(self):
        return self.read_lines.__len__()

    def BGR2RGB(self,img_bgr):
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def RGB2BGR(self,img_rgb):
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # rewrite according to darknet
    '''根据darknet自行定义的rgb转hsv方法'''
    def RGB2HSV(self,img_rgb):
        img_hsv = np.float32(np.zeros((self.H, self.W, 3)))
        r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
        h, s, v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        df = mx - mn
        v = mx
        mask1 = np.where(mx == 0, True, False)
        mask2 = np.where(mx == r, True, False)
        mask3 = np.where(mx == g, True, False)
        s[mask1] = 0
        '''+0.000001防止分母为0'''
        s[(~ mask1)] = (df / (mx + 0.000001))[(~ mask1)]
        h[mask1] = 0
        h[(~ mask1) & mask2] = ((g - b) / (df + 0.000001))[(~ mask1) & mask2]
        h[(~ mask1) & mask3] = 2 + ((b - r) / (df + 0.000001))[(~ mask1) & mask3]
        h[(~ mask1) & (~ mask2) & (~ mask3)] = 4 + ((r - g) / (df + 0.000001))[(~ mask1) & (~ mask2) & (~ mask3)]
        h[(~ mask1)] = np.where(h[(~ mask1)] < 0, h[(~ mask1)] + 6, h[(~ mask1)])
        h[(~ mask1)] = h[(~ mask1)] / 6.
        img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2] = h, s, v
        return img_hsv

    def BGR2HSV(self,img_bgr):
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # rewrite according to darknet
    '''根据darknet自行定义的hsv转rgb方法'''
    def HSV2RGB(self,img_hsv):
        img_rgb = np.float32(np.zeros((self.H, self.W, 3)))
        r, g, b = img_rgb[:, : ,0], img_rgb[:, : ,1], img_rgb[:, :, 2]
        h, s, v = 6 * img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        mask1 = np.where(s == 0, True, False)
        r[mask1], g[mask1], b[mask1] = v[mask1], v[mask1], v[mask1]
        inx = np.floor(h)
        mask2 = np.where(inx == 0, True, False)
        mask3 = np.where(inx == 1, True, False)
        mask4 = np.where(inx == 2, True, False)
        mask5 = np.where(inx == 3, True, False)
        mask6 = np.where(inx == 4, True, False)
        f = h - inx
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1-f))
        r[(~ mask1) & mask2], g[(~ mask1) & mask2], b[(~ mask1) & mask2] = v[(~ mask1) & mask2], t[(~ mask1) & mask2], p[(~ mask1) & mask2]
        r[(~ mask1) & mask3], g[(~ mask1) & mask3], b[(~ mask1) & mask3] = q[(~ mask1) & mask3], v[(~ mask1) & mask3], p[(~ mask1) & mask3]
        r[(~ mask1) & mask4], g[(~ mask1) & mask4], b[(~ mask1) & mask4] = p[(~ mask1) & mask4], v[(~ mask1) & mask4], t[(~ mask1) & mask4]
        r[(~ mask1) & mask5], g[(~ mask1) & mask5], b[(~ mask1) & mask5] = p[(~ mask1) & mask5], q[(~ mask1) & mask5], v[(~ mask1) & mask5]
        r[(~ mask1) & mask6], g[(~ mask1) & mask6], b[(~ mask1) & mask6] = t[(~ mask1) & mask6], p[(~ mask1) & mask6], v[(~ mask1) & mask6]
        r[(~ mask1) & (~ mask2) & (~ mask3) & (~ mask4) & (~ mask5) & (~ mask6)], \
        g[(~ mask1) & (~ mask2) & (~ mask3) & (~ mask4) & (~ mask5) & (~ mask6)], \
        b[(~ mask1) & (~ mask2) & (~ mask3) & (~ mask4) & (~ mask5) & (~ mask6)] = \
            v[(~ mask1) & (~ mask2) & (~ mask3) & (~ mask4) & (~ mask5) & (~ mask6)], \
            p[(~ mask1) & (~ mask2) & (~ mask3) & (~ mask4) & (~ mask5) & (~ mask6)], \
            q[(~ mask1) & (~ mask2) & (~ mask3) & (~ mask4) & (~ mask5) & (~ mask6)]
        return img_rgb

    # random crop
    '''图像裁切'''
    def rand_crop(self, img_rgb, oh, ow, pleft, ptop, swidth, sheight):
        H_index = np.arange(sheight) + ptop
        H_index = np.clip(H_index, a_min = 0, a_max = oh - 1)
        W_index = np.arange(swidth) + pleft
        W_index = np.clip(W_index, a_min = 0, a_max = ow - 1)
        return img_rgb[H_index, :, :][:, W_index, :]

    # resize image
    '''图像缩放'''
    def resize_image(self, img_crop, oh, ow):
        w_scale = np.float32(ow - 1) / np.float32(self.W - 1)
        h_scale = np.float32(oh - 1) / np.float32(self.H - 1)
        # w方向缩放
        H_index = np.arange(self.H, dtype = np.int16)
        W_index = np.arange(self.W, dtype = np.int16)
        if ow == 1:
            W_index = 0
            img_part = img_crop[:, W_index, :]
        else:
            sx = W_index * w_scale
            ix = np.trunc(sx)
            ix = ix.astype(np.int16)
            dx = sx - ix
            ix[-1] = ow - 1
            ix_add1 = ix + 1   # ix + 1
            ix_add1[-1] = ow - 1
            img_part = (1 - dx) * img_crop[:, ix, :].transpose(2, 0, 1) + dx * img_crop[:, ix_add1, :].transpose(2, 0, 1)
        # h方向缩放
        sy = H_index *h_scale
        iy = np.trunc(sy)
        iy = iy.astype(np.int16)
        iy_add1 = iy + 1
        dy = sy - iy
        iy_add1[-1] = iy[-1]
        img_resized = (1 - dy) * img_part[:, iy, :].transpose(0, 2, 1) + dy * img_part[:, iy_add1, :].transpose(0, 2, 1)
        return img_resized.transpose(2, 1, 0)

    # image flip
    '''图像的左右翻转'''
    def flip_image(self, img_resized):
        W_index = self.W - 1 - np.arange(self.W)
        return img_resized[:, W_index, :]

    # image distort
    '''对图像应用随机hsv变换'''
    def distort_image(self, img_resized):
        dhue = random.uniform(-self.hue, self.hue)
        scale = random.uniform(1, self.saturation)
        if random.randint(0, 1) == 1:
            dsat = scale
        else:
            dsat = 1 / scale
        scale = random.uniform(1, self.exposure)
        if random.randint(0, 1) == 1:
            dexp = scale
        else:
            dexp = 1 / scale
        img_resized = self.RGB2HSV(img_resized)
        # s, v
        img_resized[:, :, 1] *= dsat
        img_resized[:, :, 2] *= dexp
        # h
        img_resized[:, :, 0] += dhue
        img_resized[:, :, 0] = np.where(img_resized[:, :, 0] > 1,
                                           img_resized[:, :, 0] - 1, img_resized[:, :, 0])
        img_resized[:, :, 0] = np.where(img_resized[:, :, 0] < 0,
                                           img_resized[:, :, 0] + 1, img_resized[:, :, 0])
        img_distort = np.clip(self.HSV2RGB(img_resized), a_min = 0, a_max = 1)
        return img_distort

    '''调整图像变换后随之发生变化的labels'''
    def adj_boxes(self, boxes, num_boxes, flip_rand_num, dx, dy, sx, sy):
        # boxes randomize
        for i in range(num_boxes):
            rand_box_num = random.randint(0, num_boxes - 1)
            boxes[[i, rand_box_num], :] = boxes[[rand_box_num, i], :]
        # correct boxes
        for i in range(num_boxes):
            if boxes[i, 0] == 0 or boxes[i, 1] == 0:
                boxes[i, 0] = 999999
                boxes[i, 1] = 999999
                boxes[i, 2] = 999999
                boxes[i, 3] = 999999
                continue
            boxes[i, 4] = boxes[i, 4] * sx - dx    # left
            boxes[i, 5] = boxes[i, 5] * sx - dx    # right
            boxes[i, 6] = boxes[i, 6] * sy - dy    # top
            boxes[i, 7] = boxes[i, 7] * sy - dy    # bottom
            if flip_rand_num == 1:
                swap = boxes[i, 4].detach().clone()    # need deep clone
                boxes[i, 4] = 1 - boxes[i, 5]
                boxes[i, 5] = 1 - swap
            # constrain: left, right, top, bottom
            boxes[i, [4, 5, 6, 7]] = boxes[i, [4, 5, 6, 7]].clamp_(0, 1)
            boxes[i, 0] = (boxes[i, 4] + boxes[i, 5]) / 2
            boxes[i, 1] = (boxes[i, 6] + boxes[i, 7]) / 2
            boxes[i, 2] = (boxes[i, 5] - boxes[i, 4])
            boxes[i, 3] = (boxes[i, 7] - boxes[i, 6])
            # constrain: w, h
            boxes[i, [2, 3]] = boxes[i, [2, 3]].clamp_(0, 1)
        return boxes

    '''将labels转换为ground truth后返回'''
    # boxes -> ground truth: 1*1225(25 * 49)
    def encoder(self, boxes, num_boxes):
        truth = torch.zeros(self.side * self.side * (self.classes + 5))
        for i in range(num_boxes):
            x, y, w, h, classes = boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3], boxes[i, 8]
            if w < 0.005 or h < 0.005:
                continue
            col = int(x * self.side)
            row = int(y * self.side)
            x = x * self.side - col
            y = y * self.side - row
            index = (col + row * self.side) * (self.classes + 5)
            #     continue
            if truth[index]:
                continue
            truth[index] = 1
            if classes < self.classes:
                truth[index + 1 + int(classes)] = 1
            truth[index + 1 + self.classes], truth[index + 1 + self.classes + 1], truth[index + 1 + self.classes + 2], \
            truth[index + 1 + self.classes + 3] = x, y, w, h
        return truth

'''dataset test case'''
if __name__ == "__main__":
    my_data = yolo_v1_dataset('', 'train.txt', 448, 448, 0.2, 1.5, 1.5, 0.1, 7, 20)
    start_time = time.time()
    img, label = my_data.__getitem__(4354)
    cv2.imshow('w1', my_data.RGB2BGR(np.array(img.permute(1, 2, 0))))
    cv2.waitKey(0)
    print(time.time() - start_time)
    print(torch.masked_select(label, label != 0))
    print(torch.where(label != 0))

