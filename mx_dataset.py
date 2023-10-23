import mxnet as mx
import numpy as np
import numbers
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def _get_new_box(src_w, src_h, bbox, scale):
    x = bbox[0]
    y = bbox[1]
    box_w = bbox[2] - bbox[0]
    box_h = bbox[3] - bbox[1]

    scale = min((src_h-1)/box_h, min((src_w-1)/box_w, scale))

    new_width = box_w * scale
    new_height = box_h * scale
    center_x, center_y = box_w/2+x, box_h/2+y

    left_top_x = center_x-new_width/2
    left_top_y = center_y-new_height/2
    right_bottom_x = center_x+new_width/2
    right_bottom_y = center_y+new_height/2

    if left_top_x < 0:
        right_bottom_x -= left_top_x
        left_top_x = 0

    if left_top_y < 0:
        right_bottom_y -= left_top_y
        left_top_y = 0

    if right_bottom_x > src_w-1:
        left_top_x -= right_bottom_x-src_w+1
        right_bottom_x = src_w-1

    if right_bottom_y > src_h-1:
        left_top_y -= right_bottom_y-src_h+1
        right_bottom_y = src_h-1

    return int(left_top_x), int(left_top_y),\
            int(right_bottom_x), int(right_bottom_y)


def clamp(x, min_x, max_x):
    return min(max(x, min_x), max_x)


class MX_WFAS(Dataset):
    def __init__(self, path_imgrec, path_imgidx, transform, scale=1.0, multi_learning=False):
        super(MX_WFAS, self).__init__()
        self.transform = transform
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        self.imgidx = np.array(list(self.imgrec.keys))
        self.multi_learning = multi_learning
        self.scale = scale

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        labels = header.label
        sample = mx.image.imdecode(img).asnumpy()  # RGB
        bbox = labels[2:6].astype(np.int32)
        label = int(labels[0])
        if self.multi_learning:
            labels = [label, int(labels[1]) + 1]  # 0: live, 1->n: spoof_type
        else:
            labels = label
        
        # crop face bbox
        # scale = np.random.uniform(1.0, 1.2)
        bbox = _get_new_box(sample.shape[0], sample.shape[1], bbox, scale=self.scale)
        sample = sample[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()
        
        if self.transform is not None:
            sample = self.transform(label=label, img=sample)['image']
        sample = np.transpose(sample, (2, 0, 1)).astype(np.float32)

        return (torch.tensor(sample), torch.tensor(labels, dtype=torch.long))

    def __len__(self):
        return len(self.imgidx)
