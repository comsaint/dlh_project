import torch
import numpy as np
import cv2
from PIL import Image
from skimage.measure import label
from config import DEVICE, LOCAL_IMAGE_SIZE, GLOBAL_IMAGE_SIZE
from dataset import make_data_transform

tfx = make_data_transform(GLOBAL_IMAGE_SIZE)

preprocess = tfx['test']
ori_size = (GLOBAL_IMAGE_SIZE, GLOBAL_IMAGE_SIZE)
crop_size = (LOCAL_IMAGE_SIZE, LOCAL_IMAGE_SIZE)


def attention_gen_patches(ori_image, fm_cuda, threshold):
    # feature map -> feature mask (using feature map to crop on the original image) -> crop -> patches
    feature_conv = fm_cuda.detach().cpu().numpy()
    bz, nc, h, w = feature_conv.shape
    ori_image = ori_image.permute(0, 2, 3, 1)

    patches = torch.FloatTensor().to(DEVICE)
    for i in range(0, bz):
        feature = feature_conv[i]
        cam = feature.reshape((nc, h * w))
        cam = cam.sum(axis=0)  # sum channels
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)

        heatmap_bin = bin_image(cv2.resize(cam_img, ori_size), threshold)
        heatmap_maxconn = select_max_connect(heatmap_bin)
        heatmap_mask = heatmap_bin * heatmap_maxconn
        ind = np.argwhere(heatmap_mask != 0)
        minh = min(ind[:, 0])
        minw = min(ind[:, 1])
        maxh = max(ind[:, 0])
        maxw = max(ind[:, 1])
        #print(f"Mask Area %: {(maxh- minh)*(maxw-minw) / (ori_size[0]*ori_size[1]) * 100}")

        # to ori image
        image = ori_image[i].detach().cpu().numpy()
        image_crop = image[minh:maxh, minw:maxw, :] * 255  # because image was normalized before
        image_crop = cv2.resize(image_crop, crop_size)
        image_crop = preprocess(Image.fromarray(image_crop.astype('uint8')).convert('RGB'))
        image_crop = image_crop.reshape(3, *crop_size).unsqueeze(0).to(DEVICE)
        patches = torch.cat((patches, image_crop), 0)
    return patches


def bin_image(heatmap, threshold):
    t = int(255 * threshold)
    _, heatmap_bin = cv2.threshold(heatmap, t, 255, cv2.THRESH_BINARY)  # 178 ~= 255*0.7
    return heatmap_bin


def select_max_connect(heatmap):
    labeled_img, num = label(heatmap, connectivity=2, background=0, return_num=True)
    max_label = 0
    max_num = 0
    for i in range(1, num+1):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)
    if max_num == 0:
        lcc = (labeled_img == -1)
    lcc = lcc + 0
    return lcc
