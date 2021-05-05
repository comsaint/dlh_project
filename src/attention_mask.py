import operator
import torch
import numpy as np
import cv2
from PIL import Image
from skimage.measure import label
from config import DEVICE, NUM_CLASSES, LOCAL_IMAGE_SIZE, FINE_TUNE, GLOBAL_IMAGE_SIZE, NUM_WORKERS, HEATMAP_THRESHOLD
from model import initialize_model
from dataset import make_data_transform, load_data
from data_processing import load_data_file

tfx = make_data_transform(GLOBAL_IMAGE_SIZE)

preprocess = tfx['test']
ori_size = (GLOBAL_IMAGE_SIZE, GLOBAL_IMAGE_SIZE)
crop_size = (LOCAL_IMAGE_SIZE, LOCAL_IMAGE_SIZE)


# TODO: deprecated
def load_model(model_name, model_path=None):
    model, _, _ = initialize_model(model_name,
                                   num_classes=NUM_CLASSES,
                                   fine_tune=FINE_TUNE)
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model = model.to(DEVICE)

    # get the layer to apply heatmap algorithm on
    # FIXME: check layer for models
    if model_name == "densenet":
        fm_name = "features.norm5"
        pool_name = "features.norm5"
    elif model_name in ("resnet50", 'resnext50', 'resnext101'):
        fm_name = "layer4.2.relu"  # TODO: or layer4.2.conv3 ?
        pool_name = "avgpool"
    else:
        raise NotImplementedError(f"Unknown model '{model_name}' target layer for heatmap.")
    return model, fm_name, pool_name


def attention_gen_patches(ori_image, fm_cuda):
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

        heatmap_bin = bin_image(cv2.resize(cam_img, ori_size))
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


def bin_image(heatmap, threshold=HEATMAP_THRESHOLD):
    t = int(255*threshold)
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


def test():
    model_name = 'resnext101'
    model_path = '../models/model_experiment_resnext101_val005_adamw_1/resnext101_best.pth'
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # load model
    m, fm_hook, pool_hook = load_model(model_name, model_path)
    # register hooks to get intermediate output
    #fm_hook = 'features.norm5'
    #pool_hook = 'features.norm5'
    # e.g. activation['fm'] = m.features.norm5.register_forward_hook(get_activation('fm'))
    activation['fm'] = operator.attrgetter(fm_hook)(m).register_forward_hook(get_activation('fm'))
    # e.g. activation['pool'] = m.features.norm5.register_forward_hook(get_activation('pool'))
    activation['pool'] = operator.attrgetter(pool_hook)(m).register_forward_hook(get_activation('pool'))

    # load data file
    df_data, _ = load_data_file(sampling=16)
    df_data = df_data.reset_index()

    dataloader = load_data(df_data, batch_size=16, transform=tfx['test'], shuffle=False, num_workers=NUM_WORKERS)

    for i, (x, target, _) in enumerate(dataloader):
        x = x.cuda()
        output_global = m(x)  #
        fm_global = activation['fm']
        patchs_var = attention_gen_patches(x, fm_global)

        '''
        # plot
        for j in range(2):
            inp = x[j].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            inp = inp.astype('uint8')
            inp = Image.fromarray(inp).convert('L')
            inp.show()

            p = patchs_var[j].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            p = p.astype('uint8')
            p = Image.fromarray(p).convert('L')
            p.show()
        '''
    return None


if __name__ == "__main__":
    test()
