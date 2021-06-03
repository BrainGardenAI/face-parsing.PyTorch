#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

from EPE_detectors import DECA_FAN
from EPE_detectors import AFR

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) # + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    #vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        #cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(os.path.splitext(save_path[:-4])[0] +'.png', vis_im)
        #cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno_color)

    return vis_parsing_anno_color

def evaluate(respth='./res/seg_FAN', dspth='./data', cp='model_final_diss.pth', detector_fun=None):

    os.makedirs(respth, exist_ok=True)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in sorted(os.listdir(dspth)):
            img = Image.open(osp.join(dspth, image_path))
            w, h = img.size

            # align face size using detector
            if detector_fun:
                # crop image
                det_image = np.array(img)
                (cx, cy, size, isFace), det_res = detector_fun(image=det_image, return_image_result=True)
                if not isFace:
                    print("There is no face! {}".format(image_path))
                    continue

                hsize = int(round(size / 2))
                bbox_scale = 1.8
                k = max(h,w) / hsize / 2
                if k < 1.2:
                    new_hsize = int(max(h,w)/2)
                    print("{}: k={}, only pad".format(image_path, k))
                elif k < bbox_scale*1.15:
                    new_hsize = int(round(hsize*k))
                    print("{}: k={}, take as much as we can".format(image_path, k))
                else:
                    new_hsize = int(bbox_scale*hsize)
                    print("{}: k={}, crop twice the size of bbox".format(image_path, k))

                cx = int(round(cx))
                cy = int(round(cy - size * 0.15))
                l = max(cx-new_hsize, 0)
                t = max(cy-new_hsize, 0)
                r = min(cx+new_hsize, w)
                b = min(cy+new_hsize, h)
                pad_l = max(new_hsize-cx, 0)
                pad_t = max(new_hsize-cy, 0)
                pad_r = max(cx+new_hsize-w, 0)
                pad_b = max(cy+new_hsize-h, 0)
                seg_image = det_image[t:b, l:r, :]
                seg_image = cv2.copyMakeBorder(seg_image, left=pad_l, top=pad_t, right=pad_r, bottom=pad_b,
                                               borderType=cv2.BORDER_CONSTANT)
                seg_image = Image.fromarray(seg_image)

                cv2.rectangle(det_res, pt1=(l, t), pt2=(r, b), color=(255, 255, 255), thickness=4)
                # cv2.imwrite(os.path.join(respth, os.path.splitext(image_path)[0] + "_det.jpg"), cv2.cvtColor(det_res, cv2.COLOR_RGB2BGR))

                isCroppped = True
            else:
                seg_image = img
                isCropped = False

            seg_w, seg_h = seg_image.size
            if seg_w != seg_h:
                print("Image is not square! {}, W: {}, H: {}".format(image_path, seg_w, seg_h))
                exit()
            image = seg_image.resize((512, 512), Image.BILINEAR)
            seg_image = to_tensor(image)
            seg_image = torch.unsqueeze(seg_image, 0)
            seg_image = seg_image.cuda()
            out = net(seg_image)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            # print(parsing)
            print(np.unique(parsing))
            res = vis_parsing_maps(image, parsing, stride=1, save_im=False, save_path=osp.join(respth, image_path))

            # recover the size and position of the head fragment
            # 1 resize
            res = cv2.resize(res, (new_hsize*2, new_hsize*2), cv2.INTER_NEAREST)
            # 2 crop (remove padding that was used)
            ph, pw, _ = res.shape
            res = res[pad_t:ph-pad_b, pad_l:pw-pad_r]
            # 2 pad to position it on a canvas of frame size
            right = w - r
            bottom = h - b
            res = cv2.copyMakeBorder(res, top=t, left=l, right=right, bottom=bottom, borderType=cv2.BORDER_CONSTANT)

            cv2.imwrite(osp.join(respth, osp.splitext(image_path)[0] + ".png"), res)

if __name__ == "__main__":
    all_dataset = True
    detector_fun = DECA_FAN.return_bbox
    root_dir = "/disk/sdb1/avatars/dataset_EPE_data1"
    subdir = "real"
    # if not all_datasets, we can pick a certain actor and video
    actor_dir = "LeaSeydoux"
    video_dir = "5-LeaSeydoux"

    if all_dataset:
        actor_dirs = sorted([dir for dir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, dir))])

        La = len(actor_dirs)
        for ia, actor_dir in enumerate(actor_dirs):
            full_actor_subdir = os.path.join(root_dir, actor_dir, subdir)
            video_dirs = sorted(
                [dir for dir in os.listdir(full_actor_subdir) if os.path.isdir(os.path.join(full_actor_subdir, dir))])
            Lv = len(video_dirs)
            for iv, video_dir in enumerate(video_dirs):
                full_video_dir = os.path.join(full_actor_subdir, video_dir)
                from_path = os.path.join(full_video_dir, "frames")
                to_path = os.path.join(full_video_dir, "segments")
                evaluate(dspth=from_path, respth=to_path, cp='79999_iter.pth', detector_fun=detector_fun)
                print("{}/{} {}/{} done!".format(ia+1, La, iv+1, Lv))

    else:
        from_path = os.path.join(root_dir, actor_dir, subdir, video_dir, "frames")
        to_path = os.path.join(root_dir, actor_dir, subdir, video_dir, "segments")
        evaluate(dspth=from_path, respth=to_path, cp='79999_iter.pth', detector_fun=detector_fun)

