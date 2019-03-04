import os
import sys
#sys.path.insert(1, './')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import argparse

from data import COCODetection, VOCDetection, detection_collate, BaseTransform, preproc, COCOJUGGDetection
from layers.modules import MultiBoxLoss, RefineMultiBoxLoss
from layers.functions import Detect
from utils.nms_wrapper import nms, soft_nms
from configs.config import cfg, cfg_from_file, VOC_CLASSES, COCO_CLASSES
from utils.box_utils import draw_rects
import numpy as np
import time
import os
import pickle
from models.model_builder import SSD
import cv2


def arg_parse():
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detection')
    parser.add_argument(
        "--images",
        dest='images',
        help="Image / Directory containing images to perform detection upon",
        default="images",
        type=str)
    parser.add_argument(
        '--weights',
        default='weights/ssd_darknet_300.pth',
        type=str,
        help='Trained state_dict file path to open')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        required=True,
        help='Config file for training (and optionally testing)')
    parser.add_argument(
        '--save_folder',
        default='eval/',
        type=str,
        help='File path to save results')
    parser.add_argument(
        '--num_workers',
        default=8,
        type=int,
        help='Number of workers used in dataloading')
    parser.add_argument(
        '--retest', default=False, type=bool, help='test cache results')
    args = parser.parse_args()
    return args


def im_detect_batch(imgs, img_info, net, detector, thresh=0.01, num_classes=10):
    num_images = len(imgs)
    boxes_batch = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    with torch.no_grad():
        t1 = time.time()
        x = torch.from_numpy(np.array(imgs))
        print(x.shape)
        x = x.cuda()
        output = net(x)
        t4 = time.time()
        boxes, scores = detector.forward(output)
        t2 = time.time()
        for k in range(boxes.size(0)):
            i = k
            boxes_ = boxes[k]
            scores_ = scores[k]
            img_wh = img_info[k]
            boxes_ = boxes_.cpu().numpy()
            scores_ = scores_.cpu().numpy()
            scale = np.array([img_wh[0], img_wh[1], img_wh[0], img_wh[1]])
            boxes_ *= scale
            for j in range(1, num_classes):
                inds = np.where(scores_[:, j] > thresh)[0]
                if len(inds) == 0:
                    boxes_batch[j][i] = np.empty([0, 5], dtype=np.float32)
                    continue
                c_bboxes = boxes_[inds]
                c_scores = scores_[inds, j]
                c_dets = np.hstack((c_bboxes,
                                    c_scores[:, np.newaxis])).astype(
                    np.float32, copy=False)
                keep = nms(c_dets, cfg.TEST.NMS_OVERLAP, force_cpu=True)
                keep = keep[:50]
                c_dets = c_dets[keep, :]
                boxes_batch[j][i] = c_dets
        t3 = time.time()
        detect_time = t2 - t4
        nms_time = t3 - t2
        forward_time = t4 - t1
        fps_time = t3 - t1
        print('im_detect: forward_time: {:.3f}s, detect_time {:.3f}s, nms_time: {:.3f}s,  fps_time: {:.3f}s'.format(
            forward_time, detect_time, nms_time, fps_time))
    return boxes_batch, fps_time, forward_time, detect_time, nms_time

def get_voc_test():
    image_sets = [['2007', 'test']]
    root = '/workspace/mnt/group/blademaster/jianglielin/datasets/VOCdevkit'
    ids = []
    for (year, name) in image_sets:
        rootpath = os.path.join(root, 'VOC' + year)
        for line in open(
                os.path.join(rootpath, 'ImageSets', 'Main',
                             name + '.txt')):
            ids.append((rootpath, line.strip()))
    imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
    items = []
    for id in ids:
        real_img_path = imgpath % (id)
        items.append(real_img_path.split('/')[-1])
    return items

def pre_process(img, resize_wh=[512, 512], rgb_means=[104,117,123], swap=(2, 0, 1)):
    interp_methods = [
        cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA,
        cv2.INTER_NEAREST, cv2.INTER_LANCZOS4
    ]
    interp_method = interp_methods[0]
    img_info = [img.shape[1], img.shape[0]]
    img = cv2.resize(
        np.array(img), (resize_wh[0], resize_wh[1]),
        interpolation=interp_method).astype(np.float32)
    img -= rgb_means
    img = img.transpose(swap)
    return img, img_info


def main():
    global args
    args = arg_parse()
    cfg_from_file(args.cfg_file)
    bgr_means = cfg.TRAIN.BGR_MEAN
    dataset_name = cfg.DATASETS.DATA_TYPE
    batch_size = cfg.TEST.BATCH_SIZE
    num_workers = args.num_workers
    if cfg.DATASETS.DATA_TYPE == 'VOC':
        trainvalDataset = VOCDetection
        classes = VOC_CLASSES
        top_k = 200
    elif cfg.DATASETS.DATA_TYPE == 'JUGG':
        trainvalDataset = COCOJUGGDetection
        top_k = 200
    else:
        trainvalDataset = COCODetection
        classes = COCO_CLASSES
        top_k = 300
    valSet = cfg.DATASETS.VAL_TYPE
    num_classes = cfg.MODEL.NUM_CLASSES
    save_folder = args.save_folder
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cfg.TRAIN.TRAIN_ON = False
    net = SSD(cfg)

    checkpoint = torch.load(args.weights)
    state_dict = checkpoint['model']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

    detector = Detect(cfg)
    img_wh = cfg.TEST.INPUT_WH
    print(img_wh)
    ValTransform = BaseTransform(img_wh, bgr_means, (2, 0, 1))
    input_folder = args.images
    thresh = cfg.TEST.CONFIDENCE_THRESH
    all_items = os.listdir(input_folder)
    ##for voc debug
    all_items = get_voc_test()
    # validation
    if cfg.DATASETS.DATA_TYPE == 'JUGG':
        num_classes -= 1
        val_ann_file = '/workspace/mnt/group/blademaster/jianglielin/datasets/juggdet_train_test/juggdet_0830/Lists/annotations/juggdet_0503_test_0711.json'
        val_dataset = trainvalDataset(cfg.DATASETS.DATAROOT, valSet, val_ann_file, ValTransform, dataset_name)
        all_items = []
        for id in val_dataset.ids:
            all_items.append(id.split('/')[-1])
    else:
        val_dataset = trainvalDataset(cfg.DATASETS.DATAROOT, valSet, ValTransform, dataset_name)
    batch_size = 1
    imgs = []
    img_infos = []
    img_names = []
    all_boxes = [[[] for _ in range(len(all_items))] for _ in range(num_classes)]
    idx = 0
    all_fps_time = 0.0
    all_forward_time = 0.0
    all_detect_time = 0.0
    all_nms_time = 0.0
    net.eval()
    t1 = time.time()
    for i in range(len(all_items)):
        item = all_items[i]
        img_path = os.path.join(input_folder, item)
        #print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR + cv2.IMREAD_IGNORE_ORIENTATION)
        img_resize, img_info = pre_process(img, img_wh)
        imgs.append(img_resize)
        img_infos.append(img_info)
        img_names.append(item)
        if len(imgs) == batch_size or i == len(all_items) - 1:
            if idx % 50 == 0:
                print('idx:', idx)
            dets, fps_time, forward_time, detect_time, nms_time= im_detect_batch(imgs, img_infos, net, detector, thresh, num_classes)
            all_fps_time += fps_time
            all_forward_time += forward_time
            all_detect_time += detect_time
            all_nms_time += nms_time

            for k in range(len(imgs)):
                i = idx * batch_size + k
                for j in range(1, num_classes):
                    all_boxes[j][i] = dets[j][k]
            idx += 1
            imgs = []
            img_infos = []
    t2 = time.time()
    all_inference_time = t2 - t1
    print('all fps time: {:3f}s, per_image: {:3f}s\n'
          'all detect time: {:3f}s, per_image: {:3f}s\n'
          'all_forward_time: {:3f}s, per_image: {:3f}s\n'
          'all_post_precess_time: {:3f}s, per_image: {:3f}s\n'
          'all_nms_time: {:3f}s, per_image: {:3f}s'.format(
        all_fps_time, all_fps_time / len(all_items),
        all_inference_time, all_inference_time / len(all_items),
        all_forward_time, all_forward_time/ len(all_items),
        all_detect_time, all_detect_time / len(all_items),
        all_nms_time, all_nms_time / len(all_items)))

    date = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
    eval_save_folder = os.path.join("./eval_batch_demo/", date)
    if not os.path.exists(eval_save_folder):
        os.makedirs(eval_save_folder)
    det_file = os.path.join(eval_save_folder, 'detections_demo.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    print('Evaluating detections')
    val_dataset.evaluate_detections(all_boxes, eval_save_folder)
    print("detect time: ", time.time() - st)


if __name__ == '__main__':
    st = time.time()
    main()
    print("final time", time.time() - st)
