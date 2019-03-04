import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from data import COCODetection, VOCDetection, detection_collate, BaseTransform, preproc
from layers.modules import MultiBoxLoss, RefineMultiBoxLoss
from layers.functions import Detect
from utils.nms_wrapper import nms, soft_nms
from configs.config import cfg, cfg_from_file
from models.model_builder import SSD
import torch.optim as optim
import torch.utils.data as data
import datetime
import os
import numpy as np
import pickle
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='RefineDet Training')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument(
    '--cfg',
    dest='cfg_file',
    required=True,
    help='Config file for training (and optionally testing)')
parser.add_argument('--ngpu', default=8, type=int, help='gpus')
parser.add_argument(
    '--resume_net', default=None, help='resume net for retraining')
parser.add_argument(
    '--resume_epoch',
    default=0,
    type=int,
    help='resume iter for retraining')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument(
    '--save_folder',
    default='./weights/',
    help='Location to save checkpoint models')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # args = arg_parse()
    cfg_from_file(args.cfg_file)
    save_folder = args.save_folder
    args.num_workers = args.workers
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    args.batch_size = cfg.TRAIN.BATCH_SIZE

    ngpus_per_node = args.ngpu
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


best_map = 0


def main_worker(gpu, ngpus_per_node, args):
    global best_map
    ## deal with args
    args.gpu = gpu
    cfg_from_file(args.cfg_file)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # distributed cfgs
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    torch.cuda.set_device(args.gpu)
    net = SSD(cfg)
    # print(net)
    if args.resume_net != None:
        checkpoint = torch.load(args.resume_net)
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

        print('Loading resume network...')

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            # print(args.gpu)
            torch.cuda.set_device(args.gpu)
            net.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
        else:
            net.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            net = torch.nn.parallel.DistributedDataParallel(net)
    elif args.gpu is not None:
        # torch.cuda.set_device(args.gpu)
        net = net.cuda(args.gpu)

    # args = arg_parse()

    batch_size = args.batch_size
    print("batch_size = ", batch_size)
    bgr_means = cfg.TRAIN.BGR_MEAN
    p = 0.6

    gamma = cfg.SOLVER.GAMMA
    momentum = cfg.SOLVER.MOMENTUM
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    size = cfg.MODEL.SIZE  # size =300
    thresh = cfg.TEST.CONFIDENCE_THRESH
    if cfg.DATASETS.DATA_TYPE == 'VOC':
        trainvalDataset = VOCDetection
        top_k = 1000
    else:
        trainvalDataset = COCODetection
        top_k = 1000
    dataset_name = cfg.DATASETS.DATA_TYPE
    dataroot = cfg.DATASETS.DATAROOT
    trainSet = cfg.DATASETS.TRAIN_TYPE
    valSet = cfg.DATASETS.VAL_TYPE
    num_classes = cfg.MODEL.NUM_CLASSES
    start_epoch = args.resume_epoch
    epoch_step = cfg.SOLVER.EPOCH_STEPS
    end_epoch = cfg.SOLVER.END_EPOCH
    args.num_workers = args.workers

    # optimizer

    optimizer = optim.SGD(
        net.parameters(),
        lr=cfg.SOLVER.BASE_LR,
        momentum=momentum,
        weight_decay=weight_decay)

    if cfg.MODEL.SIZE == '300':
        size_cfg = cfg.SMALL
    else:
        size_cfg = cfg.BIG
    # if args.resume_net != None:
    #	 checkpoint = torch.load(args.resume_net)
    #    optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True

    # deal with criterion
    criterion = list()
    if cfg.MODEL.REFINE:
        detector = Detect(cfg)
        arm_criterion = RefineMultiBoxLoss(cfg, 2)
        odm_criterion = RefineMultiBoxLoss(cfg, cfg.MODEL.NUM_CLASSES)
        arm_criterion.cuda(args.gpu)
        odm_criterion.cuda(args.gpu)
        criterion.append(arm_criterion)
        criterion.append(odm_criterion)
    else:
        detector = Detect(cfg)
        ssd_criterion = MultiBoxLoss(cfg)
        criterion.append(ssd_criterion)

    # deal with dataset
    TrainTransform = preproc(size_cfg.IMG_WH, bgr_means, p)
    ValTransform = BaseTransform(size_cfg.IMG_WH, bgr_means, (2, 0, 1))

    val_dataset = trainvalDataset(dataroot, valSet, ValTransform, dataset_name)
    val_loader = data.DataLoader(
        val_dataset,
        batch_size,
        shuffle=False,
        num_workers=args.num_workers * ngpus_per_node,
        collate_fn=detection_collate)
    # deal with training dataset
    train_dataset = trainvalDataset(dataroot, trainSet, TrainTransform, dataset_name)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, collate_fn=detection_collate, pin_memory=True, sampler=train_sampler)
    ## set net in training phase
    net.train()

    for epoch in range(start_epoch + 1, end_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=args.num_workers,
        #                               collate_fn=detection_collate)

        # Training
        train(train_loader, net, criterion, optimizer, epoch, epoch_step,
              gamma, end_epoch, cfg, args)

        
        if (epoch >= 0 and epoch % 10 == 0):
            #print("here",args.rank % ngpus_per_node)
            ## validation the model
            eval_net(val_dataset, val_loader, net, detector, cfg, ValTransform, args, top_k,
                thresh=thresh,batch_size=cfg.TEST.BATCH_SIZE)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            if (epoch % 10 == 0) or (epoch % 5 == 0 and epoch >= 60):
                save_name = os.path.join(args.save_folder,
                    cfg.MODEL.TYPE + "_epoch_{}_rank_{}_{}".format(str(epoch), str(args.rank), str(size)) + '.pth')
                save_checkpoint(net, epoch, size, optimizer, batch_size, save_name)


def adjust_learning_rate(optimizer, epoch, step_epoch, gamma, epoch_size,
                         iteration):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    ## warmup
    if epoch <= cfg.TRAIN.WARMUP_EPOCH:
        if cfg.TRAIN.WARMUP:
            iteration += (epoch_size * (epoch - 1))
            lr = 1e-6 + (cfg.SOLVER.BASE_LR - 1e-6) * iteration / (
                epoch_size * cfg.TRAIN.WARMUP_EPOCH)
        else:
            lr = cfg.SOLVER.BASE_LR
    else:
        div = 0
        if epoch > step_epoch[-1]:
            div = len(step_epoch) - 1
        else:
            for idx, v in enumerate(step_epoch):
                if epoch > step_epoch[idx] and epoch <= step_epoch[idx + 1]:
                    div = idx
                    break
        lr = cfg.SOLVER.BASE_LR * (gamma ** div)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(train_loader, net, criterion, optimizer, epoch, epoch_step, gamma,
          end_epoch, cfg, args):
    begin = time.time()
    epoch_size = len(train_loader)
    for iteration, (imgs, targets, _) in enumerate(train_loader):
        t0 = time.time()
        # adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, epoch_step, gamma,
                                  epoch_size, iteration)
        # imgs = imgs.cuda()
        imgs = imgs.cuda(args.gpu, non_blocking=True)
        imgs.requires_grad_()
        with torch.no_grad():
            targets = [anno.cuda(args.gpu, non_blocking=True) for anno in targets]
        # import pdb
        # pdb.set_trace()
        output = net(imgs)
        output = [anno.cuda(args.gpu) for anno in output]
        # output.cuda(args.gpu)

        if not cfg.MODEL.REFINE:
            ssd_criterion = criterion[0]
            loss_l, loss_c = ssd_criterion(output, targets)
            loss = loss_l + loss_c
        else:
            arm_criterion = criterion[0]
            odm_criterion = criterion[1]
            # print(output)
            # print("targets",targets)
            arm_loss_l, arm_loss_c = arm_criterion(output, targets)
            odm_loss_l, odm_loss_c = odm_criterion(
                output, targets, use_arm=True, filter_object=True)
            loss = arm_loss_l + arm_loss_c + odm_loss_l + odm_loss_c
        # clear the buf
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t1 = time.time()
        iteration_time = t1 - t0

        ## get the ETA
        all_time = ((end_epoch - epoch) * epoch_size +
                    (epoch_size - iteration)) * iteration_time
        eta = str(datetime.timedelta(seconds=int(all_time)))
        if iteration % 10 == 0:
            if not cfg.MODEL.REFINE:
                print('Epoch:' + repr(epoch) + ' || epochiter: ' +
                      repr(iteration % epoch_size) + '/' + repr(epoch_size) +
                      ' || L: %.4f C: %.4f||' %
                      (loss_l.item(), loss_c.item()) +
                      'iteration time: %.4f sec. ||' % (t1 - t0) +
                      'LR: %.5f' % (lr) + ' || eta time: {}'.format(eta))
            else:
                print('Epoch:' + repr(epoch) + ' || epochiter: ' +
                      repr(iteration % epoch_size) + '/' + repr(epoch_size) +
                      '|| arm_L: %.4f arm_C: %.4f||' %
                      (arm_loss_l.item(), arm_loss_c.item()) +
                      ' odm_L: %.4f odm_C: %.4f||' %
                      (odm_loss_l.item(), odm_loss_c.item()) +
                      ' loss: %.4f||' % (loss.item()) +
                      'iteration time: %.4f sec. ||' % (t1 - t0) +
                      'LR: %.5f' % (lr) + ' || eta time: {}'.format(eta))


def save_checkpoint(net, epoch, size, optimizer, batch_size, save_name):
    print("save model in",save_name)
    torch.save({
        'epoch': epoch,
        'size': size,
        'batch_size': batch_size,
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict()
    }, save_name)


def eval_net(val_dataset,
             val_loader,
             net,
             detector,
             cfg,
             transform,
             args,
             max_per_image=300,
             thresh=0.01,
             batch_size=1
             ):
    global best_map
    print('batch_size =',batch_size)
    net.eval()
    # net.priors.cuda(args.gpu,non_blocking=True)
    num_images = len(val_dataset)
    num_classes = cfg.MODEL.NUM_CLASSES
    eval_save_folder = "./eval/" + str(args.rank) +"/"
    if not os.path.exists(eval_save_folder):
        os.mkdir(eval_save_folder)
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    det_file = os.path.join(eval_save_folder, 'detections.pkl')
    st = time.time()
    for idx, (imgs, _, img_info) in enumerate(val_loader):
        with torch.no_grad():
            t1 = time.time()
            x = imgs
            # print('xshape = ',x.size(0))
            # x = x.cuda()
            x = x.cuda(args.gpu, non_blocking=True)
            output = net(x)
            t4 = time.time()
            boxes, scores = detector.forward(output)
            boxes = boxes.cpu().numpy()
            scores = scores.cpu().numpy()
            t2 = time.time()
            # for k in range(boxes.size(0)):
            true_nms_time = 0
            for k in range(boxes.shape[0]):
                i = idx * batch_size + k
                boxes_ = boxes[k]
                scores_ = scores[k]
                # boxes_ = boxes_.cpu().numpy()
                # scores_ = scores_.cpu().numpy()
                img_wh = img_info[k]
                scale = np.array([img_wh[0], img_wh[1], img_wh[0], img_wh[1]])
                boxes_ *= scale
                for j in range(1, num_classes):
                    inds = np.where(scores_[:, j] > thresh)[0]
                    if len(inds) == 0:
                        #print('j=',j)
                        all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                        continue
                    c_bboxes = boxes_[inds]
                    c_scores = scores_[inds, j]
                    c_dets = np.hstack((c_bboxes,
                                        c_scores[:, np.newaxis])).astype(
                        np.float32, copy=False)
                    # t_nms_s = time.time()
                    # print(c_dets.shape[0])
                    keep = nms(c_dets, cfg.TEST.NMS_OVERLAP, force_cpu=True)
                    # t_nms_e = time.time()
                    # true_nms_time +=(t_nms_e-t_nms_s)

                    keep = keep[:50]
                    c_dets = c_dets[keep, :]
                    all_boxes[j][i] = c_dets
            t3 = time.time()
            detect_time = t2 - t1
            nms_time = t3 - t2
            forward_time = t4 - t1
            if idx % 10 == 0:
                print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s {:.3f}s'.format(
                    i + 1, num_images, forward_time, detect_time, nms_time))
    print("detect time: ", time.time() - st)
    with open(det_file, 'wb') as f:
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_EX) 
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    print('Evaluating detections')
    map_now = val_dataset.evaluate_detections_dist(all_boxes, eval_save_folder)
    if map_now > best_map:
        best_map = map_now
        print("higher map is =", best_map)


if __name__ == '__main__':
    main()
