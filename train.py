# System libs
import os
import time
# import math
import random
import argparse
from distutils.version import LooseVersion
# Numerical libs
import torch
import torch.nn as nn
# Our libs
from mit_semseg.config import cfg
from eval import evaluate
from mit_semseg.dataset import ValDataset
from mit_semseg.dataset import TrainDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import AverageMeter, parse_devices, setup_logger, accuracy
from mit_semseg.lib.utils import as_numpy
from mit_semseg.lib.nn import async_copy_to
from mit_semseg.lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback
import matplotlib.pyplot as plt
from datetime import datetime
from torchvision import transforms
import numpy as np


def denormalize(x):
  mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1).type_as(x)
  std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1).type_as(x)
  return x * std + mean


# train one epoch
def train(segmentation_module, iterator, optimizers, history, epoch, cfg, gpus, nets):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()
    ave_acc_binary = AverageMeter()

    segmentation_module.train(not cfg.TRAIN.fix_bn)

    # main loop
    tic = time.time()
    for i in range(cfg.TRAIN.epoch_iters):
        # load a batch of data
        batch_data = next(iterator[0])
        data_time.update(time.time() - tic)
        segmentation_module.zero_grad()

        # adjust learning rate
        cur_iter = i + (epoch - 1) * cfg.TRAIN.epoch_iters
        adjust_learning_rate(optimizers, cur_iter, cfg)

        # img_ori = batch_data[0]['img_ori']
        # print('img_ori.shape :', img_ori.shape)

        # segSize = (batch_data[0]['img_ori'].shape[1],
        #            batch_data[0]['img_ori'].shape[2])
        # print('segSize :', segSize)

        # del batch_data[0]['img_ori']

        # # forward pass
        # # print('batch_data type :', type(batch_data[0]))
        # # print('batch_data[0]img_data shape :', batch_data[0]['img_data'][0].shape)
        # # print('batch_data[0]img_ori shape:', batch_data[0]['img_ori'][0].shape)
        # batch_tensor = denormalize(batch_data[0]['img_data'][0])
        # # print('batch_tensor shape :', batch_tensor.shape)
        # image = transforms.ToPILImage()(batch_tensor) #.convert("RGB")
        # # image = batch_tensor.cpu().detach().numpy().transpose(1, 2, 0)
        # np_image = np.asarray(image)

        # batch_seg_tensor = (batch_data[0]['seg_label'][0])
        # # label = transforms.ToPILImage()(batch_seg_tensor)
        # # np_label = np.asarray(label)
        # np_label = batch_seg_tensor.cpu().detach().numpy()

        # # image = batch_data[0]['img_data'][0].cpu().detach().numpy().transpose(1, 2, 0)
        # print('image min max :', np_image.min(), np_image.max())
        # print('np_image.shape :', np_image.shape)
        # print('np_label.shape :', np_label.shape)
        # # print('image shape :', image.size)
        # print()
        # plt.subplot(131)
        # # plt.imshow(img_ori[0])
        # # plt.axis('off')
        # plt.subplot(132)
        # plt.imshow(np_image)
        # # plt.axis('off')
        # # plt.savefig('./batch_tensor/%s.png' % int(datetime.now().timestamp()))
        # plt.subplot(133)
        # plt.imshow(np_label)
        # # plt.axis('off')
        # plt.savefig('./batch_tensor/%s.png' % int(datetime.now().timestamp()))

        loss, acc, acc_binary = segmentation_module(batch_data, object_index=cfg.MODEL.object_index)
        # pred = segmentation_module(batch_data, segSize=segSize)
        # print('pred.shape :', pred.shape)
        loss = loss.mean()
        acc = acc.mean()
        acc_binary = acc_binary.mean()

        # Backward
        # optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        #             validation part               #
        if i % cfg.TRAIN.disp_iter == 0:
          val_data = next(iterator[1])
          val_data = val_data[0]
          seg_label = as_numpy(val_data['seg_label'][0])
          img_resized_list = val_data['img_data']

          torch.cuda.synchronize()
          with torch.no_grad():
              segSize = (seg_label.shape[0], seg_label.shape[1])
              # print('segSize :', segSize)

              #           Use for binary loss       #
              scores = torch.zeros(1, 2, segSize[0], segSize[1])
              # scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
              scores = async_copy_to(scores, gpus[0])

              for img in img_resized_list:
                  feed_dict = val_data.copy()
                  feed_dict['img_data'] = img
                  del feed_dict['img_ori']
                  del feed_dict['info']
                  # print('type(feed_dict) :', type(feed_dict))
                  feed_dict = async_copy_to(feed_dict, gpus[0])

                  # forward pass
                  scores_tmp = segmentation_module([feed_dict], object_index=cfg.MODEL.object_index, segSize=segSize)
                  # print('scores_tmp.shape :', scores_tmp.shape)
                  scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)

          # _, pred = torch.max(scores, dim=1)
          # pred = as_numpy(pred.squeeze(0).cpu())

          #     Uae for binary loss   #
          pred = scores[:, [0], :, :].squeeze(1)
          pred = as_numpy(pred.squeeze(0).cpu())

          binary_seg_label = np.where(seg_label == cfg.MODEL.object_index, 0, 1)

          # binary_seg_label_tensor = torch.from_numpy(binary_seg_label[np.newaxis,: ,:]).cuda().type(torch.int64)
          # print('scores.shape, binary_seg_label_tensor.shape :', scores.shape, binary_seg_label_tensor.shape)

          # val_loss = segmentation_module.crit(scores, torch.from_numpy(binary_seg_label[np.newaxis,: ,:]).cuda().type(torch.int64))
          # print('shape comparing :', scores.shape, val_data['seg_label'].shape)
          
          # print('scores.shape, seg_label.shape :', scores.shape, seg_label.shape)
          val_loss = segmentation_module.crit(scores, val_data['seg_label'].cuda())
          # acc, pix = accuracy(pred, seg_label)

          # print('pred.shape, binary_seg_label.shape :', pred.shape, binary_seg_label.shape)
          # print('pred.shape, as_numpy(seg_label_gpu)[0].shape :', pred.shape, as_numpy(seg_label_gpu)[0].shape)
          # val_acc, val_pix = accuracy(pred, binary_seg_label)
          val_acc, val_pix = accuracy(pred, seg_label)
          # print('-------------------- Epoch {} Val_Accuracy: {:4.2f}, Val_Loss: {:.6f} --------------------'.format(epoch, val_acc * 100, val_loss))

          


        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item()*100)
        ave_acc_binary.update(acc_binary.data.item()*100)

        # calculate accuracy, and display
        if i % cfg.TRAIN.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_encoder: {:.6f}, lr_decoder: {:.6f}, '
                  'Accuracy: {:4.2f}, Accuracy_Binary: {:4.2f}, Loss: {:.6f}, '
                  'Val_Accuracy: {:4.2f}, Val_Loss: {:.6f}'
                  .format(epoch, i, cfg.TRAIN.epoch_iters,
                          batch_time.average(), data_time.average(),
                          cfg.TRAIN.running_lr_encoder, cfg.TRAIN.running_lr_decoder,
                          ave_acc.average(), ave_acc_binary.average(), ave_total_loss.average(), val_acc * 100, val_loss.data.item()))

            fractional_epoch = epoch - 1 + 1. * i / cfg.TRAIN.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data.item())
            history['train']['acc'].append(acc.data.item())   

            #       Checkpoint Best Model Ever        #
            #       Monitor Loss      #
            # if abs(val_loss.data.item()) < cfg.VAL.min_loss:
            #     cfg.VAL.min_loss = abs(val_loss.data.item())
            #     print('minimum val_loss :', abs(val_loss.data.item()), end=' ')
            #     # print('minimum val_loss :', val_loss.data.item(), end=' ')
            #     checkpoint(nets, history, cfg, epoch+1)

            #       Monitor Acc       #
            if val_acc > cfg.VAL.max_acc:
                cfg.VAL.max_acc = float(val_acc)
                print('maximum val_acc :', val_acc, end=' ')
                checkpoint(nets, history, cfg, epoch+1)

    # #             validation part               #
    # val_data = next(iterator[1])
    # val_data = val_data[0]
    # seg_label = as_numpy(val_data['seg_label'][0])
    # img_resized_list = val_data['img_data']

    # torch.cuda.synchronize()
    # with torch.no_grad():
    #     segSize = (seg_label.shape[0], seg_label.shape[1])
    #     # print('segSize :', segSize)
    #     # scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
    #     scores = torch.zeros(1, 2, segSize[0], segSize[1])
    #     scores = async_copy_to(scores, gpus[0])

    #     for img in img_resized_list:
    #             feed_dict = val_data.copy()
    #             feed_dict['img_data'] = img
    #             del feed_dict['img_ori']
    #             del feed_dict['info']
    #             # print('type(feed_dict) :', type(feed_dict))
    #             feed_dict = async_copy_to(feed_dict, gpus[0])

    #             # forward pass
    #             scores_tmp = segmentation_module([feed_dict], object_index=cfg.MODEL.object_index, segSize=segSize)
    #             scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)

    # _, pred = torch.max(scores, dim=1)
    # pred = as_numpy(pred.squeeze(0).cpu())
    # # pred = scores[:, [0], :, :].squeeze(1)
    # # pred = as_numpy(pred.squeeze(0).cpu())

    # binary_seg_label = np.where(seg_label == cfg.MODEL.object_index, 0, 1)

    # binary_seg_label_tensor = torch.from_numpy(binary_seg_label[np.newaxis,: ,:]).cuda().type(torch.int64)
    # # print('scores.shape, binary_seg_label_tensor.shape :', scores.shape, binary_seg_label_tensor.shape)
    # val_loss = segmentation_module.crit(scores, binary_seg_label_tensor)
    # # acc, pix = accuracy(pred, seg_label)
    # val_acc, val_pix = accuracy(pred, binary_seg_label)
    # print('-------------------- Epoch {} Val_Accuracy: {:4.2f}, Val_Loss: {:.6f} --------------------'.format(epoch, val_acc * 100, val_loss))  


def checkpoint(nets, history, cfg, epoch):
    print('Saving checkpoints...')
    (net_encoder, net_decoder, unet, crit) = nets

    dict_encoder = net_encoder.state_dict()
    dict_decoder = net_decoder.state_dict()
    dict_unet = unet.state_dict()

    torch.save(
        history,
        '{}/history_epoch_{}.pth'.format(cfg.DIR, cfg.MODEL.object_index))
    torch.save(
        dict_encoder,
        '{}/encoder_epoch_{}.pth'.format(cfg.DIR, cfg.MODEL.object_index))
    torch.save(
        dict_decoder,
        '{}/decoder_epoch_{}.pth'.format(cfg.DIR, cfg.MODEL.object_index))
    torch.save(
        dict_unet,
        '{}/unet_epoch_{}.pth'.format(cfg.DIR, cfg.MODEL.object_index))

    # torch.save(
    #     history,
    #     '{}/history_epoch_{}.pth'.format(cfg.DIR, epoch))
    # torch.save(
    #     dict_encoder,
    #     '{}/encoder_epoch_{}.pth'.format(cfg.DIR, epoch))
    # torch.save(
    #     dict_decoder,
    #     '{}/decoder_epoch_{}.pth'.format(cfg.DIR, epoch))


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, cfg):
    # (net_encoder, net_decoder, crit) = nets
    (net_encoder, net_decoder, unet, crit) = nets
    optimizer_encoder = torch.optim.SGD(
        group_weight(net_encoder),
        lr=cfg.TRAIN.lr_encoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_decoder = torch.optim.SGD(
        group_weight(net_decoder),
        lr=cfg.TRAIN.lr_decoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_unet = torch.optim.SGD(
        group_weight(unet),
        lr=cfg.TRAIN.lr_unet,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    
    return (optimizer_encoder, optimizer_decoder, optimizer_unet)


def adjust_learning_rate(optimizers, cur_iter, cfg):
    scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder * scale_running_lr
    cfg.TRAIN.running_lr_unet = cfg.TRAIN.lr_unet * scale_running_lr

    (optimizer_encoder, optimizer_decoder, optimizer_unet) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_decoder
    for param_group in optimizer_unet.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_unet


def main(cfg, gpus):
    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder)
    unet = ModelBuilder.build_unet(n_channels=5, 
        n_classes=2, 
        bilinear=True,
        weights=cfg.MODEL.weights_unet)

    crit = nn.NLLLoss(ignore_index=-1)

    if cfg.MODEL.arch_decoder.endswith('deepsup'):
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, unet, crit, cfg.TRAIN.deep_sup_scale)
    else:
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, unet, crit)

    # Dataset and Loader
    dataset_train = TrainDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_train,
        cfg.DATASET,
        batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=len(gpus),  # we have modified data_parallel
        shuffle=False,  # we do not use this param
        collate_fn=user_scattered_collate,
        num_workers=cfg.TRAIN.workers,
        drop_last=True,
        pin_memory=True)

    dataset_val = ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET)

    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    print('1 Epoch = {} iters'.format(cfg.TRAIN.epoch_iters))

    # create loader iterator
    iterator_train = iter(loader_train)
    iterator_val = iter(loader_val)

    # load nets into gpu
    if len(gpus) > 1:
        segmentation_module = UserScatteredDataParallel(
            segmentation_module,
            device_ids=gpus)
        # For sync bn
        patch_replication_callback(segmentation_module)
    segmentation_module.cuda()

    # Set up optimizers
    # nets = (net_encoder, net_decoder, crit)
    nets = (net_encoder, net_decoder, unet, crit)
    optimizers = create_optimizers(nets, cfg)

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}

    # min_loss = np.Inf
    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
        print('cfg.VAL.min_loss :', cfg.VAL.min_loss)
        train(segmentation_module, (iterator_train, iterator_val), optimizers, history, epoch+1, cfg, gpus, nets)

        # checkpointing
        # checkpoint(nets, history, cfg, epoch+1)

    print('Training Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0-3",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # Output directory
    if not os.path.isdir(cfg.DIR):
        os.makedirs(cfg.DIR)
    logger.info("Outputing checkpoints to: {}".format(cfg.DIR))
    with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    # Start from checkpoint
    if cfg.TRAIN.start_epoch > 0:
        cfg.MODEL.weights_encoder = os.path.join(
            cfg.DIR, 'encoder_epoch_{}_{}.pth'.format(cfg.MODEL.object_index, cfg.TRAIN.start_epoch))
        cfg.MODEL.weights_decoder = os.path.join(
            cfg.DIR, 'decoder_epoch_{}_{}.pth'.format(cfg.MODEL.object_index, cfg.TRAIN.start_epoch))

        if cfg.TRAIN.load_unet:
          cfg.MODEL.weights_unet = os.path.join(
              cfg.DIR, 'unet_epoch_{}_{}.pth'.format(cfg.MODEL.object_index, cfg.TRAIN.start_epoch))

          assert os.path.exists(cfg.MODEL.weights_unet), "checkpoint does not exitst!"

        assert os.path.exists(cfg.MODEL.weights_encoder) and \
            os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]
    num_gpus = len(gpus)
    cfg.TRAIN.batch_size = num_gpus * cfg.TRAIN.batch_size_per_gpu

    cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder

    random.seed(cfg.TRAIN.seed)
    torch.manual_seed(cfg.TRAIN.seed)

    main(cfg, gpus)
