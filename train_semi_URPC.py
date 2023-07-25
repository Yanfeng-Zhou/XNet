from torchvision import transforms, datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.getnetwork import get_network
import argparse
import time
import os
import numpy as np
from torch.backends import cudnn
import random
from PIL import Image
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import sys

from config.dataset_config.dataset_cfg import dataset_cfg
from config.augmentation.online_aug import data_transform_2d, data_normalize_2d
from loss.loss_function import segmentation_loss, entropy_loss
from models.getnetwork import get_network
from dataload.dataset_2d import imagefloder_itn
from config.visdom_config.visual_visdom import visdom_initialization_EM, visualization_EM, visual_image_sup
from config.warmup_config.warmup import GradualWarmupScheduler
from config.train_test_config.train_test_config import print_train_loss_EM, print_val_loss_sup, print_train_eval_sup, print_val_eval_sup, save_val_best_sup_2d, draw_pred_sup, print_best_sup
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)


def init_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_trained_models', default='/mnt/data1/XNet/checkpoints/semi')
    parser.add_argument('--path_seg_results', default='/mnt/data1/XNet/seg_pred/semi')
    parser.add_argument('-pd', '--path_dataset', default='/mnt/data1/XNet/dataset/CREMI')
    parser.add_argument('--dataset_name', default='CREMI', help='CREMI, ISIC-2017, GlaS')
    parser.add_argument('--input1', default='image')
    parser.add_argument('--sup_mark', default='20')
    parser.add_argument('--unsup_mark', default='80')
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('-e', '--num_epochs', default=200, type=int)
    parser.add_argument('-s', '--step_size', default=50, type=int)
    parser.add_argument('-l', '--lr', default=0.5, type=float)
    parser.add_argument('-g', '--gamma', default=0.5, type=float)
    parser.add_argument('-u', '--unsup_weight', default=1, type=float)
    parser.add_argument('--loss', default='dice')
    parser.add_argument('-w', '--warm_up_duration', default=20)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow')

    parser.add_argument('-i', '--display_iter', default=5, type=int)
    parser.add_argument('-n', '--network', default='unet_urpc', type=str)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--rank_index', default=0, help='0, 1, 2, 3')
    parser.add_argument('-v', '--vis', default=True, help='need visualization or not')
    parser.add_argument('--visdom_port', default=16672)
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    rank = torch.distributed.get_rank()
    ngpus_per_node = torch.cuda.device_count()
    init_seeds(rank + 1)

    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)

    print_num = 77 + (cfg['NUM_CLASSES'] - 3) * 14
    print_num_minus = print_num - 2
    print_num_half = int(print_num / 2 - 1)

    # trained model save
    path_trained_models = args.path_trained_models + '/' + str(os.path.split(args.path_dataset)[1])
    if not os.path.exists(path_trained_models) and rank == args.rank_index:
        os.mkdir(path_trained_models)
    path_trained_models = path_trained_models + '/' + 'URPC' + '-l=' + str(args.lr) + '-e=' + str(args.num_epochs) + '-s=' + str(args.step_size) + '-g=' + str(args.gamma) + '-b=' + str(args.batch_size) + '-cw=' + str(args.unsup_weight) + '-w=' + str(args.warm_up_duration) + '-' + str(args.sup_mark) + '-' + str(args.unsup_mark) + '-' + str(args.input1)
    if not os.path.exists(path_trained_models) and rank == args.rank_index:
        os.mkdir(path_trained_models)

    # seg results save
    path_seg_results = args.path_seg_results + '/' + str(os.path.split(args.path_dataset)[1])
    if not os.path.exists(path_seg_results) and rank == args.rank_index:
        os.mkdir(path_seg_results)
    path_seg_results = path_seg_results + '/' + 'URPC' + '-l=' + str(args.lr) + '-e=' + str(args.num_epochs) + '-s=' + str(args.step_size) + '-g=' + str(args.gamma) + '-b=' + str(args.batch_size) + '-cw=' + str(args.unsup_weight) + '-w=' + str(args.warm_up_duration) + '-' + str(args.sup_mark) + '-' + str(args.unsup_mark) + '-' + str(args.input1)
    if not os.path.exists(path_seg_results) and rank == args.rank_index:
        os.mkdir(path_seg_results)

    # vis
    if args.vis and rank == args.rank_index:
        visdom_env = str('Semi-URPC-' + str(os.path.split(args.path_dataset)[1]) + '-' + args.network + '-l=' + str(args.lr) + '-e=' + str(args.num_epochs) + '-s=' + str(args.step_size) + '-g=' + str(args.gamma) + '-b=' + str(args.batch_size) + '-cw=' + str(args.unsup_weight) + '-w=' + str(args.warm_up_duration) + '-' + str(args.sup_mark) + '-' + str(args.unsup_mark) + '-' + str(args.input1))
        visdom = visdom_initialization_EM(env=visdom_env, port=args.visdom_port)

    if args.input1 == 'image':
        input1_mean = 'MEAN'
        input1_std = 'STD'
    else:
        input1_mean = 'MEAN_' + args.input1
        input1_std = 'STD_' + args.input1

    data_transforms = data_transform_2d()
    data_normalize = data_normalize_2d(cfg[input1_mean], cfg[input1_std])

    dataset_train_unsup = imagefloder_itn(
        data_dir=args.path_dataset + '/train_unsup_' + args.unsup_mark,
        input1=args.input1,
        data_transform_1=data_transforms['train'],
        data_normalize_1=data_normalize,
        sup=False,
        num_images=None,
    )
    num_images_unsup = len(dataset_train_unsup)

    dataset_train_sup = imagefloder_itn(
        data_dir=args.path_dataset + '/train_sup_' + args.sup_mark,
        input1=args.input1,
        data_transform_1=data_transforms['train'],
        data_normalize_1=data_normalize,
        sup=True,
        num_images=num_images_unsup,
    )
    dataset_val = imagefloder_itn(
        data_dir=args.path_dataset + '/val',
        input1=args.input1,
        data_transform_1=data_transforms['val'],
        data_normalize_1=data_normalize,
        sup=True,
        num_images=None,
    )

    train_sampler_sup = torch.utils.data.distributed.DistributedSampler(dataset_train_sup, shuffle=True)
    train_sampler_unsup = torch.utils.data.distributed.DistributedSampler(dataset_train_unsup, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)

    dataloaders = dict()
    dataloaders['train_sup'] = DataLoader(dataset_train_sup, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8, sampler=train_sampler_sup)
    dataloaders['train_unsup'] = DataLoader(dataset_train_unsup, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8, sampler=train_sampler_unsup)
    dataloaders['val'] = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8, sampler=val_sampler)

    num_batches = {'train_sup': len(dataloaders['train_sup']), 'train_unsup': len(dataloaders['train_unsup']), 'val': len(dataloaders['val'])}

    model1 = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])

    model1 = model1.cuda()
    model1 = DistributedDataParallel(model1, device_ids=[args.local_rank])
    dist.barrier()

    criterion = segmentation_loss(args.loss, False).cuda()
    kl_distance = nn.KLDivLoss(reduction='none')

    optimizer1 = optim.SGD(model1.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5 * 10 ** args.wd)
    exp_lr_scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup1 = GradualWarmupScheduler(optimizer1, multiplier=1.0, total_epoch=args.warm_up_duration, after_scheduler=exp_lr_scheduler1)

    since = time.time()
    count_iter = 0

    best_val_eval_list = [0 for i in range(4)]

    for epoch in range(args.num_epochs):

        count_iter += 1
        if (count_iter - 1) % args.display_iter == 0:
            begin_time = time.time()

        dataloaders['train_sup'].sampler.set_epoch(epoch)
        dataloaders['train_unsup'].sampler.set_epoch(epoch)
        model1.train()

        train_loss_sup_1 = 0.0
        train_loss_unsup = 0.0
        train_loss = 0.0

        val_loss_sup_1 = 0.0

        unsup_weight = args.unsup_weight * (epoch + 1) / args.num_epochs

        dist.barrier()

        dataset_train_sup = iter(dataloaders['train_sup'])
        dataset_train_unsup = iter(dataloaders['train_unsup'])

        for i in range(num_batches['train_sup']):

            unsup_index = next(dataset_train_unsup)
            img_train_unsup_1 = unsup_index['image']
            img_train_unsup_1 = Variable(img_train_unsup_1.cuda(non_blocking=True))

            optimizer1.zero_grad()

            pred_train_unsup1, pred_train_unsup2, pred_train_unsup3, pred_train_unsup4 = model1(img_train_unsup_1)
            pred_train_unsup1 = torch.softmax(pred_train_unsup1, 1)
            pred_train_unsup2 = torch.softmax(pred_train_unsup2, 1)
            pred_train_unsup3 = torch.softmax(pred_train_unsup3, 1)
            pred_train_unsup4 = torch.softmax(pred_train_unsup4, 1)

            preds = (pred_train_unsup1 + pred_train_unsup2 + pred_train_unsup3 + pred_train_unsup4) / 4

            variance_aux1 = torch.sum(kl_distance(torch.log(preds), pred_train_unsup1), dim=1, keepdim=True)
            exp_variance_aux1 = torch.exp(-variance_aux1)
            variance_aux2 = torch.sum(kl_distance(torch.log(preds), pred_train_unsup2), dim=1, keepdim=True)
            exp_variance_aux2 = torch.exp(-variance_aux2)
            variance_aux3 = torch.sum(kl_distance(torch.log(preds), pred_train_unsup3), dim=1, keepdim=True)
            exp_variance_aux3 = torch.exp(-variance_aux3)
            variance_aux4 = torch.sum(kl_distance(torch.log(preds), pred_train_unsup4), dim=1, keepdim=True)
            exp_variance_aux4 = torch.exp(-variance_aux4)

            consistency_dist_aux1 = (preds - pred_train_unsup1) ** 2
            consistency_loss_aux1 = torch.mean(consistency_dist_aux1 * exp_variance_aux1) / (torch.mean(exp_variance_aux1) + 1e-8) + torch.mean(variance_aux1)
            consistency_dist_aux2 = (preds - pred_train_unsup2) ** 2
            consistency_loss_aux2 = torch.mean(consistency_dist_aux2 * exp_variance_aux2) / (torch.mean(exp_variance_aux2) + 1e-8) + torch.mean(variance_aux2)
            consistency_dist_aux3 = (preds - pred_train_unsup3) ** 2
            consistency_loss_aux3 = torch.mean(consistency_dist_aux3 * exp_variance_aux3) / (torch.mean(exp_variance_aux3) + 1e-8) + torch.mean(variance_aux3)
            consistency_dist_aux4 = (preds - pred_train_unsup4) ** 2
            consistency_loss_aux4 = torch.mean(consistency_dist_aux4 * exp_variance_aux4) / (torch.mean(exp_variance_aux4) + 1e-8) + torch.mean(variance_aux4)
            loss_train_unsup = (consistency_loss_aux1 + consistency_loss_aux2 + consistency_loss_aux3 + consistency_loss_aux4) / 4

            loss_train_unsup = loss_train_unsup * unsup_weight
            loss_train_unsup.backward(retain_graph=True)
            torch.cuda.empty_cache()

            sup_index = next(dataset_train_sup)
            img_train_sup = sup_index['image']
            img_train_sup = Variable(img_train_sup.cuda(non_blocking=True))
            mask_train_sup = sup_index['mask']
            mask_train_sup = Variable(mask_train_sup.cuda(non_blocking=True))

            pred_train_sup1, pred_train_sup2, pred_train_sup3, pred_train_sup4 = model1(img_train_sup)

            if count_iter % args.display_iter == 0:
                if i == 0:
                    score_list_train1 = pred_train_sup1
                    mask_list_train = mask_train_sup
                # else:
                elif 0 < i <= num_batches['train_sup'] / 32:
                    score_list_train1 = torch.cat((score_list_train1, pred_train_sup1), dim=0)
                    mask_list_train = torch.cat((mask_list_train, mask_train_sup), dim=0)

            loss_train_sup1 = (criterion(pred_train_sup1, mask_train_sup)+criterion(pred_train_sup2, mask_train_sup)+criterion(pred_train_sup3, mask_train_sup)+criterion(pred_train_sup4, mask_train_sup)) / 4
            loss_train_sup = loss_train_sup1
            loss_train_sup.backward()

            optimizer1.step()
            torch.cuda.empty_cache()

            loss_train = loss_train_unsup + loss_train_sup
            train_loss_unsup += loss_train_unsup.item()
            train_loss_sup_1 += loss_train_sup1.item()
            train_loss += loss_train.item()

        scheduler_warmup1.step()
        torch.cuda.empty_cache()

        if count_iter % args.display_iter == 0:

            score_gather_list_train1 = [torch.zeros_like(score_list_train1) for _ in range(ngpus_per_node)]
            torch.distributed.all_gather(score_gather_list_train1, score_list_train1)
            score_list_train1 = torch.cat(score_gather_list_train1, dim=0)

            mask_gather_list_train = [torch.zeros_like(mask_list_train) for _ in range(ngpus_per_node)]
            torch.distributed.all_gather(mask_gather_list_train, mask_list_train)
            mask_list_train = torch.cat(mask_gather_list_train, dim=0)

            if rank == args.rank_index:
                torch.cuda.empty_cache()
                print('=' * print_num)
                print('| Epoch {}/{}'.format(epoch + 1, args.num_epochs).ljust(print_num_minus, ' '), '|')
                train_epoch_loss_sup1, train_epoch_loss_cps, train_epoch_loss = print_train_loss_EM(train_loss_sup_1, train_loss_unsup, train_loss, num_batches, print_num, print_num_minus)
                train_eval_list1, train_m_jc1 = print_train_eval_sup(cfg['NUM_CLASSES'], score_list_train1, mask_list_train, print_num_minus)
                torch.cuda.empty_cache()

            with torch.no_grad():
                model1.eval()

                for i, data in enumerate(dataloaders['val']):

                    # if 0 <= i <= num_batches['val'] / 16:

                    inputs_val = Variable(data['image'].cuda(non_blocking=True))
                    mask_val = Variable(data['mask'].cuda(non_blocking=True))
                    name_val = data['ID']

                    optimizer1.zero_grad()
                    outputs_val1, outputs_val2, outputs_val3, outputs_val4 = model1(inputs_val)
                    torch.cuda.empty_cache()

                    if i == 0:
                        score_list_val1 = outputs_val1
                        mask_list_val = mask_val
                        name_list_val = name_val
                    else:
                        score_list_val1 = torch.cat((score_list_val1, outputs_val1), dim=0)
                        mask_list_val = torch.cat((mask_list_val, mask_val), dim=0)
                        name_list_val = np.append(name_list_val, name_val, axis=0)

                    loss_val_sup1 = criterion(outputs_val1, mask_val)
                    val_loss_sup_1 += loss_val_sup1.item()

                torch.cuda.empty_cache()
                score_gather_list_val1 = [torch.zeros_like(score_list_val1) for _ in range(ngpus_per_node)]
                torch.distributed.all_gather(score_gather_list_val1, score_list_val1)
                score_list_val1 = torch.cat(score_gather_list_val1, dim=0)

                mask_gather_list_val = [torch.zeros_like(mask_list_val) for _ in range(ngpus_per_node)]
                torch.distributed.all_gather(mask_gather_list_val, mask_list_val)
                mask_list_val = torch.cat(mask_gather_list_val, dim=0)

                name_gather_list_val = [None for _ in range(ngpus_per_node)]
                torch.distributed.all_gather_object(name_gather_list_val, name_list_val)
                name_list_val = np.concatenate(name_gather_list_val, axis=0)

                if rank == args.rank_index:
                    val_epoch_loss_sup1 = print_val_loss_sup(val_loss_sup_1, num_batches, print_num, print_num_minus)
                    val_eval_list1, val_m_jc1 = print_val_eval_sup(cfg['NUM_CLASSES'], score_list_val1, mask_list_val, print_num_minus)
                    best_val_eval_list = save_val_best_sup_2d(cfg['NUM_CLASSES'], best_val_eval_list, model1, score_list_val1, name_list_val, val_eval_list1, path_trained_models, path_seg_results, cfg['PALETTE'], 'URPC')
                    torch.cuda.empty_cache()

                    if args.vis:
                        draw_img = draw_pred_sup(cfg['NUM_CLASSES'], mask_train_sup, mask_val, pred_train_sup1, outputs_val1, train_eval_list1, val_eval_list1)
                        visualization_EM(visdom, epoch + 1, train_epoch_loss, train_epoch_loss_sup1, train_epoch_loss_cps, train_m_jc1, val_epoch_loss_sup1, val_m_jc1)
                        visual_image_sup(visdom, draw_img[0], draw_img[1], draw_img[2], draw_img[3])

                    print('-' * print_num)
                    print('| Epoch Time: {:.4f}s'.format((time.time() - begin_time) / args.display_iter).ljust(print_num_minus, ' '), '|')
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()

    if rank == args.rank_index:
        time_elapsed = time.time() - since
        m, s = divmod(time_elapsed, 60)
        h, m = divmod(m, 60)

        print('=' * print_num)
        print('| Training Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
        print('-' * print_num)
        print_best_sup(cfg['NUM_CLASSES'], best_val_eval_list, print_num_minus)
        print('=' * print_num)