from torchvision import transforms, datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import os
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.backends import cudnn
import random
import torchio as tio

from config.dataset_config.dataset_cfg import dataset_cfg
from config.train_test_config.train_test_config import print_train_loss_XNet, print_val_loss, print_train_eval_XNet, print_val_eval, save_val_best_3d, print_best
from config.visdom_config.visual_visdom import visdom_initialization_XNet, visualization_XNet
from config.warmup_config.warmup import GradualWarmupScheduler
from config.augmentation.online_aug import data_transform_3d
from loss.loss_function import segmentation_loss
from models.getnetwork import get_network
from dataload.dataset_3d import dataset_iit
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
    parser.add_argument('--path_trained_models', default='/mnt/data1/XNet/checkpoints/sup_xnet')
    parser.add_argument('--path_seg_results', default='/mnt/data1/XNet/seg_pred/sup_xnet')
    parser.add_argument('--path_dataset', default='/mnt/data1/XNet/dataset/Atrial')
    parser.add_argument('--dataset_name', default='Atrial', help='LiTS, Atrial')
    parser.add_argument('--input1', default='L')
    parser.add_argument('--input2', default='H')
    parser.add_argument('--sup_mark', default='100', help='100')
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    parser.add_argument('-e', '--num_epochs', default=200, type=int)
    parser.add_argument('-s', '--step_size', default=50, type=int)
    parser.add_argument('-l', '--lr', default=0.5, type=float)
    parser.add_argument('-g', '--gamma', default=0.5, type=float)
    parser.add_argument('-u', '--unsup_weight', default=5, type=float)
    parser.add_argument('--loss', default='dice', type=str)
    parser.add_argument('--patch_size', default=(96, 96, 80))
    parser.add_argument('-w', '--warm_up_duration', default=20)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow')
    parser.add_argument('--queue_length', default=48, type=int)
    parser.add_argument('--samples_per_volume_train', default=4, type=int)
    parser.add_argument('--samples_per_volume_val', default=8, type=int)

    parser.add_argument('-i', '--display_iter', default=5, type=int)
    parser.add_argument('-n', '--network', default='xnet3d', type=str)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--rank_index', default=0, help='0, 1, 2, 3')
    parser.add_argument('-v', '--vis', default=True, help='need visualization or not')
    parser.add_argument('--visdom_port', default=16672, help='16672')
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

    path_trained_models = args.path_trained_models + '/' + str(os.path.split(args.path_dataset)[1])
    if not os.path.exists(path_trained_models) and rank == args.rank_index:
        os.mkdir(path_trained_models)
    path_trained_models = path_trained_models+'/'+str(args.network)+'-l='+str(args.lr)+'-e='+str(args.num_epochs)+'-s='+str(args.step_size)+'-g='+str(args.gamma)+'-b='+str(args.batch_size)+'-cw='+str(args.unsup_weight)+'-w='+str(args.warm_up_duration)+'-'+str(args.sup_mark)+str(args.input1)+'-'+str(args.input2)
    if not os.path.exists(path_trained_models) and rank == args.rank_index:
        os.mkdir(path_trained_models)

    path_seg_results = args.path_seg_results + '/' + str(os.path.split(args.path_dataset)[1])
    if not os.path.exists(path_seg_results) and rank == args.rank_index:
        os.mkdir(path_seg_results)
    path_seg_results = path_seg_results+'/'+str(args.network)+'-l='+str(args.lr)+'-e='+str(args.num_epochs)+'-s='+str(args.step_size)+'-g='+str(args.gamma)+'-b='+str(args.batch_size)+'-cw='+str(args.unsup_weight)+'-w='+str(args.warm_up_duration)+'-'+str(args.sup_mark)+str(args.input1)+'-'+str(args.input2)
    if not os.path.exists(path_seg_results) and rank == args.rank_index:
        os.mkdir(path_seg_results)
    path_mask_results = path_seg_results + '/mask'
    if not os.path.exists(path_mask_results) and rank == args.rank_index:
        os.mkdir(path_mask_results)
    path_seg_results_1 = path_seg_results + '/pred'
    if not os.path.exists(path_seg_results_1) and rank == args.rank_index:
        os.mkdir(path_seg_results_1)

    if args.vis and rank == args.rank_index:
        visdom_env = str('Sup-XNet-'+str(os.path.split(args.path_dataset)[1])+'-'+args.network+'-l='+str(args.lr)+'-e='+str(args.num_epochs)+'-s='+str(args.step_size)+'-g='+str(args.gamma)+'-b='+str(args.batch_size)+'-cw='+str(args.unsup_weight)+'-w='+str(args.warm_up_duration)+'-'+str(args.sup_mark))+str(args.input1)+'-'+str(args.input2)
        visdom = visdom_initialization_XNet(env=visdom_env, port=args.visdom_port)

    # Dataset
    data_transform = data_transform_3d(cfg['NORMALIZE'])

    dataset_train_sup = dataset_iit(
        data_dir=args.path_dataset + '/train_sup_' + args.sup_mark,
        input1=args.input1,
        input2=args.input2,
        transform_1=data_transform['train'],
        queue_length=args.queue_length,
        samples_per_volume=args.samples_per_volume_train,
        patch_size=args.patch_size,
        num_workers=8,
        shuffle_subjects=True,
        shuffle_patches=True,
        sup=True,
        num_images=None
    )
    dataset_val = dataset_iit(
        data_dir=args.path_dataset + '/val',
        input1=args.input1,
        input2=args.input2,
        transform_1=data_transform['val'],
        queue_length=args.queue_length,
        samples_per_volume=args.samples_per_volume_val,
        patch_size=args.patch_size,
        num_workers=8,
        shuffle_subjects=False,
        shuffle_patches=False,
        sup=True,
        num_images=None
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train_sup.queue_train_set_1, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val.queue_train_set_1, shuffle=False)


    dataloaders = dict()
    dataloaders['train'] = DataLoader(dataset_train_sup.queue_train_set_1, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0, sampler=train_sampler)
    dataloaders['val'] = DataLoader(dataset_val.queue_train_set_1, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0, sampler=val_sampler)

    num_batches = {'train_sup': len(dataloaders['train']), 'val': len(dataloaders['val'])}

    # Model
    model = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])
    model = model.cuda()
    model = DistributedDataParallel(model, device_ids=[args.local_rank])

    # Training Strategy
    criterion = segmentation_loss(args.loss, False).cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5*10**args.wd)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=args.warm_up_duration, after_scheduler=exp_lr_scheduler)

    # Train & Val
    since = time.time()
    count_iter = 0

    best_model = model
    best_result = 'Result1'
    best_val_eval_list = [0 for i in range(4)]

    for epoch in range(args.num_epochs):

        count_iter += 1
        if (count_iter-1) % args.display_iter == 0:
            begin_time = time.time()

        dataloaders['train'].sampler.set_epoch(epoch)
        model.train()

        train_loss_sup_1 = 0.0
        train_loss_sup_2 = 0.0
        train_loss_unsup = 0.0
        train_loss = 0.0
        val_loss_sup_1 = 0.0
        val_loss_sup_2 = 0.0

        unsup_weight = args.unsup_weight * (epoch + 1) / args.num_epochs

        dist.barrier()
        for i, data in enumerate(dataloaders['train']):

            inputs_train_1 = Variable(data['image'][tio.DATA].cuda())
            inputs_train_2 = Variable(data['image2'][tio.DATA].cuda())
            mask_train = Variable(data['mask'][tio.DATA].squeeze(1).long().cuda())

            optimizer.zero_grad()
            outputs_train_1, outputs_train_2 = model(inputs_train_1, inputs_train_2)
            torch.cuda.empty_cache()

            if count_iter % args.display_iter == 0:
                if i == 0:
                    score_list_train_1 = outputs_train_1
                    score_list_train_2 = outputs_train_2
                    mask_list_train = mask_train
                # else:
                elif 0 < i <= num_batches['train_sup'] / 32:
                    score_list_train_1 = torch.cat((score_list_train_1, outputs_train_1), dim=0)
                    score_list_train_2 = torch.cat((score_list_train_2, outputs_train_2), dim=0)
                    mask_list_train = torch.cat((mask_list_train, mask_train), dim=0)

            max_train1 = torch.max(outputs_train_1, dim=1)[1]
            max_train2 = torch.max(outputs_train_2, dim=1)[1]
            max_train1 = max_train1.long()
            max_train2 = max_train2.long()

            loss_train_sup1 = criterion(outputs_train_1, mask_train)
            loss_train_sup2 = criterion(outputs_train_2, mask_train)
            loss_train_unsup = criterion(outputs_train_1, max_train2) + criterion(outputs_train_2, max_train1)
            loss_train_unsup = loss_train_unsup * unsup_weight
            loss_train = loss_train_sup1 + loss_train_sup2 + loss_train_unsup

            loss_train.backward()
            optimizer.step()

            train_loss_sup_1 += loss_train_sup1.item()
            train_loss_sup_2 += loss_train_sup2.item()
            train_loss_unsup += loss_train_unsup.item()
            train_loss += loss_train.item()

        scheduler_warmup.step()
        torch.cuda.empty_cache()

        if count_iter % args.display_iter == 0:

            score_gather_list_train_1 = [torch.zeros_like(score_list_train_1) for _ in range(ngpus_per_node)]
            torch.distributed.all_gather(score_gather_list_train_1, score_list_train_1)
            score_list_train_1 = torch.cat(score_gather_list_train_1, dim=0)

            score_gather_list_train_2 = [torch.zeros_like(score_list_train_2) for _ in range(ngpus_per_node)]
            torch.distributed.all_gather(score_gather_list_train_2, score_list_train_2)
            score_list_train_2 = torch.cat(score_gather_list_train_2, dim=0)

            mask_gather_list_train = [torch.zeros_like(mask_list_train) for _ in range(ngpus_per_node)]
            torch.distributed.all_gather(mask_gather_list_train, mask_list_train)
            mask_list_train = torch.cat(mask_gather_list_train, dim=0)

            if rank == args.rank_index:
                torch.cuda.empty_cache()
                print('=' * print_num)
                print('| Epoch {}/{}'.format(epoch + 1, args.num_epochs).ljust(print_num_minus, ' '), '|')
                train_epoch_loss_sup_1, train_epoch_loss_sup_2, train_epoch_loss_cps, train_epoch_loss = print_train_loss_XNet(train_loss_sup_1, train_loss_sup_2, train_loss_unsup, train_loss, num_batches, print_num, print_num_half)
                train_eval_list_1, train_eval_list_2, train_m_jc_1, train_m_jc_2 = print_train_eval_XNet(cfg['NUM_CLASSES'], score_list_train_1, score_list_train_2, mask_list_train, print_num_half)
                torch.cuda.empty_cache()

            with torch.no_grad():
                model.eval()

                for i, data in enumerate(dataloaders['val']):

                    # if 0 <= i <= num_batches['val']:

                    inputs_val_1 = Variable(data['image'][tio.DATA].cuda())
                    inputs_val_2 = Variable(data['image2'][tio.DATA].cuda())
                    mask_val = Variable(data['mask'][tio.DATA].squeeze(1).long().cuda())

                    optimizer.zero_grad()
                    outputs_val_1, outputs_val_2 = model(inputs_val_1, inputs_val_2)
                    torch.cuda.empty_cache()

                    if i == 0:
                        score_list_val_1 = outputs_val_1
                        score_list_val_2 = outputs_val_2
                        mask_list_val = mask_val
                    else:
                        score_list_val_1 = torch.cat((score_list_val_1, outputs_val_1), dim=0)
                        score_list_val_2 = torch.cat((score_list_val_2, outputs_val_2), dim=0)
                        mask_list_val = torch.cat((mask_list_val, mask_val), dim=0)

                    loss_val_sup_1 = criterion(outputs_val_1, mask_val)
                    loss_val_sup_2 = criterion(outputs_val_2, mask_val)

                    val_loss_sup_1 += loss_val_sup_1.item()
                    val_loss_sup_2 += loss_val_sup_2.item()

                torch.cuda.empty_cache()
                score_gather_list_val_1 = [torch.zeros_like(score_list_val_1) for _ in range(ngpus_per_node)]
                torch.distributed.all_gather(score_gather_list_val_1, score_list_val_1)
                score_list_val_1 = torch.cat(score_gather_list_val_1, dim=0)

                score_gather_list_val_2 = [torch.zeros_like(score_list_val_2) for _ in range(ngpus_per_node)]
                torch.distributed.all_gather(score_gather_list_val_2, score_list_val_2)
                score_list_val_2 = torch.cat(score_gather_list_val_2, dim=0)

                mask_gather_list_val = [torch.zeros_like(mask_list_val) for _ in range(ngpus_per_node)]
                torch.distributed.all_gather(mask_gather_list_val, mask_list_val)
                mask_list_val = torch.cat(mask_gather_list_val, dim=0)
                torch.cuda.empty_cache()

                if rank == args.rank_index:
                    val_epoch_loss_sup_1, val_epoch_loss_sup_2 = print_val_loss(val_loss_sup_1, val_loss_sup_2, num_batches, print_num, print_num_half)
                    val_eval_list_1, val_eval_list_2, val_m_jc_1, val_m_jc_2 = print_val_eval(cfg['NUM_CLASSES'], score_list_val_1, score_list_val_2, mask_list_val, print_num_half)
                    best_val_eval_list, best_model, best_result = save_val_best_3d(cfg['NUM_CLASSES'], best_model, best_val_eval_list, best_result, model, model, score_list_val_1, score_list_val_2, mask_list_val, val_eval_list_1, val_eval_list_2, path_trained_models, path_seg_results, path_mask_results, cfg['FORMAT'])
                    torch.cuda.empty_cache()

                    if args.vis:
                        visualization_XNet(visdom, epoch+1, train_epoch_loss, train_epoch_loss_sup_1, train_epoch_loss_sup_2, train_epoch_loss_cps, train_m_jc_1, train_m_jc_2, val_epoch_loss_sup_1, val_epoch_loss_sup_2, val_m_jc_1, val_m_jc_2)

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
        print_best(cfg['NUM_CLASSES'], best_val_eval_list, best_model, best_result, path_trained_models, print_num_minus)
        print('=' * print_num)