import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from plot import train_loss_plot, val_loss_plot, metrics_plot
from nets.unet import Unet
# from Unet_from_UnetZoo import Unet as Unet_from_UnetZoo
from nets.unet_training import CE_Loss, Dice_loss
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.metrics import f_score
# from self_attention_cv.transunet import TransUnet
# from trans_unet2 import TransUnet
# from trans_unet_SE import TransUnet
# from trans_unet_AttentionUnet import TransUnet
# from trans_unet_CA import TransUnet
# from trans_unet_MHCA_CA import TransUnet
# from trans_unet_MHCA_CA_2 import TransUnet
# from UCATR_100 import TransUnet
# from UCATR_010 import TransUnet
# from UCATR_001 import TransUnet
# from UCATR_110 import TransUnet
# from UCATR_011 import TransUnet
# from UCATR_101 import TransUnet
# from UCATR_no_Transformer import TransUnet
# from ResUnet import Unet
# from Attention_Unet_from_UnetZoo import AttU_Net
# from trans_unet import TransUnet
# from swin_transformer_unet import SUnet
from load_swin_transformer import SwinUnet as ViT_seg
from nets.pspnet import PSPNet
from nets.deeplabv3_plus import DeepLab
from logger import get_logger
from torchinfo import summary
import pandas as pd
from collections import OrderedDict
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(net, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):
    total_loss = 0
    total_f_score = 0

    val_toal_loss = 0
    val_total_f_score = 0

    total_score = 0
    total_stroke = 0
    total_background = 0
    total_nonclass = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    val_total_score = 0
    val_total_stroke = 0
    val_total_background = 0
    val_total_nonclass = 0
    val_total_tp = 0
    val_total_fn = 0
    val_total_fp = 0

    net = net.train()
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            imgs, pngs, labels = batch

            #             print(iteration.dtype())

            with torch.no_grad():
                # imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
                # pngs = Variable(torch.from_numpy(pngs).type(torch.FloatTensor)).long()
                # labels = Variable(torch.from_numpy(labels).type(torch.FloatTensor))

                imgs = Variable(imgs)
                pngs = Variable(pngs).long()
                labels = Variable(labels)
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()

            optimizer.zero_grad()

#             ttser = (torch.rand(1, 3, 224, 224),)
#             flops = FlopCountAnalysis(net, ttser)
#             print("FLOPs: ", flops.total())

            outputs = net(imgs)
            #             print("shape:",outputs.shape)
            loss = CE_Loss(outputs, pngs, num_classes=NUM_CLASSES)
            #             print('CE_Loss:',loss)
            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                #                 print('Dice_Loss:',main_dice)
                loss = 0.5 * loss + 0.5 * main_dice
            #                 loss      = main_dice
            #                 loss      = loss + main_dice
            #                 loss      = loss + main_dice

            with torch.no_grad():
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                #                 _f_score = f_score(outputs, labels)
                score, stroke, background, tp, fn, fp = f_score(outputs, labels)
                _f_score = torch.mean(score)
            #                 print(score)
            #                 logger.info('iteration=={:.1f}\t tp={:.8f}\t fn={:.16f}\t fp={:.8f}'.format((iteration+1), tp, fn, fp))

            loss.backward()
            optimizer.step()

            total_score += score
            total_stroke += stroke.item()
            total_background += background.item()
#             total_nonclass = nonclass.item()
            total_tp += tp
            total_fn += fn
            total_fp += fp
            total_loss += loss.item()
            total_f_score += _f_score.item()
            trian_avg_loss = total_loss / (iteration + 1)
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'train_background': total_background/(iteration+1),
#                                 'train_nonclass': total_nonclass/(iteration+1),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

        train_loss_list.append(trian_avg_loss)
        logger.info('Epoch:[{}/{}]\t train loss={:.8f}\t train dice={:.8f}\t lr={:.8f}\n'.format(epoch + 1, Epoch,
                                                                                                total_loss / (
                                                                                                            iteration + 1),
                                                                                                total_f_score / (
                                                                                                            iteration + 1),
                                                                                                get_lr(optimizer)))
        logger.info('train score={}\n'.format(total_score / (iteration + 1)))
        logger.info(
            'train stroke={}\t background={}\t score={}\n'.format(total_stroke / (iteration + 1),
                                                                                        total_background / (iteration + 1),
                                                                                        total_f_score / (iteration + 1)))
#         logger.info('****************   ****************')
#         logger.info('*              *   *              *')
#         logger.info('*    Stroke    *   *  background  *')
#         logger.info('*              *   *              *')
#         logger.info('****************   ****************')
#         logger.info('*              *   *              *')
#         logger.info(
#             '* {:.4}\t *   * {:.4}\t *'.format(total_stroke / (iteration + 1), total_background / (iteration + 1)))
#         logger.info('*              *   *              *')
#         logger.info('****************   ****************')

#         log = OrderedDict([
#             ('train_loss', total_loss / (iteration + 1)),
#             ('train_stroke_dice', total_stroke / (iteration + 1)),
#             ('train_background_dice', total_background / (iteration + 1)),
#             ('train_dice', total_f_score / (iteration + 1)),
#         ])

    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            imgs, pngs, labels = batch

            with torch.no_grad():
                imgs = Variable(imgs)
                pngs = Variable(pngs).long()
                labels = Variable(labels)
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()

                outputs = net(imgs)
                val_loss = CE_Loss(outputs, pngs, num_classes=NUM_CLASSES)
                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    val_loss = 0.5 * val_loss + 0.5 * main_dice
                #                     val_oss = main_dice
                #                     val_loss = val_loss + main_dice

                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                score, stroke, background, tp, fn, fp = f_score(outputs, labels)
                _f_score = torch.mean(score)
                #                 print(score)
                #                 print(_f_score)
                #                 logger.info('iteration=={:.1f}\t tp={:.8f}\t fn={:.16f}\t fp={:.8f}'.format((iteration+1), tp, fn, fp))

                val_toal_loss += val_loss.item()
                val_total_f_score += _f_score.item()
                val_total_stroke += stroke.item()
                val_total_background += background.item()
#                 val_total_nonclass += nonclass.item()

            val_total_score += score
            # val_total_stroke += stroke
            # val_total_background += background
            val_total_tp += tp
            val_total_fn += fn
            val_total_fp += fp
            Total_loss = val_toal_loss / (iteration + 1)
            dice_score = val_total_f_score / (iteration + 1)
            stroke_dice = val_total_stroke / (iteration+1)
            #             print((iteration + 1))
            pbar.set_postfix(**{'total_loss': val_toal_loss / (iteration + 1),
                                'val_background': val_total_background / (iteration + 1),
#                                 'val_nonclass': val_total_nonclass / (iteration + 1),
                                'f_score': val_total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
        #             logger.info('iteration={}\t score={}\t tp={}\t fn={}\t fp={}'.format(iteration, score, tp, fn, fp))

        val_loss_list.append(Total_loss)
        #         print(loss_list)
        dice_list.append(dice_score)
        logger.info(
            'Epoch:[{}/{}]\t iteration:{} val loss={:.8f}\t val dice={:.8f}\t lr={:.8f}'.format(epoch + 1, Epoch,
                                                                                                 iteration + 1,
                                                                                                 val_toal_loss / (
                                                                                                             iteration + 1),
                                                                                                 val_total_f_score / (
                                                                                                             iteration + 1),
                                                                                                 get_lr(optimizer)))
        # logger.info(
        #     'score={}\t tp={}\t fn={}\t fp={}'.format(val_total_score / (iteration + 1), val_total_tp / (iteration + 1),
        #                                               val_total_fn / (iteration + 1), val_total_fp / (iteration + 1)))
        logger.info(
            'val stroke={}\t background={}\t score={}\n'.format(val_total_stroke / (iteration+1), val_total_background /(iteration+1), val_total_score / (iteration + 1)))
#         logger.info('****************   ****************')
#         logger.info('*              *   *              *')
#         logger.info('*    Stroke    *   *  background  *')
#         logger.info('*              *   *              *')
#         logger.info('****************   ****************')
#         logger.info('*              *   *              *')
#         logger.info('* {:.4}\t *   * {:.4}\t *'.format(val_total_stroke / (iteration+1), val_total_background / (iteration+1)))
#         logger.info('*              *   *              *')

#         log = OrderedDict([
#             ('val_loss', val_toal_loss / (iteration + 1)),
#             ('val_stroke_dice', val_total_stroke / (iteration+1)),
#             ('val_background_dice', val_total_background /(iteration+1)),
#             ('val_dice', val_total_f_score / (iteration + 1)),
#         ])


    net.train()
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f' % (total_loss / (epoch_size + 1)))
    print('Saving state, iter:', str(epoch + 1))
    torch.save(model.state_dict(), 'logs/Epoch%d-train_loss%.4f-val_Loss%.4f-valStrokeDice_%.4f-valDice_%.4f.pth' % (
    (epoch + 1), train_loss_list[len(train_loss_list) - 1], Total_loss, stroke_dice, dice_score))

#     return log


if __name__ == "__main__":
    device = torch.device('cuda:0')
    log_dir = "logs/"
    # ------------------------------#
    #   输入图片的大小
    # ------------------------------#
    inputs_size = [224, 224, 3]
    # ---------------------#
    #   分类个数+1
    #   背景+边缘
    # ---------------------#
    NUM_CLASSES = 2
    # --------------------------------------------------------------------#
    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    # ---------------------------------------------------------------------#
    dice_loss = True
    # -------------------------------#
    #   主干网络预训练权重的使用
    # -------------------------------#
    pretrained = False
    # -------------------------------#
    #   Cuda的使用
    # -------------------------------#
    Cuda = True

    # -------------------------------#
    #   plot.py中图片命名设定
    #   Epoch在最下面
    # -------------------------------#
    BATCH_SIZE = 1
    dataset = 'oxford-iiit-pet'
    Model = 'SwinUnet'

    pretrain_path = 'pretrain/swin_tiny_patch4_window7_224.pth'

    #     获取model
#     model = Unet(num_classes=NUM_CLASSES, in_channels=inputs_size[-1], pretrained=pretrained).train()
    
#     model = PSPNet(num_classes=NUM_CLASSES, backbone="resnet50", downsample_factor=16, pretrained=False, aux_branch=False)
#     model = DeepLab(num_classes=NUM_CLASSES, backbone="xception", downsample_factor=16, pretrained=False)

    #     model = Unet_from_UnetZoo(3,NUM_CLASSES).train()

    #     model = Unet(3,NUM_CLASSES).train()
    #     model = TransUnet(in_channels=3, img_dim=224, vit_blocks=12, vit_heads=12 ,vit_dim_linear_mhsa_block=3072, classes=3)
    model = ViT_seg(img_size=224, num_classes=NUM_CLASSES)
    model.load_from(pretrain_path)
    #     model = SUnet(hidden_dim=192, layers=(2, 2, 18, 2), heads=(6, 12, 24, 48),num_classes=NUM_CLASSES,channels=inputs_size[-1], dropout=0.2).train()

    # model = AttU_Net(3,NUM_CLASSES).train()


    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # 打开数据集的txt
    with open(r"/openbayes/home/CV_Project/Medical_Datasets/ImageSets/cv_project/train.txt", "r") as f:
        train_lines = f.readlines()
    # 打开数据集的txt
    with open(r"/openbayes/home/CV_Project/Medical_Datasets/ImageSets/cv_project/val.txt", "r") as f:
        val_lines = f.readlines()

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'train_loss', 'train_stroke_dice', 'train_background_dice', 'train_dice', 'val_loss', 'val_stroke_dice', 'val_background_dice', 'val_dice'
    ])
    stroke_dice_list = []
    train_loss_list = []
    val_loss_list = []
    dice_list = []
    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Interval_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        lr = 1e-4
        Init_Epoch = 0
        Interval_Epoch = 100
        Batch_size = 8

        #         optimizer = optim.SGD(model.parameters(),lr, momentum=0.9, weight_decay=0.005)
        #         optimizer = optim.Adam(model.parameters(),lr, weight_decay=0.0001)
        optimizer = optim.AdamW(model.parameters(), lr, weight_decay=0.01)
        #         lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.92)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 32, eta_min=7e-6, last_epoch=-1)
        #         lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=9,T_mult=1,eta_min=7e-6,last_epoch=-1)

        train_dataset = DeeplabDataset(train_lines, inputs_size, NUM_CLASSES, False,
                                       dataset_path = '/openbayes/input/input3')
        val_dataset = DeeplabDataset(val_lines, inputs_size, NUM_CLASSES, False,
                                     dataset_path = '/openbayes/input/input3')
        gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=deeplab_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=False, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=deeplab_dataset_collate)

        epoch_size = max(1, len(train_lines) // Batch_size)
        epoch_size_val = max(1, len(val_lines) // Batch_size)
        #         for param in model.vgg.parameters():
        #             param.requires_grad = False

        logger = get_logger('save_log/train_log.log')
        logger.info(summary(model, input_size=(Batch_size,3,224,224)))
        logger.info('start training!')
        for epoch in range(Init_Epoch, Interval_Epoch):
            trainval_log = fit_one_epoch(model, epoch, epoch_size, epoch_size_val, gen, gen_val, Interval_Epoch, Cuda)
            lr_scheduler.step()
#             tmp = pd.Series([
#                 epoch,
#                 get_lr(optimizer),
#                 trainval_log['train_loss'],
#                 trainval_log['train_stroke_dice'],
#                 trainval_log['train_background_dice'],
#                 trainval_log['train_dice'],
#                 trainval_log['val_loss'],
#                 trainval_log['val_stroke_dice'],
#                 trainval_log['val_background_dice'],
#                 trainval_log['val_dice'],
#             ], index=['epoch', 'lr', 'train_loss', 'train_stroke_dice', 'train_background_dice', 'train_dice', 'val_loss', 'val_stroke_dice', 'val_background_dice', 'val_dice'])
#             log = log.append(tmp, ignore_index=True)
#             log.to_csv('logs/log.csv', index=False)

        #         print(loss_list)
        logger.info('finish training!')
        train_loss_plot(Interval_Epoch, Model, BATCH_SIZE, dataset, train_loss_list)
        val_loss_plot(Interval_Epoch, Model, BATCH_SIZE, dataset, val_loss_list)
        metrics_plot(Interval_Epoch, Model, BATCH_SIZE, dataset, 'trianloss&valloss', train_loss_list, val_loss_list)
        metrics_plot(Interval_Epoch, Model, BATCH_SIZE, dataset, 'dice', dice_list)

    if True:
        lr = 1e-4
        Interval_Epoch = 150
        Epoch = 150
        Batch_size = 1

        optimizer = optim.Adam(model.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset = DeeplabDataset(train_lines, inputs_size, NUM_CLASSES, True)
        val_dataset = DeeplabDataset(val_lines, inputs_size, NUM_CLASSES, False)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=deeplab_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=deeplab_dataset_collate)
        epoch_size = max(1, len(train_lines) // Batch_size)
        epoch_size_val = max(1, len(val_lines) // Batch_size)
        #         for param in model.vgg.parameters():
        #             param.requires_grad = True

        for epoch in range(Interval_Epoch, Epoch):
            fit_one_epoch(model, epoch, epoch_size, epoch_size_val, gen, gen_val, Epoch, Cuda)
            lr_scheduler.step()

