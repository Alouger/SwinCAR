import time
import colorsys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from torchvision import transforms
import os
import cv2
from medpy import metric
from metric_hd import get_hd
# from nets.unet import Unet
from nets.unet_training import CE_Loss, Dice_loss
from utils.test_dataloader_medical import DeeplabDataset, deeplab_dataset_collate
from utils.metrics import f_score
from utils.metrics import Sensitivity
from utils.metrics import Precision
from utils.hd import hd95
# from vit_seg_modeling import VisionTransformer as ViT_seg
# from vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
# from utils.dataloader_liver import RandomGenerator
# from self_attention_cv.transunet import TransUnet
# from trans_unet2 import TransUnet
# from nets.unet import Unet
import matplotlib.pyplot as plt
# from trans_unet2 import TransUnet
# from trans_unet_MHCA_CA import TransUnet
from Attention_Unet_from_UnetZoo import AttU_Net
# from UCATR_100 import TransUnet
# from UCATR_010 import TransUnet
# from UCATR_001 import TransUnet
# from UCATR_110 import TransUnet
# from UCATR_011 import TransUnet
# from UCATR_101 import TransUnet
# from UCATR_no_Transformer import TransUnet
from plot import test_metrics_plot
from Unet_from_UnetZoo import Unet as Unet_from_UnetZoo
# from trans_unet import TransUnet
# from swin_transformer_unet import SUnet
from load_swin_transformer import SwinUnet as ViT_seg
from logger import get_logger
from monai.metrics import HausdorffDistanceMetric
from hausdorff import hausdorff_distance
import xlwt
import xlrd
from xlutils.copy import copy

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(net, genval, cuda,filesname):

    val_total_f_score = 0
    val_total_sensitivity = 0
    val_total_precision = 0
    val_total_hd = 0
    val_total_sc = 0
    val_total_sp = 0
    val_total_jc = 0
    val_total_stroke = 0
    val_total_background = 0
    val_total_nonclass = 0

    i = 0
    net.eval()
    print('Start Validation')
    with tqdm(total=1478, desc=f'Epoch {0 + 1}/{1}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            imgs, pngs, labels,imgs_path,label_path = batch
            # orininal_h = imgs.shape[0]
            # orininal_w = imgs.shape[1]
            # imgs1 = (imgs.shape[0]+1)//2*255
            # m = imgs1.shape
            # print(m)
            image = np.squeeze(imgs, 0)
            # print(image.shape)
            image = image.transpose(1, 2, 0)
            # print(image.shape)
            image = Image.fromarray(np.uint8(image))
            # label = Image.fromarray(labels)
            # image, nw, nh = letterbox_image(image, (model_image_size[1], model_image_size[0]))

            # images = [np.array(image) / 255]
            # images = np.transpose(images, (0, 3, 1, 2))

            # print('imgs.shape:',imgs.size)
            with torch.no_grad():
                imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
                # images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                pngs = Variable(torch.from_numpy(pngs).type(torch.FloatTensor)).long()
                labels = Variable(torch.from_numpy(labels).type(torch.FloatTensor))
                if cuda:
                    # images = images.cuda()
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()

                # outputs = net(imgs).cuda()
                outputs = net(imgs)
#                 print(outputs.max())
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                _f_score, stroke_score, background,tp,fp,fn = f_score(outputs, labels)
                sensitivity = Sensitivity(outputs,labels)
                precision = Precision(outputs,labels)
                # hd = hd95(outputs,labels)
#                 _hd = hd95(outputs,labels)
#                 hd95 = hd(outputs,cv2.imread(label_path[0])
#                 print(_f_score)
                i +=1
                print(str(imgs_path),':',stroke_score)

                # sheet.write(i,0, str(imgs_path).split('/')[-1])
                # sheet.write(i,4,stroke_score.numpy().item())

                val_total_f_score += _f_score.item() if (_f_score != 0) else None
                val_total_sensitivity += sensitivity.item() if (sensitivity != 0) else None
                val_total_precision += precision.item() if (precision != 0) else None
                predict = torch.squeeze(outputs,0).cpu().numpy()
                # print(predict.shape)
                predict = predict.transpose(1, 2, 0)
                # print(predict.shape)
                predict = Image.fromarray(np.uint8(predict))
                # print('outputs:',outputs.shape)
                pr = torch.squeeze(outputs,0)
                pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
                # pr = F.softmax(pr, dim=-1).cpu().numpy().argmax(axis=-1)

                # print('pr:', pr.shape)
                # pr = pr[int((model_image_size[0] - nh) // 2):int((model_image_size[0] - nh) // 2 + nh),
                #      int((model_image_size[1] - nw) // 2):int((model_image_size[1] - nw) // 2 + nw)]
                val_total_stroke += stroke_score.item()
                val_total_background += background.item()
                # val_total_nonclass += nonclass.item()
                stroke_dice_list.append(stroke_score)
                background_dice_list.append(background)
                # nonclass_dice_list.append(nonclass)

            seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # print('seg_img:',seg_img.shape)
            for c in range(num_classes):
                seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
                seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
                seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
            # print('seg_img:', seg_img.dtype)
            # r_image = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h))
            # r_image = Image.fromarray(np.uint8(seg_img)).resize((650,650))
            
            r_image = Image.fromarray(np.uint8(seg_img))
            r_image = r_image.convert('L')
            r_image = np.array(r_image)
            r_image[r_image == 0] = 1
            r_image[r_image == 255] = 0

            # r_image = r_image.astype(np.uint8)
            # tmp_r_image = np.zeros(r_image.shape)
            label = cv2.imread(label_path[0])
            # label1 = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
#             Hausdorff1 = hd95(seg_img,label)
#             print(seg_img.min())
#             print(seg_img.max())
#             if seg_img.max() != 0:
#                 Hausdorff1=metric.hd95(r_image,label1)
#                 print("Hausdorff_1: "+str(Hausdorff1))
#             Hausdorff_2 = get_hd(r_image,label1)
#             print("Hausdorff_2: "+str(Hausdorff_2))
#             val_total_hd += Hausdorff_2 if (Hausdorff_2 != 0) else None
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.set_title('input')
            plt.imshow(Image.open(imgs_path[0]))
#             # # print(pic_path[0])
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.set_title('predict')
#             # plt.imshow(r_image, cmap='Greys_r')
            print(imgs_path[0])
            cv2.imwrite('result/plot/tempout1/' + imgs_path[0].split('/')[-1], r_image)
            plt.imshow(r_image, cmap='Greys_r')
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.set_title('mask')
            plt.imshow(Image.open(label_path[0]), cmap='Greys_r')
            mm = Image.open(label_path[0])
            mm = np.array(mm)
            if cv2.countNonZero(r_image) != 0:
                hd = metric.hd(r_image,mm)
                jc = metric.jc(r_image,mm)
                sc = metric.sensitivity(r_image,mm)
                sp = metric.specificity(r_image,mm)

            print('hd:',hd)
            val_total_hd += hd
            val_total_sc += sc
            val_total_sp += sp
            val_total_jc += jc
            print('jc:', jc)
            print('sc:', sc)
            print('sp:', sp)
            plt.savefig('result/plot/tempout1/savefig_' + imgs_path[0].split('/')[-1], bbox_inches='tight', pad_inches=0.1, dpi=300)
            # plt.show()
            average_dice_score = val_total_f_score / (iteration+1)
            average_sensitivity = val_total_sensitivity / (iteration+1)
            average_precision = val_total_precision / (iteration+1)
            average_stroke_dice = val_total_stroke / (iteration+1)
            average_backgroun = val_total_background / (iteration + 1)
            # average_nonclass = val_total_nonclass / (iteration + 1)
            average_hd = val_total_hd / (iteration+1)
            average_sc = val_total_sc / (iteration+1)
            average_sp = val_total_sp / (iteration + 1)
            average_jc = val_total_jc / (iteration + 1)
#             print(filesname,' average_dice:',average_dice_score)
#             print(filesname,' average_sensitivity:',average_sensitivity)
#             print(filesname,' average_precision:',average_precision)
#             print(filesname,' average_hd:',average_hd)
            pbar.set_postfix(**{'f_score': val_total_f_score / (iteration + 1)})
#             pbar.set_postfix(**{'sensitivity': val_total_sensitivity / (iteration + 1)})
#             pbar.set_postfix(**{'precision': val_total_precision / (iteration + 1)})
            pbar.update(1)

    dice_list.append(average_dice_score)
    sensitivity_list.append(average_sensitivity)
    precision_list.append(average_precision)
#     hd_list.append(average_hd)
    # print('i:',i)
    logger.info('filesname={}\t average_dice={:.16f}\t average_sensitivity={:.16f}\t average_precision={:.16f}'.format(filesname, average_dice_score, average_sensitivity, average_precision ))
    print(filesname,' average_dice:',average_dice_score)
    print(filesname,' average_sensitivity:',average_sensitivity)
    print(filesname,' average_precision:',average_precision)
    print(filesname,' average_stroke_dice:',average_stroke_dice)
    print(filesname,' average_backgroun:',average_backgroun)
    # print(filesname, ' average_nonclass:', average_nonclass)
    print(filesname, ' average_hd:', average_hd)
    print(filesname, ' average_sc:', average_sc)
    print(filesname, ' average_sp:', average_sp)
    print(filesname, ' average_jc:', average_jc)
#     print(filesname,' average_hd:',average_hd)
    # print('Finish Validation')


if __name__ == "__main__":
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
    Cuda = False

    # -------------------------------#
    #   plot.py中图片命名设定
    #   Epoch在最下面
    # -------------------------------#
    BATCH_SIZE = 1
    dataset = 'BrainCT'
    Model = 'SwinUnet'


    if Cuda:
#         net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
#         net = net.cuda()

    # 打开数据集的txt
#     with open("/openbayes/home/Unet_Bubbliiiing/Medical_Datasets/ImageSets/StrokeCT/test.txt", "r") as f:
    with open("/openbayes/home/CV_Project/Medical_Datasets/ImageSets/cv_project/test.txt", "r") as f:
#     with open("/openbayes/home/Unet_Bubbliiiing/Medical_Datasets/ImageSets/Abdomen_dataset/val.txt", "r") as f:
        val_lines = f.readlines()

    stroke_dice_list = []
    background_dice_list = []
    nonclass_dice_list = []
    loss_list = []
    dice_list = []
    sensitivity_list = []
    precision_list = []
    hd_list = []
    logger = get_logger('save_log/test_log.log')
    logger.info('start testing!')
    for root, dirs,files in os.walk('/openbayes/home/CV_Project/logs'):
#         for filesname in sorted(files,key=lambda x: str(x)):
        for filesname in sorted(files):
#             print(filesname)
            # model_path = 'logs/BrainCT_TransUnet_MHCA_SGD_weight_decay_0.001_Epoch110/Epoch56-Total_Loss0.3912-valDice_0.4857.pth'
            model_path = '/openbayes/home/CV_Project/logs/'+filesname
            model_image_size= [224, 224, 3]
            num_classes = 2
            cuda = False
            # --------------------------------#
            #   blend参数用于控制是否
            #   让识别结果和原图混合
            # --------------------------------#
            blend= False


            # ---------------------------------------------------#
            #   获得所有的分类
            # ---------------------------------------------------#
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            # net = Unet(num_classes=num_classes, in_channels=model_image_size[-1]).eval()
#             net = TransUnet(in_channels=3, img_dim=224, vit_blocks=12, vit_heads=12 ,vit_dim_linear_mhsa_block=3072, classes=2).eval()
#             net = SUnet(hidden_dim=128, layers=(2, 2, 18, 2), heads=(4, 8, 16, 32),num_classes=2).eval()
            net = ViT_seg(img_size=224, num_classes=NUM_CLASSES).eval()
#             net = Unet_from_UnetZoo(3,NUM_CLASSES).eval()
#             net = AttU_Net(3,NUM_CLASSES).eval()

            # map_location = torch.device('cpu')
            state_dict = torch.load(model_path, map_location='cpu')
            net.load_state_dict(state_dict)

            if cuda:
                net = nn.DataParallel(net)
                cudnn.benchmark = True
                net = net.cuda()

            print('{} model loaded.'.format(model_path))

            if num_classes == 2:
                colors = [(255, 255, 255), (0, 0, 0)]
            elif num_classes == 3:
                colors = [(255,255,255), (0, 0, 0), (128, 128, 128)]
            elif num_classes <= 21:
                colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                                (0, 128, 128),
                                (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
                                (192, 0, 128),
                                (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                                (0, 64, 128), (128, 64, 12)]
            # else:
            #     # 画框设置不同的颜色
            #     hsv_tuples = [(x / len(class_names), 1., 1.)
            #                     for x in range(len(class_names))]
            #     colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            #     colors = list(
            #         map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            #             colors))


            def letterbox_image(image, size):
                image = image.convert("RGB")
                iw, ih = image.size
                w, h = size
                scale = min(w / iw, h / ih)
                nw = int(iw * scale)
                nh = int(ih * scale)

                image = image.resize((nw, nh), Image.BICUBIC)
                new_image = Image.new('RGB', size, (128, 128, 128))
                new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
                return new_image, nw, nh


            # ---------------------------------------------------#
            #   检测图片
            # ---------------------------------------------------#

            # ------------------------------------------------------#
            #   主干特征提取网络特征通用，冻结训练可以加快训练速度
            #   也可以在训练初期防止权值被破坏。
            #   Init_Epoch为起始世代
            #   Interval_Epoch为冻结训练的世代
            #   Epoch总训练世代
            #   提示OOM或者显存不足请调小Batch_size
            # ------------------------------------------------------#

            # lr = 1e-4
            # Init_Epoch = 0
            # Interval_Epoch = 50
            # Batch_size = 1

            # optimizer = optim.Adam(net.parameters(), lr)
            # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

            # train_dataset = DeeplabDataset(train_lines, inputs_size, NUM_CLASSES, True)
            val_dataset = DeeplabDataset(val_lines, inputs_size, NUM_CLASSES, False, dataset_path = '/openbayes/input/input3')
            # gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=1, pin_memory=True,
            #                  drop_last=True, collate_fn=deeplab_dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=1, collate_fn=deeplab_dataset_collate)

            # epoch_size = max(1, len(train_lines) // Batch_size)
            epoch_size_val = max(1, len(val_lines) // 1)

#             for param in net.vgg.parameters():
#                 param.requires_grad = False

            # for epoch in range(Init_Epoch, Interval_Epoch):
            # book = xlrd.open_workbook('F:/Pytorch_Code/Unet_Bubbliiiing.output/Output.xls',True)
            # book2 = copy(book)
            # sheet = book2.get_sheet(0)
            # sheet.write(0, 0, 'No')  # 其中的'0-行, 0-列'指定表中的单元，'X'是向该单元写入的内容
            # sheet.write(0, 4, 'Attention U-Net')
            fit_one_epoch(net , gen_val, Cuda,filesname)
            # book2.save('Output.xls')
        # lr_scheduler.step()
    logger.info('finish testing!')
    # test_metrics_plot(30, Model, BATCH_SIZE, dataset, 'dice', stroke_dice_list)
    print(np.mean(stroke_dice_list))
    print(np.mean(background_dice_list))
    # print(np.mean(nonclass_dice_list))
    print(np.mean(dice_list))
    print(np.mean(sensitivity_list))
    print(np.mean(precision_list))
    print(np.mean(hd_list))
    



