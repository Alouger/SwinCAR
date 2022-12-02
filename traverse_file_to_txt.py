import os
import random
import shutil

random.seed(0)

# segfilepath是原图image的文件夹路径
segfilepath = r'E:/oxford-iiit-pet/processed/train_image'

# seglabelpath是segfilepath的label文件夹路径
# seglabelpath = r'E:/oxford-iiit-pet/processed/annotations'

# 数据集分割好后的txt文件保存路径
saveBasePath = r"E:/oxford-iiit-pet"

# 把写入txt文件的图片名字，对照看是训练集还是验证集，测试集，然后保存到以下路径中
# savetrainimagePath = r'E:/oxford-iiit-pet/processed/train_image'
# savetrainlabelPath = r'E:/oxford-iiit-pet/processed/train_label'
# savevalimagePath = r'E:/oxford-iiit-pet/processed/val_image'
# savevallabelPath = r'E:/oxford-iiit-pet/processed/val_label'
# savetestimagePath = r'E:/oxford-iiit-pet/processed/test_image'
# savetestlabelPath = r'E:/oxford-iiit-pet/processed/test_label'

# ----------------------------------------------------------------------#
#   医药数据集的例子没有验证集
# ----------------------------------------------------------------------#
trainval_percent = 1
train_percent = 1

temp_seg = os.listdir(segfilepath)
total_seg = []
for seg in temp_seg:
    if seg.endswith(".png"):
        total_seg.append(seg)

num = len(total_seg)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

print("train and val size", tv)
print("traub suze", tr)
ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

for i in list:
    name = total_seg[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
            # shutil.copy(segfilepath + '/' + total_seg[i][:-4] + '.png', savetrainimagePath)
            # shutil.copy(seglabelpath + '/' + total_seg[i][:-4] + '.png', savetrainlabelPath)
        else:
            fval.write(name)
            # shutil.copy(segfilepath + '/' + total_seg[i][:-4] + '.png', savevalimagePath)
            # shutil.copy(seglabelpath + '/' + total_seg[i][:-4] + '.png', savevallabelPath)
    else:
        ftest.write(name)
        # shutil.copy(segfilepath + '/' + total_seg[i][:-4] + '.png', savetestimagePath)
        # shutil.copy(seglabelpath + '/' + total_seg[i][:-4] + '.png', savetestlabelPath)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
