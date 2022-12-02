import torch
import torch.nn.functional as F  

# def countX(lst, x):
#     count = 0
#     for i in range(224):
#         for j in range(224):
# #             print(lst[i,j])
#             if (lst[i,j] == x):
#                 count = count + 1
#     return count


def f_score(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
#     print("shape:",inputs[0].shape)
    pr = F.softmax(inputs[0].permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
#     print("pr_shape:",pr.shape)
#     print(countX(pr, 0))
#     print(countX(pr, 1))
#     print("max_pr:",pr.max())
#     print("min_pr:",pr.min())
#     print("max:",temp_inputs.max())
#     print("min:",temp_inputs.min())
    temp_target = target.view(n, -1, ct)
#     print(temp_inputs.max())
    #--------------------------------------------#
    #   计算dice系数
    #--------------------------------------------#
    temp_inputs = torch.gt(temp_inputs,threhold).float()
#     print("max1:",temp_inputs.max())
#     print("min1:",temp_inputs.min())
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp
    
#     print('tp:', tp)
#     print('fn:', fn)
#     print('fp:', fp)
    
#     score = ((1 + beta ** 2) * tp ) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
#     print('score length:', score.size())
#     print('score:',score)
    stroke = score[0]
#     print("stroke:",stroke)
    background = score[1]
#     print('background:',background)
#     nonclass = score[2]
#     print('nonclass:', nonclass)
    score = torch.mean(score)
    return score,stroke,background,tp,fp,fn
    # return score, stroke


# def hd(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
#     n, c, h, w = inputs.size()
# #     nt, ht, wt, ct = target.size()
    
#     if h != ht and w != wt:
#         inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
#     temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
# #     temp_target = target.view(n, -1, ct)
#     print(temp_inputs.max())
#     #--------------------------------------------#
#     #   计算dice系数
#     #--------------------------------------------#
#     temp_inputs = torch.gt(temp_inputs,threhold).float()
    
    
#     return hd95

def Sensitivity(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)
#     print(temp_inputs.max())
    #--------------------------------------------#
    #   计算dice系数
    #--------------------------------------------#
    temp_inputs = torch.gt(temp_inputs,threhold).float()
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = (tp + smooth) / ((beta ** 2) * tp + beta ** 2 * fn + smooth)
#     print('score:',score)
    score = torch.mean(score)
    return score


def Precision(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)
#     print(temp_inputs.max())
    #--------------------------------------------#
    #   计算dice系数
    #--------------------------------------------#
    temp_inputs = torch.gt(temp_inputs,threhold).float()
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ( tp + smooth) / ( tp + fp + smooth)
#     print('score:',score)
    score = torch.mean(score[1])
    return score