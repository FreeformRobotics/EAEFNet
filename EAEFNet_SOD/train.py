import os
import torch
import torch.nn.functional as F

import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from config import opt
from torch.cuda import amp
# set the device for training
cudnn.benchmark = True
cudnn.enabled = True


os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU:', opt.gpu_id)

# build the model
#from LSNet import LSNet
from EAEFNet_50 import FATNet_pp
model = FATNet_pp(n_class=1)

if (opt.load is not None):
    model.load_state_dict(torch.load(opt.load))
    print('load model from ', opt.load)
model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

# set the path
train_dataset_path = opt.train_root

val_dataset_path = opt.val_root

save_path = opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')
if opt.task =='RGBT':
    from rgbt_dataset import get_loader, test_dataset
    image_root = train_dataset_path  + '/RGB/'
    ti_root = train_dataset_path  + '/T/'
    gt_root = train_dataset_path  + '/GT/'
    val_image_root = val_dataset_path + '/RGB/'
    val_ti_root = val_dataset_path + '/T/'
    val_gt_root = val_dataset_path + '/GT/'
elif opt.task == 'RGBD':
    from rgbd_dataset import get_loader, test_dataset
    image_root = train_dataset_path + '/RGB/'
    ti_root = train_dataset_path + '/depth/'
    gt_root = train_dataset_path + '/GT/'
    val_image_root = val_dataset_path + '/RGB/'
    val_ti_root = val_dataset_path + '/depth/'
    val_gt_root = val_dataset_path + '/GT/'
else:
    raise ValueError(f"Unknown task type {opt.task}")

train_loader = get_loader(image_root, gt_root, ti_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(val_image_root, val_gt_root,val_ti_root, opt.trainsize)
total_step = len(train_loader)
# print(total_step)

logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("Model:")
logging.info(model)

logging.info(save_path + "Train")
logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

# set loss function
import torch.nn as nn

class IOUBCE_loss(nn.Module):
    def __init__(self):
        super(IOUBCE_loss, self).__init__()
        self.nll_lose = nn.BCEWithLogitsLoss()

    def forward(self, input_scale, taeget_scale):
        b,_,_,_ = input_scale.size()
        loss = []
        for inputs, targets in zip(input_scale, taeget_scale):
            bce = self.nll_lose(inputs,targets)
            pred = torch.sigmoid(inputs)
            inter = (pred * targets).sum(dim=(1, 2))
            union = (pred + targets).sum(dim=(1, 2))
            IOU = (inter + 1) / (union - inter + 1)
            loss.append(1- IOU + bce)
        total_loss = sum(loss)
        return total_loss / b


CE = torch.nn.BCEWithLogitsLoss().cuda()
IOUBCE = IOUBCE_loss().cuda()
class IOUBCEWithoutLogits_loss(nn.Module):
    def __init__(self):
        super(IOUBCEWithoutLogits_loss, self).__init__()
        self.nll_lose = nn.BCELoss()

    def forward(self, input_scale, target_scale):
        b,c,h,w = input_scale.size()
        loss = []
        for inputs, targets in zip(input_scale, target_scale):

            bce = self.nll_lose(inputs,targets)

            inter = (inputs * targets).sum(dim=(1, 2))
            union = (inputs + targets).sum(dim=(1, 2))
            IOU = (inter + 1) / (union - inter + 1)
            loss.append(1- IOU + bce)
        total_loss = sum(loss)
        return total_loss / b

IOUBCEWithoutLogits = IOUBCEWithoutLogits_loss().cuda()

step = 0
writer = SummaryWriter(save_path + 'summary', flush_secs = 30)
best_mae = 1
best_epoch = 0
Sacler = amp.GradScaler()

# BBA
def tesnor_bound(img, ksize):

    '''
    :param img: tensor, B*C*H*W
    :param ksize: tensor, ksize * ksize
    :param 2patches: tensor, B * C * H * W * ksize * ksize
    :return: tensor, (inflation - corrosion), B * C * H * W
    '''

    B, C, H, W = img.shape
    pad = int((ksize - 1) // 2)
    img_pad = F.pad(img, pad=[pad, pad, pad, pad], mode='constant',value = 0)
    # unfold in the second and third dimensions
    patches = img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    corrosion, _ = torch.min(patches.contiguous().view(B, C, H, W, -1), dim=-1)
    inflation, _ = torch.max(patches.contiguous().view(B, C, H, W, -1), dim=-1)
    return inflation - corrosion



# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, tis) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.cuda()
            tis = tis.cuda()
            gts = gts.cuda()
            if opt.task == 'RGBD':
                tis = torch.cat((tis, tis, tis), dim=1)

            bound = tesnor_bound(gts, 3).cuda()

            out_1,out_2 = model(images, tis)
            loss_sod_1 = IOUBCE(out_1, gts)
            loss_sod_2 = IOUBCE(out_2, gts)
            loss_sod = loss_sod_1 + loss_sod_2

            predict_bound0 = out_1
            predict_bound0 = tesnor_bound(torch.sigmoid(predict_bound0), 3)
            loss_bound_1 = IOUBCEWithoutLogits(predict_bound0, bound)
            ###############################################################
            predict_bound1 = out_2
            predict_bound1 = tesnor_bound(torch.sigmoid(predict_bound1), 3)
            loss_bound_2 = IOUBCEWithoutLogits(predict_bound1, bound)
            loss_bound = loss_bound_1 + loss_bound_2

            loss = loss_sod + loss_bound
            loss_trans = loss

            loss.backward()
            optimizer.step()
            step = step + 1
            epoch_step = epoch_step + 1
            loss_all = loss.item() + loss_all
            if i % 10 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}, loss_sod: {:.4f},'
                      'loss_bound: {:.4f},loss_trans: {:.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.item(),
                             loss_sod.item(),loss_bound.item(), loss_trans.item()))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}, loss_sod: {:.4f},'
                              'loss_bound: {:.4f},loss_trans: {:.4f} '.
                             format(epoch, opt.epoch, i, total_step, loss.item(),
                                    loss_sod.item(),loss_bound.item(), loss_trans.item()))
                writer.add_scalar('Loss', loss, global_step=step)
                # grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                # writer.add_image('train/RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('train/Ground_truth', grid_image, step)
                grid_image = make_grid(bound[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('train/bound', grid_image, step)

                # grid_image = make_grid(body[0].clone().cpu().data, 1, normalize=True)
                # writer.add_image('train/body', grid_image, step)
                out = out_1 + out_2
                res = out[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('OUT/out', torch.tensor(res), step, dataformats='HW')
                res = predict_bound0[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('OUT/bound', torch.tensor(res), step, dataformats='HW')


        loss_all /= epoch_step
        # logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise


# test function
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, ti, name = test_loader.load_data()
            gt = gt.cuda()
            image = image.cuda()
            ti = ti.cuda()
            if opt.task == 'RGBD':
                tis = torch.cat((tis, tis, tis), dim=1)

            res_1,res_2 = model(image, ti)
            res = res_1 + res_2
            res = torch.sigmoid(res)
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_train = torch.sum(torch.abs(res - gt)) * 1.0 / (torch.numel(gt))
            # print(mae_train)
            mae_sum = mae_train.item() + mae_sum
        # print(test_loader.size)
        mae = mae_sum / test_loader.size
        # print(test_loader.size)
        writer.add_scalar('MAE', torch.as_tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch+1):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        test(test_loader, model, epoch, save_path)
