import os, argparse, time, datetime, stat, shutil
import random
import warnings
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from util.MF_dataset import MF_dataset
from util.augmentation import RandomFlip,RandomCrop
from util.util import compute_results, visualize
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from pytorch_toolbelt import losses as L
from loss_hub.losses import DiceLoss,SoftCrossEntropyLoss,CrossEntropyLoss
from torch.cuda.amp import autocast,GradScaler
from model.FATNet import FAENet_p_p
#############################################################################################
parser = argparse.ArgumentParser(description='Train with pytorch')
#############################################################################################
parser.add_argument('--model_name', '-m', type=str, default='FATNet_pst900_96')
parser.add_argument('--batch_size', '-b', type=int, default=3)
parser.add_argument('--seed', default=3407, type=int,help='seed for initializing training.')
parser.add_argument('--lr_start', '-ls', type=float, default=0.02)
parser.add_argument('--gpu', '-g', type=int, default=0)
#############################################################################################
parser.add_argument('--lr_decay', '-ld', type=float, default=0.95)
parser.add_argument('--epoch_max', '-em', type=int, default=60)
parser.add_argument('--epoch_from', '-ef', type=int, default=0)
parser.add_argument('--num_workers', '-j', type=int, default=4)
parser.add_argument('--n_class', '-nc', type=int, default=5)
parser.add_argument('--loss_weight', '-lw', type=float, default=0.5)
parser.add_argument('--data_dir', '-dr',type=str, default='D:/pst900/pst900_thermal_rgb-master/PST900_RGBT_Dataset/')
args = parser.parse_args()
#############################################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
scaler = GradScaler()

def train(epo, model, train_loader, optimizer):
    model.train()
    for it, (images, thermal, labels, names) in enumerate(train_loader):
        images = Variable(images).cuda(args.gpu)
        thermal = Variable(thermal).cuda(args.gpu)
        labels = Variable(labels).cuda(args.gpu)
        images = torch.cat([images,thermal],dim=1)
        with autocast():
            start_t = time.time()  # time.time() returns the current time
            optimizer.zero_grad()
            DiceLoss_fn = DiceLoss(mode='multiclass')
            SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1)
            criterion = L.JointLoss(first=DiceLoss_fn, second=SoftCrossEntropy_fn, first_weight=0.4,second_weight=0.5).cuda()
            logits_S, logits_T = model(images)
            loss_1 = criterion(logits_S, labels)
            loss_2 = criterion(logits_T, labels)
            loss = loss_1 + loss_2
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        lr_this_epo = 0
        for param_group in optimizer.param_groups:
            lr_this_epo = param_group['lr']
        print('Train: %s, epo %s/%s, iter %s/%s, lr %.8f, %.2f img/sec, loss %.4f, time %s'\
              % (args.model_name, epo, args.epoch_max, it + 1, len(train_loader), lr_this_epo,
                 len(names) / (time.time() - start_t), float(loss),
                 datetime.datetime.now().replace(microsecond=0) - start_datetime))
        if accIter['train'] % 1 == 0:
            writer.add_scalar('Train/loss', loss, accIter['train'])
        view_figure = True  # note that I have not colorized the GT and predictions here
        if accIter['train'] % 500 == 0:
            if view_figure:
                input_rgb_images = vutils.make_grid(images[:, :3], nrow=8,padding=10)  # can only display 3-channel images, so images[:,:3]
                writer.add_image('Train/input_rgb_images', input_rgb_images, accIter['train'])
                scale = max(1,255 // args.n_class)  # label (0,1,2..) is invisable, multiply a constant for visualization
                groundtruth_tensor = labels.unsqueeze(1) * scale  # mini_batch*480*640 -> mini_batch*1*480*640
                groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor),1)  # change to 3-channel for visualization
                groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10)
                writer.add_image('Train/groudtruth_images', groudtruth_images, accIter['train'])
                predicted_tensor = logits_T.argmax(1).unsqueeze(1) * scale  # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor),1)  # change to 3-channel for visualization, mini_batch*1*480*640
                predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10)
                writer.add_image('Train/predicted_images', predicted_images, accIter['train'])
        accIter['train'] = accIter['train'] + 1


def validation(epo, model, val_loader):
    model.eval()
    with torch.no_grad():
        for it, (images, thermal, labels, names) in enumerate(val_loader):
            images = Variable(images).cuda(args.gpu)
            thermal = Variable(thermal).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            images = torch.cat([images, thermal], dim=1)
            start_t = time.time()  # time.time() returns the current time
            DiceLoss_fn = DiceLoss(mode='multiclass')
            SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1)
            criterion = L.JointLoss(first=DiceLoss_fn, second=SoftCrossEntropy_fn, first_weight=0.4,second_weight=0.5).cuda()
            logits_S,logits_T = model(images)
            loss_1 = criterion(logits_S, labels)
            loss_2 = criterion(logits_T, labels)
            loss = loss_1 + loss_2
            print('Val: %s, epo %s/%s, iter %s/%s, %.2f img/sec, loss %.4f, time %s' \
                  % (
                  args.model_name, epo, args.epoch_max, it + 1, len(val_loader), len(names) / (time.time() - start_t),
                  float(loss),
                  datetime.datetime.now().replace(microsecond=0) - start_datetime))
            if accIter['val'] % 1 == 0:
                writer.add_scalar('Validation/loss', loss, accIter['val'])
            view_figure = False  # note that I have not colorized the GT and predictions here
            if accIter['val'] % 100 == 0:
                if view_figure:
                    input_rgb_images = vutils.make_grid(images[:, :3], nrow=8,
                                                        padding=10)  # can only display 3-channel images, so images[:,:3]
                    writer.add_image('Validation/input_rgb_images', input_rgb_images, accIter['val'])
                    scale = max(1,255 // args.n_class)  # label (0,1,2..) is invisable, multiply a constant for visualization
                    groundtruth_tensor = labels.unsqueeze(1) * scale  # mini_batch*480*640 -> mini_batch*1*480*640
                    groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor),1)  # change to 3-channel for visualization
                    groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10)
                    writer.add_image('Validation/groudtruth_images', groudtruth_images, accIter['val'])
                    predicted_tensor = logits_T.argmax(1).unsqueeze(1) * scale  # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                    predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor),1)  # change to 3-channel for visualization, mini_batch*1*480*640
                    predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10)
                    writer.add_image('Validation/predicted_images', predicted_images, accIter['val'])
            accIter['val'] += 1
    return loss


def testing(epo, model, test_loader):
    model.eval()
    conf_total = np.zeros((args.n_class, args.n_class))
    label_list = ["unlabeled", "car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]
    testing_results_file = os.path.join(weight_dir, 'testing_results_file.txt')
    with torch.no_grad():
        for it, (images, thermal, labels, names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            thermal = Variable(thermal).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            images = torch.cat([images, thermal], dim=1)
            #BBS使用双输出
            logit,logits = model(images)
            logit_mix = (logit + logits)
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logit_mix.argmax(1).cpu().numpy().squeeze().flatten()  # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0, 1, 2, 3, 4])  # conf is args.n_class*args.n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
            print('Test: %s, epo %s/%s, iter %s/%s, time %s' % (
            args.model_name, epo, args.epoch_max, it + 1, len(test_loader),
            datetime.datetime.now().replace(microsecond=0) - start_datetime))
    precision, recall, IoU = compute_results(conf_total)
    writer.add_scalar('Test/average_precision', precision.mean(), epo)
    writer.add_scalar('Test/average_recall', recall.mean(), epo)
    writer.add_scalar('Test/average_IoU', IoU.mean(), epo)
    for i in range(len(precision)):
        writer.add_scalar("Test(class)/precision_class_%s" % label_list[i], precision[i], epo)
        writer.add_scalar("Test(class)/recall_class_%s" % label_list[i], recall[i], epo)
        writer.add_scalar('Test(class)/Iou_%s' % label_list[i], IoU[i], epo)
    if epo == 0:
        with open(testing_results_file, 'w') as f:
            f.write("# %s, initial lr: %s, batch size: %s, date: %s \n" % (
            args.model_name, args.lr_start, args.batch_size, datetime.date.today()))
            f.write(
                "# epoch: Background, Hand-Drill, Backpack, Fire-Extinguisher, Survivor, average(nan_to_num). (Acc %, IoU %)\n")
    with open(testing_results_file, 'a') as f:
        f.write(str(epo) + ': ')
        for i in range(len(precision)):
            f.write('%0.4f, %0.4f, ' % (100 * recall[i], 100 * IoU[i]))
        f.write('%0.4f, %0.4f\n' % (100 * np.mean(np.nan_to_num(recall)), 100 * np.mean(np.nan_to_num(IoU))))
    print('saving testing results.')
    with open(testing_results_file, "r") as file:
        writer.add_text('testing_results', file.read().replace('\n', '  \n'), epo)

def testing_s(epo, model, test_loader):
    model.eval()
    conf_total = np.zeros((args.n_class, args.n_class))
    label_list = ["unlabeled", "car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]
    testing_results_file = os.path.join(weight_dir, 'testing_results_s_file.txt')
    with torch.no_grad():
        for it, (images, thermal, labels, names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            thermal = Variable(thermal).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            images = torch.cat([images, thermal], dim=1)
            #BBS使用双输出
            _,logits = model(images)
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(1).cpu().numpy().squeeze().flatten()  # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0, 1, 2, 3, 4])  # conf is args.n_class*args.n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
            print('Test: %s, epo %s/%s, iter %s/%s, time %s' % (
                args.model_name, epo, args.epoch_max, it + 1, len(test_loader),
                datetime.datetime.now().replace(microsecond=0) - start_datetime))
    precision, recall, IoU = compute_results(conf_total)
    writer.add_scalar('Test/average_precision', precision.mean(), epo)
    writer.add_scalar('Test/average_recall', recall.mean(), epo)
    writer.add_scalar('Test/average_IoU', IoU.mean(), epo)
    for i in range(len(precision)):
        writer.add_scalar("Test(class)/precision_class_%s" % label_list[i], precision[i], epo)
        writer.add_scalar("Test(class)/recall_class_%s" % label_list[i], recall[i], epo)
        writer.add_scalar('Test(class)/Iou_%s' % label_list[i], IoU[i], epo)
    if epo == 0:
        with open(testing_results_file, 'w') as f:
            f.write("# %s, initial lr: %s, batch size: %s, date: %s \n" % (
                args.model_name, args.lr_start, args.batch_size, datetime.date.today()))
            f.write(
                "# epoch: Background, Hand-Drill, Backpack, Fire-Extinguisher, Survivor, average(nan_to_num). (Acc %, IoU %)\n")
    with open(testing_results_file, 'a') as f:
        f.write(str(epo) + ': ')
        for i in range(len(precision)):
            f.write('%0.4f, %0.4f, ' % (100 * recall[i], 100 * IoU[i]))
        f.write('%0.4f, %0.4f\n' % (100 * np.mean(np.nan_to_num(recall)), 100 * np.mean(np.nan_to_num(IoU))))
    print('saving testing results.')
    with open(testing_results_file, "r") as file:
        writer.add_text('testing_results', file.read().replace('\n', '  \n'), epo)



def testing_t(epo, model, test_loader):
    model.eval()
    conf_total = np.zeros((args.n_class, args.n_class))
    label_list = ["unlabeled", "car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]
    txt_dir = "/media/sky/E/lmj_model_save/"
    testing_results_file = os.path.join(weight_dir, 'testing_results_t_file.txt')
    with torch.no_grad():
        for it, (images, thermal, labels, names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            thermal = Variable(thermal).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            images = torch.cat([images, thermal], dim=1)
            #BBS使用双输出
            logit,_ = model(images)
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logit.argmax(
                1).cpu().numpy().squeeze().flatten()  # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0, 1, 2, 3,4])  # conf is args.n_class*args.n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
            print('Test: %s, epo %s/%s, iter %s/%s, time %s' % (
                args.model_name, epo, args.epoch_max, it + 1, len(test_loader),
                datetime.datetime.now().replace(microsecond=0) - start_datetime))
        precision, recall, IoU = compute_results(conf_total)
        writer.add_scalar('Test/average_precision', precision.mean(), epo)
        writer.add_scalar('Test/average_recall', recall.mean(), epo)
        writer.add_scalar('Test/average_IoU', IoU.mean(), epo)
        for i in range(len(precision)):
            writer.add_scalar("Test(class)/precision_class_%s" % label_list[i], precision[i], epo)
            writer.add_scalar("Test(class)/recall_class_%s" % label_list[i], recall[i], epo)
            writer.add_scalar('Test(class)/Iou_%s' % label_list[i], IoU[i], epo)
        if epo == 0:
            with open(testing_results_file, 'w') as f:
                f.write("# %s, initial lr: %s, batch size: %s, date: %s \n" % (
                    args.model_name, args.lr_start, args.batch_size, datetime.date.today()))
                f.write(
                    "# epoch: Background, Hand-Drill, Backpack, Fire-Extinguisher, Survivor, average(nan_to_num). (Acc %, IoU %)\n")
        with open(testing_results_file, 'a') as f:
            f.write(str(epo) + ': ')
            for i in range(len(precision)):
                f.write('%0.4f, %0.4f, ' % (100 * recall[i], 100 * IoU[i]))
            f.write('%0.4f, %0.4f\n' % (100 * np.mean(np.nan_to_num(recall)), 100 * np.mean(np.nan_to_num(IoU))))
        print('saving testing results.')
        with open(testing_results_file, "r") as file:
            writer.add_text('testing_results', file.read().replace('\n', '  \n'), epo)


if __name__ == '__main__':
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')
    model = FAENet_p_p.FATNet_pp(args.n_class)
    if args.gpu >= 0: model.cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_start, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay, last_epoch=-1)

    weight_dir = os.path.join("./runs/", args.model_name)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    os.chmod(weight_dir,
             stat.S_IRWXO)  # allow the folder created by docker read, written, and execuated by local machine

    writer = SummaryWriter("./runs/tensorboard_log")
    os.chmod("./runs/tensorboard_log",stat.S_IRWXO)  # allow the folder created by docker read, written, and execuated by local machine
    os.chmod("./runs", stat.S_IRWXO)

    print('training %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    print('from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('weight will be saved in: %s' % weight_dir)

    train_dataset = MF_dataset(data_dir=args.data_dir+"train/", split='train')
    val_dataset = MF_dataset(data_dir=args.data_dir+"test/", split='val')
    test_dataset = MF_dataset(data_dir=args.data_dir+"test/", split='test')

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    start_datetime = datetime.datetime.now().replace(microsecond=0)
    accIter = {'train': 0, 'val': 0}
    for epo in range(args.epoch_from, args.epoch_max):
        print('\ntrain %s, epo #%s begin...' % (args.model_name, epo))
        train(epo, model, train_loader, optimizer)
        loss = validation(epo, model, val_loader)
        checkpoint_model_file = os.path.join(weight_dir, str(epo) + '.pth')
        print('saving check point %s: ' % checkpoint_model_file)
        torch.save(model.state_dict(), checkpoint_model_file)
        testing_s(epo, model, test_loader)
        testing_t(epo, model, test_loader)
        testing(epo, model, test_loader)
        scheduler.step()