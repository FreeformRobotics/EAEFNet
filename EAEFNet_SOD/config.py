import argparse
parser = argparse.ArgumentParser()
# train/val
parser.add_argument('--task', type=str, default='RGBT', help='epoch number')
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
parser.add_argument('--trainsize', type=int, default=224, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--train_root', type=str, default='./RGBT_dataset/train/', help='the train images root')
parser.add_argument('--val_root', type=str, default='./RGBT_dataset/val/', help='the val images root')
parser.add_argument('--save_path', type=str, default='./run/', help='the path to save models and logs')
# test(predict)
parser.add_argument('--testsize', type=int, default=224, help='testing size')
parser.add_argument('--test_path',type=str,default='',help='test dataset path')
opt = parser.parse_args()
