import os, argparse, time, datetime, sys, shutil, stat, torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from util.MF_dataset import MF_dataset
from util.util import compute_results, visualize
from sklearn.metrics import confusion_matrix
from model.EAEFNet import EAEFNet
#############################################################################################
parser = argparse.ArgumentParser(description='Test with pytorch')
#############################################################################################
parser.add_argument('--model_name', '-m', type=str, default='EAEFNet')
parser.add_argument('--weight_name', '-w', type=str, default='EAEFNet')
parser.add_argument('--file_name', '-f', type=str, default='best.pth')
parser.add_argument('--dataset_split', '-d', type=str, default='test')  # test, test_day, test_night
parser.add_argument('--gpu', '-g', type=int, default=0)
#############################################################################################
parser.add_argument('--img_height', '-ih', type=int, default=480)
parser.add_argument('--img_width', '-iw', type=int, default=640)
parser.add_argument('--num_workers', '-j', type=int, default=0)
parser.add_argument('--n_class', '-nc', type=int, default=5)
parser.add_argument('--data_dir', '-dr', type=str, default='./test/')
args = parser.parse_args()
#############################################################################################

if __name__ == '__main__':
    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')
    model_dir = os.path.join('./runs/', args.weight_name)
    if os.path.exists(model_dir) is False:
        sys.exit("the %s does not exit." % (model_dir))
    model_file = os.path.join(model_dir, args.file_name)
    if os.path.exists(model_file) is True:
        print('use the final model file.')
    else:
        sys.exit('no model file found.')
    print('testing %s: %s on GPU #%d with pytorch' % (args.model_name, args.weight_name, args.gpu))

    conf_total = np.zeros((args.n_class, args.n_class))
    model = EAEFNet.EAEFNet(args.n_class)
    if args.gpu >= 0: model.cuda(args.gpu)
    print('loading model file %s... ' % model_file)
    pretrained_weight = torch.load(model_file, map_location=lambda storage, loc: storage.cuda(args.gpu))
    own_state = model.state_dict()

    for name, param in pretrained_weight.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)
    print('done!')

    for name, param in pretrained_weight.items():
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param)
    print('done!')

    batch_size = 1  # do not change this parameter!
    test_dataset = MF_dataset(data_dir=args.data_dir, split=args.dataset_split, input_h=args.img_height,
                              input_w=args.img_width)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    ave_time_cost = 0.0

    model.eval()
    with torch.no_grad():
        for it, (images, thermal, labels, names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            thermal = Variable(thermal).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            images = torch.cat([images, thermal], dim=1)
            start_time = time.time()
            logit,logits = model(images)
            end_time = time.time()
            if it >= 5:  # # ignore the first 5 frames
                ave_time_cost += (end_time - start_time)
            # convert tensor to numpy 1d array
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(1).cpu().numpy().squeeze().flatten() # prediction and label are both 1-d array, size: minibatch*640*480
            # generate confusion matrix frame-by-frame
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0, 1, 2, 3, 4])  # conf is an n_class*n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
            # save demo images
            visualize(image_name=names, predictions=logits.argmax(1), weight_name='Pred_' + args.weight_name)
            print("%s, %s, frame %d/%d, %s, time cost: %.2f ms, demo result saved."
                  % (
                      args.model_name, args.weight_name, it + 1, len(test_loader), names,
                      (end_time - start_time) * 1000))

    precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total)
    conf_total_matfile = os.path.join('D:/pst900_thermal_rgb-master/RTFNet/runs/Pred_' + args.weight_name, 'conf_' + args.weight_name + '.mat')

    print('\n###########################################################################')
    print('\n%s: %s test results (with batch size %d) on %s using %s:' % (
        args.model_name, args.weight_name, batch_size, datetime.date.today(), torch.cuda.get_device_name(args.gpu)))
    print('\n* the tested dataset name: %s' % args.dataset_split)
    print('* the tested image count: %d' % len(test_loader))
    print('* the tested image size: %d*%d' % (args.img_height, args.img_width))
    print('* the weight name: %s' % args.weight_name)
    print('* the file name: %s' % args.file_name)
    print(
        "* recall per class: \n   Background: %.6f, Hand - Drill: %.6f, Backpack: %.6f, Fire - Extinguisher: %.6f,  Survivor: %.6f"
        % (recall_per_class[0], recall_per_class[1], recall_per_class[2], recall_per_class[3], recall_per_class[4]))
    print(
        "* iou per class: \n     Background: %.6f, Hand - Drill: %.6f, Backpack: %.6f, Fire - Extinguisher: %.6f,  Survivor: %.6f"  \
        % (iou_per_class[0], iou_per_class[1], iou_per_class[2], iou_per_class[3], iou_per_class[4]))
    print("\n* average values (np.mean(x)): \n recall: %.6f, iou: %.6f" \
          % (recall_per_class.mean(), iou_per_class.mean()))
    print("* average values (np.mean(np.nan_to_num(x))): \n recall: %.6f, iou: %.6f" \
          % (np.mean(np.nan_to_num(recall_per_class)), np.mean(np.nan_to_num(iou_per_class))))
    print(
        '\n* the average time cost per frame (with batch size %d): %.2f ms, namely, the inference speed is %.2f fps' % (
            batch_size, ave_time_cost * 1000 / (len(test_loader) - 5),
            1.0 / (ave_time_cost / (len(test_loader) - 5))))  # ignore the first 10 frames
    # print('\n* the total confusion matrix: ')
    # np.set_printoptions(precision=8, threshold=np.inf, linewidth=np.inf, suppress=True)
    # print(conf_total)
    print('\n###########################################################################')
