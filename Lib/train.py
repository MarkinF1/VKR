import os
import shutil
import argparse
import torch as t
import tqdm
import gc
from torch.utils.data import DataLoader
from torchnet.meter import AverageValueMeter
from data.voc_dataset import VOCDataset
from data.heridal_dataset import HRDLDataset
from models.faster_rcnn_base import LossTuple
from models.faster_rcnn import FasterRCNN
from models.feature_pyramid_network import FPN
from Lib.utils.config import opt
from Lib.utils.eval_tool import evaluate
import Lib.utils.array_tool as at

"""
class Object:
    def __init__(self, name, pose, trunc, difficult, xmin, xmax, ymin, ymax):
        self.name = name
        self.pose = pose
        self.trunc = trunc
        self.difficult = difficult
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def print(self):
        print("name = ", self.name)
        print("pose = ", self.pose)
        print("trunc = ", self.trunc)
        print("dif = ", self.difficult)
        print("xmin = ", self.xmin)
        print("xmax = ", self.xmax)
        print("ymin = ", self.ymin)
        print("ymax = ", self.ymax)


class Label:
    def __init__(self, filename, width, height, depth, arr):
        self.filename = filename
        self.width = width
        self.height = height
        self.depth = depth
        self.objects = arr

    def print(self):
        print("filename = ", self.filename)
        print("w = ", self.width)
        print("h = ", self.height)
        print("d = ", self.depth)
        count = 1
        for i in self.objects:
            print("-" * 40)
            print("Object", count)
            count += 1
            i.print()


def parseXML(xml_file, print_data=0):
    tree = ET.ElementTree(file=xml_file)
    root = tree.getroot()
    filename, width, height, depth = "", 0, 0, 0
    arr = []
    for child in root:
        if child.tag == "filename":
            filename = child.text
        elif child.tag == "size":
            for step in child:
                if step.tag == "width":
                    width = step.text
                elif step.tag == "height":
                    height = step.text
                elif step.tag == "depth":
                    depth = step.text
        elif child.tag == "object":
            name, pose, trunc, difficult, xmin, xmax, ymin, ymax = "", "", 0, 0, 0, 0, 0, 0
            for step in child:
                if step.tag == "name":
                    name = step.text
                elif step.tag == "pose":
                    pose = step.text
                elif step.tag == "truncated":
                    trunc = step.text
                elif step.tag == "difficult":
                    difficult = step.text
                elif step.tag == "bndbox":
                    for steps in step:
                        if steps.tag == "xmin":
                            xmin = steps.text
                        elif steps.tag == "xmax":
                            xmax = steps.text
                        elif steps.tag == "ymin":
                            ymin = steps.text
                        elif steps.tag == "ymax":
                            ymax = steps.text
            arr.append(Object(name, pose, trunc, difficult, xmin, xmax, ymin, ymax))
    out = Label(filename, width, height, depth, arr)

    if print_data:
        print("*" * 40)
        print("*" * 40)
        print("Вывод файла = ", out.filename)
        print("-" * 40)
        out.print()
        print("\n")
"""

def parse_args():

    parser = argparse.ArgumentParser(description='CLI options for training a model.')
    parser.add_argument('--model', type=str, default='fpn', help='Model name: frcnn, fpn (default=fpn).')
    parser.add_argument('--backbone', type=str, default='vgg16', help='Backbone network: vgg16 (default=vgg16).')
    parser.add_argument('--n_features', type=int, default=1, help='The number of features to use for RoI-pooling (default=1).')
    parser.add_argument('--dataset', type=str, default='heridal', help='Training dataset: voc07, heridal (default=heridal).')
    parser.add_argument('--data_dir', type=str, default='C:/Python/Datasets/heridal/test_train_Images/', help='Training dataset directory (default_heridal=C:\Python\Datasets\heridal; default_VOC=C:\Python\Datasets\VOC\\train).')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/best_save', help='Saving directory (default=./Models/best_model_heridal).')
    parser.add_argument('--min_size', type=int, default=500, help='Minimum input image size (default=600).')
    parser.add_argument('--max_size', type=int, default=500, help='Maximum input image size (default=1000).')
    parser.add_argument('--n_workers_train', type=int, default=2, help='The number of workers for a train loader (default=2).')
    parser.add_argument('--n_workers_test', type=int, default=2, help='The number of workers for a test loader (default=2).')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default=1e-3).')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='Learning rate decay (default=0.1; 1e-3 -> 1e-4).')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (default=5e-4).')
    parser.add_argument('--epoch', type=int, default=16, help='Total epochs (default=15).')
    parser.add_argument('--epoch_decay', type=int, default=10, help='The epoch to decay learning rate (default=10).')
    parser.add_argument('--nms_thresh', type=int, default=0.3, help='IoU threshold for NMS (default=0.3).')
    parser.add_argument('--score_thresh', type=int, default=0.05, help='BBoxes with scores less than this are excluded (default=0.05).')
    args = parser.parse_args()
    return args


def get_optimizer(model): # оптимизация для разных параметров, скорость обучения для bias быстрее в 2 раза
    lr = opt.lr
    params = []
    for k, v in dict(model.named_parameters()).items():
        if v.requires_grad:
            if 'bias' in k:
                params += [{'params': [v], 'lr': lr * 2, 'weight_decay': 0}]
            else:
                params += [{'params': [v], 'lr': lr, 'weight_decay': opt.weight_decay}]

    return t.optim.SGD(params, momentum=0.9)


def reset_meters(meters):
    for key, meter in meters.items():
        meter.reset()


def update_meters(meters, losses):
    loss_d = {k: at.scalar(v) for k, v, in losses._asdict().items()}
    for key, meter in meters.items():
        meter.add(loss_d[key])


def get_meter_data(meters):
    return {k: v.value()[0] for k, v in meters.items()}


def save_model(model, model_name, epoch):
    save_path = f'./checkpoints/{model_name}/{epoch}.pth'
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    t.save(model.state_dict(), save_path)

    return save_path


def save_checkpoint(path, epoch, model, optimizer, save_path, best_map, best_path, type_model='frcnn'):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer,
             'save_path': save_path,
             'best_map': best_map,
             'best_path': best_path}
    filename = path + '/checkpoint_' + type_model + '.pth.tar'
    t.save(state, filename)


if __name__ == "__main__":
    """
    contain_of_labels = []
    for root, dirs, files in os.walk('C:\Python\Datasets\heridal\\trainImages\labels'):
        for name in files:
            contain_of_labels.append(root + "\\" + name)

    for label in contain_of_labels:
        parseXML(label, 1)

    print("*" * 40)
    print("Все данные по фотографиям загружены, начинается загрузка фотографий")
    print("*" * 40)

#--------------------------------------------------------------------------------------

    """

    checkpoint = "C:\Python\python_projects\Diplom\Lib\checkpoints\\fpn\\checkpoint_fpn.pth.tar"

    opt._parse(vars(parse_args()))

    t.multiprocessing.set_sharing_strategy('file_system')

    if opt.dataset == 'voc07':
        n_fg_class = 20
        train_data = [VOCDataset(opt.data_dir + '/VOCdevkit/VOC2007', 'trainval', 'train')]
        test_data = VOCDataset(opt.data_dir + '/VOCdevkit/VOC2007', 'test', 'test', True)
    elif opt.dataset == 'heridal':
        n_fg_class = 20
        train_data = [HRDLDataset(opt.data_dir, 'trainval', 'train')]
        test_data = HRDLDataset(opt.data_dir, 'test', 'test', True)
    else:
        raise ValueError('Invalid dataset.')

    train_loaders = [DataLoader(dta, 1, True, num_workers=opt.n_workers_train) for dta in train_data]
    test_loader = DataLoader(test_data, 1, False, num_workers=opt.n_workers_test)

    print('Dataset loaded.')

    if checkpoint is None:
        start_epoch = 1
        if opt.model == 'frcnn':
            model = FasterRCNN(n_fg_class).cuda()
            save_path = f'{opt.save_dir}/{opt.model}_{opt.backbone}.pth'
        elif opt.model == 'fpn':
            model = FPN(n_fg_class).cuda()
            save_path = f'{opt.save_dir}/{opt.model}_{opt.backbone}_{opt.n_features}.pth'
        else:
            raise ValueError('Invalid model. It muse be either frcnn or fpn.')

        print('Model construction completed.')

        optim = get_optimizer(model)
        print('Optimizer loaded.')

        i = 1
    else:
        checkpoint = t.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optim = checkpoint['optimizer']
        #save_path = checkpoint['save_path']
        save_path = f'{opt.save_dir}/{opt.model}_{opt.backbone}.pth'
        best_map = checkpoint['best_map']
        best_path = checkpoint['best_path']
        i = 0

    meters = {k: AverageValueMeter() for k in LossTuple._fields} # что-то типо счетчиков

    # Очистка памяти CUDA
    t.cuda.empty_cache()
    gc.collect()

    lr = opt.lr
    model.cuda()

    for e in range(start_epoch, opt.epoch + 1):
        model.train() # устанавливаем модель в режим тренировки
        reset_meters(meters) # сброс счетчиков
        for train_loader in train_loaders:
            for img, bbox, label, scale in tqdm.tqdm(train_loader):# умная штукенция, которая прикольно выводит строку загрузки
                scale = at.scalar(scale) # преобразуем в строку если надо и возвращаем указатель на первый элемент
                img, bbox, label = img.cuda().float(), bbox.cuda(), label.cuda() # перевод всех модулей на GPU
                optim.zero_grad() # обнуление  счетчиков градиента для обучения
                losses = model.forward(img, scale, bbox, label) # запуск модели на обучение вперед, получение ошибок
                losses.total_loss.backward() # запуск в обратную сторону
                optim.step() # шаг оптимизации
                update_meters(meters, losses) # обновление счетчиков и ошибок
                del img, bbox, label, scale

        md = get_meter_data(meters) # возврат в более удобном формате
        """        log = f'Epoch: {e:2}, lr: {str(lr)}, ' + \
              f'rpn_loc_loss: {md["rpn_loc_loss"]:.4f}, rpn_cls_loss: {md["rpn_cls_loss"]:.4f}, ' + \
              f'roi_loc_loss: {md["roi_loc_loss"]:.4f}, roi_cls_loss: {md["roi_cls_loss"]:.4f}, ' + \
              f'total_loss: {md["total_loss"]:.4f}'
        """
        log = f'FPN Epoch: {e:2}, lr: {str(lr)}, total_loss: {md["total_loss"]:.4f}'

        print(log)

        model.eval() # устанавливаем модель в режим оценки

        map = evaluate(test_loader, model)

        # update best mAP
        if i or map['map'] > best_map['map']:
            best_map = map
            best_path = save_model(model, opt.model, e)
            best_save = save_checkpoint("./checkpoints/best_save", e, model, optim, save_path, best_map, best_path, opt.model)

        if e == opt.epoch_decay: # config.epoch_decay = 10, config.epoch = 15???
            state_dict = t.load(best_path)
            model.load_state_dict(state_dict)
            # decay learning rate скорость обучения спада
            for param_group in optim.param_groups:
                param_group['lr'] *= opt.lr_decay # уменьшаем скорость обучения для всех параметров
            lr = lr * opt.lr_decay

        save_checkpoint("./checkpoints/" + opt.model, e, model, optim, save_path, best_map, best_path, opt.model)
        shutil.copyfile(best_path, save_path)

    # save best model to opt.save_dir
    shutil.copyfile(best_path, save_path)
