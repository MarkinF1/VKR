import torch

from utils.utils import *
from datasets import PascalVOCDataset
from tqdm import tqdm
from Lib.data.heridal_dataset import HRDLDataset
from pprint import PrettyPrinter
import statistics as st

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
data_folder = 'C:/Python/Datasets/heridal/test_train_Images/'
keep_difficult = True
batch_size = 2
workers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = './checkpoints/best_save/checkpoint_fpn.pth.tar'

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)

# Load test data
test_dataset = HRDLDataset(data_folder, 'test', 'test', use_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, 1, False, num_workers=workers)


def chek_crossing(a, b):
    if a[0] >= b[2] or a[1] >= b[3]:
        return 0, 0
    if b[0] <= a[0] <= b[2]:
        if b[0] <= a[2] <= b[2]:
            if a[1] >= b[3]:
                return 0, 0
            if b[1] <= a[1] <= b[3]:
                if b[1] <= a[3] <= b[3]:
                    return 1, (a[2] - a[0]) * (a[3] - a[1])
                return 1, (a[2] - a[0]) * (b[3] - a[1])
            if a[3] <= b[1]:
                return 0, 0
            if b[1] <= a[3] <= b[3]:
                return 1, (a[2] - a[0]) * (a[3] - b[1])
            if a[3] >= b[3]:
                return 1, (a[2] - a[0]) * (b[3] - b[1])
            return 0, 0
        if b[1] <= a[1] <= b[3]:
            if b[1] <= a[3] <= b[3]:
                return 1, (b[2] - a[0]) * (a[3] - a[1])
            return 1, (b[2] - a[0]) * (b[3] - a[1])
        if a[1] <= b[1] <= a[3] <= b[3]:
            return 1, (b[2] - a[0]) * (a[3] - b[1])
        if a[1] <= b[1] and a[3] >= b[3]:
            return 1, (b[2] - a[0]) * (b[3] - b[1])
        return 0, 0
    if a[2] <= b[0]:
        return 0, 0
    if a[2] >= b[2]:
        if a[1] >= b[3]:
            return 0, 0
        if b[1] <= a[1] <= b[3]:
            if b[1] <= a[3] <= b[3]:
                return 1, (b[2] - b[0]) * (a[3] - a[1])
            return 1, (b[2] - b[0]) * (b[3] - a[1])
        if b[1] <= a[3] <= b[3]:
            return 1, (b[2] - b[0]) * (a[3] - b[1])
        if a[3] >= b[3]:
            return 1, (b[2] - b[0]) * (b[3] - b[1])
        return 0, 0
    if b[1] <= a[1] <= a[3] <= b[3]:
        return 1, (a[2] - b[0]) * (a[3] - a[1])
    if a[1] <= b[1] <= a[3] <= b[3]:
        return 1, (a[2] - b[0]) * (a[3] - b[1])
    if b[1] <= a[1] <= b[3] <= a[3]:
        return 1, (a[2] - b[0]) * (b[3] - a[1])
    if a[1] <= b[1] <= b[3] <= a[3]:
        return 1, (a[2] - b[0]) * (b[3] - b[1])
    return 0, 0


def evaluate(test_loader, model):

    model.eval()

    with torch.no_grad():
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for images, boxes, labels, difficulties in tqdm(test_loader):

            images = images.to(device)

            det_boxes_batch, det_labels_batch, det_scores_batch = model(images)
            det_boxes_batch = torch.from_numpy(det_boxes_batch)

            true = 0
            mean_boxes_square = st.mean([((k[2] - k[0]) * (k[3] - k[1])).tolist() for k in boxes[0]])
            for i in boxes[0]:
                for k in det_boxes_batch:
                    fl1, sq1 = chek_crossing(i, k)
                    fl2, sq2 = chek_crossing(k, i)
                    if fl1 == 0 and fl2 == 0:
                        continue
                    if sq2 > sq1:
                        sq = sq2
                    else:
                        sq = sq1
                    if sq >= 0.4 * (i[2] - i[0]) * (i[3] - i[1]):
                        true += 1
                        break

            true_negative += 500 * 500/mean_boxes_square - true - (len(boxes[0]) - true) - (len(det_boxes_batch) - true)
            true_positive += true
            false_negative += len(boxes[0]) - true
            false_positive += len(det_boxes_batch) - true

    print("*" * 40)
    print("FPN:")
    print("*" * 40)
    print([true_positive, false_positive])
    print([false_negative, true_negative], '\n')

    recall = true_positive / (true_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)

    print("Recall =", recall)
    print("Precision =", precision)


if __name__ == '__main__':
    evaluate(test_loader, model)