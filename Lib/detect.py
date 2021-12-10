import torch
from torchvision import transforms
from Lib.utils.utils import *
from PIL import Image, ImageDraw, ImageFont
import torch as t

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
name_model = 'fpn'
colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#d2f53c', '#fabebe',
          '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#e6beff', '#808080']

# Load model checkpoint
checkpoint = './checkpoints/best_save/checkpoint_' + name_model + '.pth.tar'
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((500, 500))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
count = 0


def change_model(name):
    global name_model
    if name == 'fpn' or name =='frcnn':
        name_model = name
    else:
        print("Incorrect model name")
        exit(-1)


def chek(box1, box2):
    x, y = [box1[0], box1[2]], [box1[1], box1[3]]
    xmin, ymin, xmax, ymax = box2[0], box2[1], box2[2], box2[3]
    flag = 0
    if xmin >= x[0] and xmin <= x[1] and ((ymin >= y[0] and ymin <= y[1]) or (ymax >= y[0] and ymax <= y[1])):
            flag = 1
    if xmax >= x[0] and xmax <= x[1] and ((ymin >= y[0] and ymin <= y[1]) or (ymax >= y[0] and ymax <= y[1])):
            flag = 1

    if flag:
        if xmin > x[0]:
            xmin = x[0]
        if xmax < x[1]:
            xmax = x[1]
        if ymin > y[0]:
            ymin = y[0]
        if ymax < y[1]:
            ymax = y[1]

    return xmin, ymin, xmax, ymax, flag


def resize_image(image, overlap_x=100, overlap_y=200, width=500, height=500, save_dir=None, name='image'):
    new_images = []
    x = []
    y = []
    x_step = 0
    wid = image.width
    hei = image.height
    while x_step < wid:
        x.append([x_step, x_step + width if x_step + width < wid else wid])
        x_step += width - overlap_x
    y_step = 0
    while y_step < hei:
        y.append([y_step, y_step + height if y_step + height < hei else hei])
        y_step += 500 - overlap_y

    for i in range(len(y)):
        images = []
        for k in range(len(x)):
            im_crop = image.crop((x[k][0], y[i][0], x[k][1], y[i][1]))  # xmin, ymin, xmax, ymax
            images.append(im_crop)
            if save_dir is not None:
                im_crop.save(save_dir + name + '_' + str(i * 10 + k + 1) + '.jpg', quality=95)
        new_images.append(images)

    return new_images, x, y


def detect(original_image, num_of_files=None):
    global count
    print("*" * 40)
    count += 1
    print("IMAGE", count) if num_of_files is None else print("IMAGE", str(count) + '/' + str(num_of_files))
    print("*" * 40)
    images, x, y = resize_image(original_image)
    boxes = []
    for i in range(len(y)):
        for k in range(len(x)):
            print("Part of image:", str(i * 10 + k + 1) + '/' + str(len(y) * len(x)))
            # Transform
            image = normalize(to_tensor(resize(images[i][k])))

            # Move to default device
            image = image.to(device)

            # Forward prop.
            det_boxes, det_labels, det_scores = model(image.unsqueeze(0))
            det_boxes = t.from_numpy(det_boxes)

            # Move detections to the CPU
            det_boxes = det_boxes.to('cpu')

            # Decode class integer labels
            det_labels = ['human' if l > 0 else 'nohuman' for l in det_labels]

            if det_labels == ['nohuman'] or not len(det_labels):
                continue

            det_boxes = det_boxes.tolist()
            for l in range(len(det_boxes)):
                det_boxes[l][0], det_boxes[l][1] = det_boxes[l][1], det_boxes[l][0]
                det_boxes[l][2], det_boxes[l][3] = det_boxes[l][3], det_boxes[l][2]
                det_boxes[l] = [x + y for x, y in zip(det_boxes[l], [x[k][0], y[i][0], x[k][0], y[i][0]])]
            boxes.extend(det_boxes)

    len_boxes = len(boxes)
    flag = 1
    while flag:
        flag = 0
        for i in range(len_boxes):
            if i >= len_boxes:
                break
            for k in range(i + 1, len_boxes):
                if k >= len_boxes:
                    break
                xi, yi, xa, ya, fl = chek(boxes[i], boxes[k])
                if fl:
                    boxes.pop(k)
                    boxes[i] = [xi, yi, xa, ya]
                    len_boxes -= 1
                    flag = 1

    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibril.ttf", 15)

    for i, box_location in enumerate(boxes):
        # Boxes
        draw.rectangle(xy=box_location, outline=colors[i % len(colors)])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=colors[i % len(colors)])

        # Text
        text_size = font.getsize('Human')
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1],
                            box_location[0] + text_size[0] + 4., box_location[1]]
        draw.rectangle(xy=textbox_location, fill=colors[i % len(colors)])
        draw.text(xy=text_location, text='Human', fill='white', font=font)
    del draw
    return annotated_image, len(boxes)


if __name__ == '__main__':
    img_data_dir = 'C:/Python/python_projects/Diplom/Lib/files for detecting/'
    save_dir = 'C:/Python/python_projects/Diplom/Lib/detected files/'
    list_of_models = ['fpn', 'frcnn']
    change_model(list_of_models[0])
    for root, dirs, files in os.walk(img_data_dir):
        for name in files:
            original_image = Image.open(root + "\\" + name, mode='r')
            original_image = original_image.convert('RGB')
            out_image, num_objects = detect(original_image, len(files))
            out_image.save(save_dir + name_model + '_' + str(num_objects) + '_' + name)
