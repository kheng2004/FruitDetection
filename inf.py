import torch
import argparse
import os
import yaml
import random
from tqdm import tqdm
from ssd.model.ssd import SSD
import numpy as np
import cv2
from ssd.dataset.voc import VOCDataset
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as T
from torchvision.io import read_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')



def load_model_and_dataset(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    voc = VOCDataset('test',
                     im_sets=dataset_config['test_im_sets'])
    test_dataset = DataLoader(voc, batch_size=1, shuffle=False)

    model = SSD(config=model_config,
                num_classes=dataset_config['num_classes'])
    model.to(device=torch.device(device))
    model.eval()

    assert os.path.exists(os.path.join(train_config['task_name'],
                                       train_config['ckpt_name'])), \
        "No checkpoint exists at {}".format(os.path.join(train_config['task_name'],
                                                         train_config['ckpt_name']))
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                       train_config['ckpt_name']),
                                     map_location=device))
    return model, voc, test_dataset, config

im_size = 300
im_mean = [123.0, 117.0, 104.0]
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

tranforms = T.v2.Compose([
                T.v2.Resize(size=(im_size, im_size)),
                T.v2.ToPureTensor(),
                T.v2.ToDtype(torch.float32, scale=True),
                T.v2.Normalize(mean=imagenet_mean,
                                    std=imagenet_std)])

def infer(args):
    model, voc, test_dataset, config = load_model_and_dataset(args)
    conf_threshold = config['train_params']['infer_conf_threshold']
    model.low_score_threshold = conf_threshold

    im = read_image("test.jpg")
    im = tranforms(im)
    # im_tensor, target, fname = voc[140]
    fname = "test.jpg"
    im_tensor = tranforms(im)
    print(type(im_tensor))

    _, ssd_detections = model(im_tensor.unsqueeze(0).to(device))

    gt_im = cv2.imread(fname)
    h, w = gt_im.shape[:2]
    gt_im_copy = gt_im.copy()
    # Getting predictions from trained model
    boxes = ssd_detections[0]['boxes']
    labels = ssd_detections[0]['labels']
    scores = ssd_detections[0]['scores']
    im = cv2.imread(fname)
    im_copy = im.copy()

    # Saving images with predicted boxes
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box.detach().cpu().numpy()
        x1, y1, x2, y2 = int(w * x1), int(h * y1), int(w * x2), int(h * y2)
        cv2.rectangle(im, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
        cv2.rectangle(im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
        text = '{} : {:.2f}'.format(voc.idx2label[labels[idx].detach().cpu().item()],
                                    scores[idx].detach().cpu().item())
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
        text_w, text_h = text_size
        cv2.rectangle(im_copy, (x1, y1), (x1 + 10 + text_w, y1 + 10 + text_h), [255, 255, 255], -1)
        cv2.putText(im, text=text,
                    org=(x1 + 5, y1 + 15),
                    thickness=1,
                    fontScale=1,
                    color=[0, 0, 0],
                    fontFace=cv2.FONT_HERSHEY_PLAIN)
        cv2.putText(im_copy, text=text,
                    org=(x1 + 5, y1 + 15),
                    thickness=1,
                    fontScale=1,
                    color=[0, 0, 0],
                    fontFace=cv2.FONT_HERSHEY_PLAIN)
    cv2.addWeighted(im_copy, 0.7, im, 0.3, 0, im)
    cv2.imshow('Predictions', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('Done Detecting...')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ssd inference')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc.yaml', type=str)
    parser.add_argument('--evaluate', dest='evaluate',
                        default=False, type=bool)
    parser.add_argument('--infer_samples', dest='infer_samples',
                        default=True, type=bool)
    args = parser.parse_args()

    with torch.no_grad():
        infer(args)

