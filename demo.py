import torch
import argparse
import os
import yaml
import random
from tqdm import tqdm
from ssd.model.ssd import SSD
from adssd.model.adssd import ADSSD
import numpy as np
import cv2
from dataset.voc import VOCDataset
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms.v2 as T
from torchvision.io import read_image
import streamlit as st
from time import time
from PIL import Image
from ultralytics import YOLO


device = torch.device('cpu')
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
im_size = 300
transforms = T.Compose([
    T.Resize(size=(im_size, im_size)),
    T.ToPureTensor(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=imagenet_mean, std=imagenet_std)
])


def load_model_and_dataset(config_path, model_name='ssd'):
    print('Loading model and dataset...')
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    voc = VOCDataset('test',
                     im_sets=dataset_config['test_im_sets'])
    test_dataset = DataLoader(voc, batch_size=1, shuffle=False)
    if (model_name == 'adssd'):
        model = ADSSD(config=model_config,
                    num_classes=dataset_config['num_classes'])
    else:
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




def infer(model, im_path):
    # model.eval()
    im = read_image(im_path)
    im = transforms(im)
    # im_tensor, target, fname = voc[140]
    fname = im_path
    im_tensor = transforms(im)
    print(type(im_tensor))
    start = time()
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
    end = time()
    return im, end-start

def infer_yolo(model, im_path):
    # im = read_image(im_path)
    # im = transforms(im)
    im = cv2.imread(im_path)
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    start = time()
    results = model(im, conf=0.5)
    end = time()
    return results[0].plot(), end - start


if __name__ == '__main__':
    model_ssd, voc, test_dataset, config = load_model_and_dataset(config_path='config/voc.yaml')
    conf_threshold = config['train_params']['infer_conf_threshold']
    model_ssd.low_score_threshold = conf_threshold
    print("SSD Loaded")

    model_adssd, voc, test_dataset, config = load_model_and_dataset(config_path='config/advoc.yaml', model_name='adssd')
    conf_threshold = config['train_params']['infer_conf_threshold']
    model_adssd.low_score_threshold = conf_threshold
    print("ADSSD Loaded")

    model_yolov8 = YOLO('results/YOLOV8_BEST.pt')
    model_yolov8.predict()
    # model_yolov8 = torch.load('results/YOLOV8_BEST.pt')
    model_yolov8.to(device)
    

    
    st.title('SSD/ADSSD/YoloV8 Demo')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image.save("test.jpg")

        ssd_pred, ssd_time = infer(model_ssd, 'test.jpg')
        ssd_pred = cv2.cvtColor(ssd_pred, cv2.COLOR_BGR2RGB)

        adssd_pred, adssd_time = infer(model_adssd, 'test.jpg')
        adssd_pred = cv2.cvtColor(adssd_pred, cv2.COLOR_BGR2RGB)

        yolo_pred, yolo_time = infer_yolo(model_yolov8, 'test.jpg')
        # yolo_pred = yolo_pred[0].cpu().numpy()
        yolo_pred = cv2.cvtColor(yolo_pred, cv2.COLOR_BGR2RGB)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(ssd_pred, caption=f'SSD Prediction\nTime: {ssd_time:.9f}s', use_container_width=True)

        with col2:
            st.image(adssd_pred, caption=f'ADSSD Prediction\nTime: {adssd_time:.9f}s', use_container_width=True)

        with col3:
            st.image(yolo_pred, caption=f'YOLOv8 Prediction\nTime: {yolo_time:.9f}s', use_container_width=True)

