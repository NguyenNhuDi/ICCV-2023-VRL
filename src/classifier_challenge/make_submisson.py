import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision import models
import sys
import argparse
import json
from PIL import Image
import yaml
from tqdm import tqdm
from DSAL import DSAL
import os
import albumentations as A

def read_image(image_path, transform):
    out_image = np.array(Image.open(image_path))
    image_name = os.path.basename(image_path)

    if transform is not None:
        augmented = transform(image=out_image)
        out_image = augmented['image']
        
    # converting the image and mask into tensors

    out_image = torch.from_numpy(out_image).permute(2,0,1)
    return out_image, image_name

def evaluate(model, val_batches, device):
    model.eval()
    # total_correct = 0
    # total_loss = 0
    # total = 0
    for batch in val_batches:
        image, label = batch
        image, label = image.to(device), label.to(device)

        with torch.no_grad():
            outputs = model(image)
            for i in range(len(outputs)):
                prediction_index = torch.argmax(outputs[i]).cpu().numpy()
                


    # loss = total_loss / total
    # accuracy = total_correct / total



def generate(model_path, test_image_dir, batch_size):
    

    img_dir = np.array(glob.glob(f'{test_image_dir}/*.jpg'))


    HEIGHT = 600
    WIDTH = 600
    transform = A.Compose(
        transforms=[
            A.RandomCrop(height=HEIGHT, width=WIDTH, always_apply=True),
            A.Normalize(mean=((0.5385, 0.4641, 0.3378)), std=(0.5385, 0.4641, 0.3378))
        ],
        p=1.0,
    )

    model = torch.load(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    image_batch = []
    name_batch = []

    counter = 0
    batch_counter = 0

    temp_img = []
    temp_name = []

    for i in tqdm(range(len(img_dir))):
        image, image_name = read_image(img_dir[i],transform=transform)

        temp_img.append(image)
        temp_name.append(image_name)

        batch_counter += 1

        if batch_counter == batch_size:
            temp_img = torch.stack(temp_img, dim=0)


            image_batch.append(temp_img)
            name_batch.append(temp_name)
            temp_img = []
            temp_name = []

            batch_counter = 0
        
    if len(temp_img) > 0:
        temp_img = torch.stack(temp_img, dim=0)

        image_batch.append(temp_img)
        name_batch.append(temp_name)


    predictions_20 = []
    predictions_21 = []

    for i in tqdm(range(len(image_batch))):
        image = image_batch[i].to(device)
        output = model(image)

        for j in range(len(output)):
            name = name_batch[i][j]
            prediction = int(torch.argmax(output[j]).cpu().numpy())

            if name[0:4] == '2020':
                predictions_20.append((name, prediction))
            else:
                predictions_21.append((name, prediction))


    f = open('predictions_WW2020.txt', 'w')

    for i in predictions_20:
        f.write(f'{i[0]} {i[1]}\n')

    f.close()

    f = open('predictions_WR2021.txt', 'w')
             
    for i in predictions_21:
        f.write(f'{i[0]} {i[1]}\n')
    
    f.close()

if __name__ == '__main__':

    model_path = r'/home/nhu.nguyen2/ICCV_2023/classifier_challenge/saved_models/en2_l_140_combined_ALBUMENTATION_LAST.pth'

    test_dir = r'/home/nhu.nguyen2/ICCV_2023/classifier_challenge/combined/test_image'

    generate(model_path, test_dir, 8)
