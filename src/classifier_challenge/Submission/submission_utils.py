import os
import albumentations as A
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def lambda_transform(x: torch.Tensor, **kwargs) -> torch.Tensor:
    return x / 255


def read_image(image_dir):
    # out_image = np.array(Image.open(image_path), dtype='float32') / 255.0
    all_months = []
    march = []
    april = []
    may = []
    for i in tqdm(image_dir):
        image_name = os.path.basename(i)
        curr_item = (np.array(Image.open(i), dtype='uint8'), image_name)
        all_months.append(curr_item)

        if image_name[5] == '3':
            march.append(curr_item)
        elif image_name[5] == '4':
            april.append(curr_item)
        else:
            may.append(curr_item)

    return all_months, march, april, may


def read_val_image(image_dir, labels):
    # out_image = np.array(Image.open(image_path), dtype='float32') / 255.0
    all_months = []
    march = []
    april = []
    may = []
    for i in tqdm(image_dir):
        image_name = os.path.basename(i)

        if image_name not in labels:
            continue

        curr_item = (np.array(Image.open(i), dtype='uint8'), image_name)
        all_months.append(curr_item)

        if image_name[5] == '3':
            march.append(curr_item)
        elif image_name[5] == '4':
            april.append(curr_item)
        else:
            may.append(curr_item)

    return all_months, march, april, may


def process_image(image, transform):
    out_image = image[0]

    image_name = image[1]

    if transform is not None:
        augmented = transform(image=out_image)
        out_image = augmented['image']

    # converting the image and mask into tensors

    out_image = torch.from_numpy(out_image).permute(2, 0, 1)
    return out_image, image_name


def generate(model_path, images_arr, batch_size, transform, predict_dict):
    model = torch.load(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    image_batch = []
    name_batch = []

    batch_counter = 0

    temp_img = []
    temp_name = []

    for i in tqdm(range(len(images_arr))):
        image, image_name = process_image(images_arr[i], transform)

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

    for i in tqdm(range(len(image_batch))):
        image = image_batch[i].to(device)
        output = model(image)

        for j in range(len(output)):
            name = name_batch[i][j]
            prediction = int(torch.argmax(output[j]).cpu().numpy())

            if name in predict_dict:
                predict_dict[name][prediction] += 1
            else:
                predict_dict[name] = [0 for i in range(7)]
                predict_dict[name][prediction] += 1

    return predict_dict


def make_prediction(predict_dict, models_paths, images, batch_size, image_size, means, std, run_amount):
    counter = 0
    for model_path in models_paths:
        transform = A.Compose(
            transforms=[
                A.Resize(image_size[counter], image_size[counter]),
                A.Lambda(image=lambda_transform),
                A.Normalize(mean=means[counter], std=std[counter], max_pixel_value=1.0)
            ],
            p=1.0,
        )

        print(f'\n\n\n ----curr model {os.path.basename(model_path)}---\n\n')
        counter += 1

        for i in range(run_amount):
            print(f'\n\n ---iteration {i}---\n\n')

            predict_dict = generate(model_path, images, batch_size, transform, predict_dict)

    return predict_dict


"""This is for subset chooser portion"""


def make_single_prediction(predict_dict, model_path, images, batch_size, image_size, mean, std, run_amount):
    transform = A.Compose(
        transforms=[
            A.Resize(image_size, image_size),
            A.Lambda(image=lambda_transform),
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0)
        ],
        p=1.0,
    )

    print(f'\n\n\n ----curr model {os.path.basename(model_path)}---\n\n')

    for i in range(run_amount):
        print(f'\n\n ---iteration {i}---\n\n')
        predict_dict = generate(model_path, images, batch_size, transform, predict_dict)

    return predict_dict


