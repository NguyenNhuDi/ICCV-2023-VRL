import numpy as np
import torch
import sys

sys.path.append(r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\ICCV-2023-VRL')
from src.data_loading.DSAL import DSAL
from PIL import Image
import albumentations as A
import yaml


def transform_image_label(image_path, label, transform):
    image = np.array(Image.open(image_path))

    unfertilized_label = 1 if label == 'unfertilized' else 0
    _PKCa_label = 1 if label == '_PKCa' else 0
    N_KCa_label = 1 if label == 'N_KCa' else 0
    NP_Ca_label = 1 if label == 'NP_Ca' else 0
    NPK__label = 1 if label == 'NPK_' else 0
    NPKCa_label = 1 if label == 'NPKCa' else 0
    NPKCa_m_s_label = 1 if label == 'NPKCa+m+s' else 0

    out_label = torch.tensor([
        unfertilized_label,
        _PKCa_label,
        N_KCa_label,
        NP_Ca_label,
        NPK__label,
        NPKCa_label,
        NPKCa_m_s_label
    ])

    if transform is not None:
        augmented = transform(image=image)
        image = augmented['image']

    # converting the image and mask into tensors
    image = torch.from_numpy(image).permute(2, 1, 0)

    return image, out_label


if __name__ == '__main__':
    transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.HueSaturationValue()
    ])

    images = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WR2021\train_image'
    yaml_path = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WR2021\labels_trainval.yml'

    with open(yaml_path, 'r') as f:
        labels = yaml.safe_load(f)

    batch_size = 32
    epochs = 2
    num_processes = 6

    dsal = DSAL(images,
                yaml_path,
                transform_image_label,
                batch_size=batch_size,
                epochs=epochs,
                num_processes=num_processes,
                max_queue_size=num_processes * 3,
                transform=transform)

    print('starting pathing...')
    dsal.start()
    print('pathing finished')

    for i in range(dsal.num_batches):
        image, label = dsal.get_item()

        print(f'image shape: {image.shape}, label: {label}')

    dsal.join()
