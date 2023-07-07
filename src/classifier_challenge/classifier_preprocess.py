import os
import shutil

import pandas as pd
import yaml
from tqdm import tqdm

images_path = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WR2021\images'
train_im_save_path = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WR2021\train_image'
test_im_save_path = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WR2021\test_image'
csv_save_path = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WR2021\WR2021.csv'


def copy_image(images, save_path):
    for i in images:
        if '\n' in i:
            i = i[:-1]

        curr_img = os.path.join(images_path, i)
        shutil.copy(curr_img, save_path)


if __name__ == '__main__':
    yaml_path = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WR2021\labels_trainval.yml'

    train_im = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WR2021\trainval.txt'
    test_im = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WR2021\test.txt'

    train_images = set(open(train_im))
    test_images = set(open(test_im))

    copy_image(train_images, train_im_save_path)
    copy_image(test_images, test_im_save_path)

    csv = {
        'image_name': [],
        'unfertilized': [],
        '_PKCa': [],
        'N_KCa': [],
        'NP_Ca': [],
        'NPK_': [],
        'NPKCa': [],
        'NPKCa+m+s': []
    }

    with open(yaml_path, 'r') as f:
        labels = yaml.safe_load(f)

    for image in tqdm(train_images):
        if '\n' in image:
            image = image[:-1]

        label = labels[image]
        csv['image_name'].append(image)

        csv['unfertilized'].append(1 if label == 'unfertilized' else 0)
        csv['_PKCa'].append(1 if label == '_PKCa' else 0)
        csv['N_KCa'].append(1 if label == 'N_KCa' else 0)
        csv['NP_Ca'].append(1 if label == 'NP_Ca' else 0)
        csv['NPK_'].append(1 if label == 'NPK_' else 0)
        csv['NPKCa'].append(1 if label == 'NPKCa' else 0)
        csv['NPKCa+m+s'].append(1 if label == 'NPKCa+m+s' else 0)

    df = pd.DataFrame.from_dict(csv)
    df.to_csv(csv_save_path, index=False)
