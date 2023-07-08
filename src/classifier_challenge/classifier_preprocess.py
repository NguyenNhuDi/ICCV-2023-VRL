import math
import os
import shutil
import random
from tqdm import tqdm


# this program will split training and validator apart

def get_image_paths(images_text, image_path):
    out = []

    with open(images_text, 'r') as f:
        image_names = f

        for i in image_names:
            if '\n' in i:
                i = i[:-1]
            curr_img = os.path.join(image_path, i)
            out.append(curr_img)
    return out


def copy_images(src, dst):
    for i in src:
        shutil.copy(i, dst)


if __name__ == '__main__':
    images_path = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WR2021\images'
    train_im_save_path = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WR2021\train_image'
    test_im_save_path = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WR2021\test_image'
    val_im_save_path = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WR2021\val_images'

    image_val_text = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WR2021\trainval.txt'
    test_text = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WR2021\test.txt'

    train_val_path = get_image_paths(image_val_text, images_path)
    test_paths = get_image_paths(test_text, images_path)

    copy_images(test_paths, test_im_save_path)

    train_val_len = len(train_val_path)

    no_repeat = set()
    val_amount = math.floor(train_val_len * 0.15)

    while True:
        if len(no_repeat) == val_amount:
            break
        no_repeat.add(random.randint(0, train_val_len - 1))

    val_paths = []
    train_paths = []

    for i in range(train_val_len):
        if i in no_repeat:
            val_paths.append(train_val_path[i])
        else:
            train_paths.append(train_val_path[i])

    copy_images(val_paths, val_im_save_path)
    copy_images(train_paths, train_im_save_path)
