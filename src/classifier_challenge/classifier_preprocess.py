import math
import os
import shutil
from PIL import Image
import numpy as np
import yaml
from tqdm import tqdm
import albumentations as A


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
    for i in tqdm(src):
        shutil.copy(i, dst)


def tile_and_update_yml(image_paths, save_path, yml_dict):
    # indexes = []
    # for i in range(8):
    #     for (j) in range(8):
    #         indexes.append((i, j))
    #
    # random_indexes = indexes.copy()
    # np.random.shuffle(random_indexes)

    transform = A.Compose(transforms=[
        A.Resize(1024, 1024)
    ])

    for i in tqdm(image_paths):
        path = i[0]
        name = os.path.basename(path)
        class_name = i[1]

        image = np.array(Image.open(path))

        augmented = transform(image=image)
        image = augmented['image']

        # save the current image
        Image.fromarray(image).save(os.path.join(save_path, name))

        # tile it now

        # generating the tuples

        # augment_pos = []
        #
        # for index in range(len(indexes)):
        #     original_col_index = indexes[index][0]
        #     original_row_index = indexes[index][1]
        #
        #     new_col_index = random_indexes[index][0]
        #     new_row_index = random_indexes[index][1]
        #
        #     augment_pos.append((new_col_index * 128, new_row_index * 128,
        #                         original_col_index * 128, original_row_index * 128,
        #                         128, 128))

        # augment_pos = np.array(augment_pos)

        # image = A.augmentations.functional.swap_tiles_on_image(image, augment_pos)

        # new_name = f'{name[:-4]}_tiled.jpg'

        # Image.fromarray(image).save(os.path.join(save_path, new_name))

        # add the names to yaml file
        yml_dict[0][name] = class_name
        # yml_dict[0][new_name] = class_name


def save_images(text_image_val,
                text_test,
                image_dir,
                test_save,
                val_save,
                train_save,
                yml_path,
                yml_dict):
    train_val_path = get_image_paths(text_image_val, image_dir)
    test_paths = get_image_paths(text_test, image_dir)

    with open(yml_path, 'r') as f:
        labels = yaml.load(f, Loader=yaml.FullLoader)

    labels_dict = {
        'unfertilized': [],
        '_PKCa': [],
        'N_KCa': [],
        'NP_Ca': [],
        'NPK_': [],
        'NPKCa': [],
        'NPKCa+m+s': []
    }

    val_paths = []
    train_paths = []

    for i in train_val_path:
        curr_im = os.path.basename(i)
        labels_dict[labels[curr_im]].append(i)

    for label in labels_dict:
        labels_dict[label] = np.array(labels_dict[label])
        np.random.shuffle(labels_dict[label])

    for label in labels_dict:
        curr_len = math.floor(len(labels_dict[label]) * 0.152)
        for i in range(curr_len):
            curr_item = labels_dict[label][i]
            val_paths.append((curr_item, label))
            labels_dict[label] = np.delete(labels_dict[label], np.where(labels_dict[label] == curr_item))

    for label in labels_dict:
        for i in labels_dict[label]:
            train_paths.append((i, label))

    tile_and_update_yml(val_paths, val_save, yml_dict)
    tile_and_update_yml(train_paths, train_save, yml_dict)

    copy_images(test_paths, test_save)

    # writing the new yml
    # with open(yml_save, 'w') as file:
    #     d = yaml.dump(yml_dict, file)
    #
    return yml_dict


if __name__ == '__main__':
    _21_tv_text = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WR2021\trainval.txt'
    _21_test_text = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WR2021\test.txt'
    _21_images_path = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WR2021\images'
    _21_yml_path = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WR2021\labels_trainval.yml'

    _20_tv_text = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WW2020\trainval.txt'
    _20_test_text = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WW2020\test.txt'
    _20_images_path = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WW2020\images'
    _20_yml_path = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WW2020\labels_trainval.yml'

    test_save_dir = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WW2020\test_image'
    val_save_dir = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WW2020\val_image'
    train_save_dir = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WW2020\train_image'
    yml_save_dir = r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WW2020\updated_yml.yml'

    yml_dict = [{}]

    # yml_dict = save_images(_21_tv_text, _21_test_text,
    #                        _21_images_path, test_save_dir,
    #                        val_save_dir, train_save_dir,
    #                        _21_yml_path, yml_dict)

    yml_dict = save_images(_20_tv_text, _20_test_text,
                           _20_images_path, test_save_dir,
                           val_save_dir, train_save_dir,
                           _20_yml_path, yml_dict)

    with open(yml_save_dir, 'w') as file:
        d = yaml.dump(yml_dict, file)

