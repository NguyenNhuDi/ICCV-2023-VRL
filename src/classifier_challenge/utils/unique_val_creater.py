import numpy as np
import yaml

# This is a percent
SIZE_PER_CLASS = 0.04

MARCH_MAX_SIZE = [192 * SIZE_PER_CLASS, 180 * SIZE_PER_CLASS, 192 * SIZE_PER_CLASS, 192 * SIZE_PER_CLASS,
                  192 * SIZE_PER_CLASS, 192 * SIZE_PER_CLASS, 192 * SIZE_PER_CLASS]
APRIL_MAX_SIZE = [64 * SIZE_PER_CLASS, 60 * SIZE_PER_CLASS, 64 * SIZE_PER_CLASS, 64 * SIZE_PER_CLASS,
                  64 * SIZE_PER_CLASS, 64 * SIZE_PER_CLASS, 64 * SIZE_PER_CLASS]
MAY_MAX_SIZE = [128 * SIZE_PER_CLASS, 120 * SIZE_PER_CLASS, 128 * SIZE_PER_CLASS, 128 * SIZE_PER_CLASS,
                128 * SIZE_PER_CLASS, 128 * SIZE_PER_CLASS, 128 * SIZE_PER_CLASS]

if __name__ == '__main__':

    val_yaml_paths = [
        r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WR2021\labels_trainval.yml',
        r'C:\Users\coanh\Desktop\Uni Work\ICCV 2023\DND-Diko-WWWR\WW2020\labels_trainval.yml'
    ]

    unique_val_set = {}
    train_set = {}

    april_total = 0
    april_labels = [[] for i in range(7)]

    march_total = 0
    march_labels = [[] for i in range(7)]

    may_total = 0
    may_labels = [[] for i in range(7)]

    labels = {}

    for yaml_path in val_yaml_paths:
        with open(yaml_path, 'r') as f:
            labels.update(yaml.safe_load(f))

    labels_ = []
    for key in labels:
        labels_.append((key, labels[key]))

    labels = labels_.copy()

    np.random.shuffle(labels)

    for _ in labels:
        key, curr_class = _

        class_index = 0

        if curr_class == '_PKCa':
            class_index = 1
        elif curr_class == 'N_KCa':
            class_index = 2
        elif curr_class == 'NP_Ca':
            class_index = 3
        elif curr_class == 'NPK_':
            class_index = 4
        elif curr_class == 'NPKCa':
            class_index = 5
        elif curr_class == 'NPKCa+m+s':
            class_index = 6

        if int(key[5]) == 3:
            march_total += 1

            if len(march_labels[class_index]) < MARCH_MAX_SIZE[class_index]:
                unique_val_set[key] = curr_class
            else:
                train_set[key] = curr_class

            march_labels[class_index].append(key)

        if int(key[5]) == 4:
            april_total += 1

            if len(april_labels[class_index]) < APRIL_MAX_SIZE[class_index]:
                unique_val_set[key] = curr_class
            else:
                train_set[key] = curr_class
            april_labels[class_index].append(key)

        if int(key[5]) == 5:
            if len(may_labels[class_index]) < MAY_MAX_SIZE[class_index]:
                unique_val_set[key] = curr_class
            else:
                train_set[key] = curr_class
            may_labels[class_index].append(key)
            may_total += 1

    print(
        f' mar: {march_total} --- 0: {len(march_labels[0])} --- 1: {len(march_labels[1])} --- 2: {len(march_labels[2])} --- 3: {len(march_labels[3])} --- 4: {len(march_labels[4])} --- 5: {len(march_labels[5])} --- 6: {len(march_labels[6])}')
    print(
        f' apr: {april_total} --- 0: {len(april_labels[0])} --- 1: {len(april_labels[1])} --- 2: {len(april_labels[2])} --- 3: {len(april_labels[3])} --- 4: {len(april_labels[4])} --- 5: {len(april_labels[5])} --- 6: {len(april_labels[6])}')
    print(
        f' may: {may_total} --- 0: {len(may_labels[0])} --- 1: {len(may_labels[1])} --- 2: {len(may_labels[2])} --- 3: {len(may_labels[3])} --- 4: {len(may_labels[4])} --- 5: {len(may_labels[5])} --- 6: {len(may_labels[6])}')

    print(len(train_set))
    print(len(unique_val_set))

    # saving unique val set

    with open('unique_val.yml', 'w') as f:
        yaml.dump(unique_val_set, f)

    with open('train_set.yml', 'w') as f:
        yaml.dump(train_set, f)
