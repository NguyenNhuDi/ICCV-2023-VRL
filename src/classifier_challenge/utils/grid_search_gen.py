from itertools import product
import json as json
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    prog='Winter Wheat, Winter Rye Classifier pre process',
    description='This program will create different val and train set for classification of winter wheat and '
                'winter rye based on a  and hash function',
    epilog='Vision Research Lab')

    parser.add_argument('-c', '--config', required=True,
                        help='The path to the config file')
    args = parser.parse_args()

    with open(args.config) as f:
        args = json.load(f)

    lr = args['lr']
    m = args['m']
    wd = args['wd']
    g = args['g']
    image_size = args['image_size']
    batch_size = args['batch_size']
    epochs = args['epochs']
    save_dir = args['save_dir']
    model = args['model']
    model_name = args['model_name']

    out_json_0 = {"yaml_path": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/CSV_Implemented_training/updated_yml'
                               '.yml',
                  "best_save_name": [],
                  "last_save_name": [],
                  "save_dir": save_dir,
                  "which_months": [3, 4, 5],
                  "which_val_set": [0, 1],
                  "which_train_set": [0, 1],
                  "image_size": image_size,
                  "batch_size": batch_size,
                  "epochs": epochs,
                  "weight_decay": [],
                  "num_processes": 20,
                  "learning_rate": [],
                  "momentum": [],
                  "unfreeze_epoch": 3,
                  "epoch_step": 10,
                  "gamma": [],
                  "csv": [],
                  "image_dir_20": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/2020_data/images',
                  "image_dir_21": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/2021_data/images',
                  "model_to_load": [],
                  "model": model,
                  "model_name": model_name,
                  "log_name": [],
                  "cut_mix": False

                  }

    out_json_1 = {"yaml_path": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/CSV_Implemented_training/updated_yml.yml',
                  "best_save_name": [],
                  "last_save_name": [],
                  "save_dir": save_dir,
                  "which_months": [3, 4, 5],
                  "which_val_set": [0, 1],
                  "which_train_set": [0, 1],
                  "image_size": image_size,
                  "batch_size": batch_size,
                  "epochs": epochs,
                  "weight_decay": [],
                  "num_processes": 20,
                  "learning_rate": [],
                  "momentum": [],
                  "unfreeze_epoch": 3,
                  "epoch_step": 10,
                  "gamma": [],
                  "csv": [],
                  "image_dir_20": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/2020_data/images',
                  "image_dir_21": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/2021_data/images',
                  "model_to_load": [],
                  "model": model,
                  "model_name": model_name,
                  "log_name": [],
                  "cut_mix": False

                  }

    out_json_2 = {"yaml_path": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/CSV_Implemented_training/updated_yml.yml',
                  "best_save_name": [],
                  "last_save_name": [],
                  "save_dir": save_dir,
                  "which_months": [3, 4, 5],
                  "which_val_set": [0, 1],
                  "which_train_set": [0, 1],
                  "image_size": image_size,
                  "batch_size": batch_size,
                  "epochs": epochs,
                  "weight_decay": [],
                  "num_processes": 20,
                  "learning_rate": [],
                  "momentum": [],
                  "unfreeze_epoch": 3,
                  "epoch_step": 10,
                  "gamma": [],
                  "csv": [],
                  "image_dir_20": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/2020_data/images',
                  "image_dir_21": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/2021_data/images',
                  "model_to_load": [],
                  "model": model,
                  "model_name": model_name,
                  "log_name": [],
                  "cut_mix": False

                  }

    out_json_3 = {"yaml_path": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/CSV_Implemented_training/updated_yml.yml',
                  "best_save_name": [],
                  "last_save_name": [],
                  "save_dir": save_dir,
                  "which_months": [3, 4, 5],
                  "which_val_set": [0, 1],
                  "which_train_set": [0, 1],
                  "image_size": image_size,
                  "batch_size": batch_size,
                  "epochs": epochs,
                  "weight_decay": [],
                  "num_processes": 20,
                  "learning_rate": [],
                  "momentum": [],
                  "unfreeze_epoch": 3,
                  "epoch_step": 10,
                  "gamma": [],
                  "csv": [],
                  "image_dir_20": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/2020_data/images',
                  "image_dir_21": '/home/nhu.nguyen2/ICCV_2023/classifier_challenge/2021_data/images',
                  "model_to_load": [],
                  "model": model,
                  "model_name": model_name,
                  "log_name": [],
                  "cut_mix": False

                  }

    all_combo = (list(product(lr, m, wd, g)))
    all_len = len(all_combo)

    counter = 0
    index = 0

    while counter < all_len//4:
        for i in all_combo[index]:
            if i[0] == 'lr':
                out_json_0['learning_rate'].append(i[1])
            elif i[0] == 'm':
                out_json_0['momentum'].append(i[1])
            elif i[0] == 'wd':
                out_json_0['weight_decay'].append(i[1])
            else:
                out_json_0['gamma'].append(i[1])

        out_json_0['best_save_name'].append('best')
        out_json_0['last_save_name'].append('last')
        out_json_0['csv'].append('/home/nhu.nguyen2/ICCV_2023/classifier_challenge/CSV_Implemented_training/CSV/Efficient Net v2 S/en2s_0.csv')
        out_json_0['log_name'].append(f'test{index}')
        out_json_0['model_to_load'].append('')

        index += 1
        counter += 1

    counter = 0
    while counter < all_len//4:
        for i in all_combo[index]:
            if i[0] == 'lr':
                out_json_1['learning_rate'].append(i[1])
            elif i[0] == 'm':
                out_json_1['momentum'].append(i[1])
            elif i[0] == 'wd':
                out_json_1['weight_decay'].append(i[1])
            else:
                out_json_1['gamma'].append(i[1])

        out_json_1['best_save_name'].append('best')
        out_json_1['last_save_name'].append('last')
        out_json_1['csv'].append('/home/nhu.nguyen2/ICCV_2023/classifier_challenge/CSV_Implemented_training/CSV/Efficient Net v2 S/en2s_0.csv')
        out_json_1['log_name'].append(f'test{index}')
        out_json_1['model_to_load'].append('')

        index += 1
        counter += 1

    counter = 0
    while counter < all_len//4:
        for i in all_combo[index]:
            if i[0] == 'lr':
                out_json_2['learning_rate'].append(i[1])
            elif i[0] == 'm':
                out_json_2['momentum'].append(i[1])
            elif i[0] == 'wd':
                out_json_2['weight_decay'].append(i[1])
            else:
                out_json_2['gamma'].append(i[1])

        out_json_2['best_save_name'].append('best')
        out_json_2['last_save_name'].append('last')
        out_json_2['csv'].append('/home/nhu.nguyen2/ICCV_2023/classifier_challenge/CSV_Implemented_training/CSV/Efficient Net v2 S/en2s_0.csv')
        out_json_2['log_name'].append(f'test{index}')
        out_json_2['model_to_load'].append('')

        index += 1
        counter += 1

    while index < len(all_combo):
        for i in all_combo[index]:
            if i[0] == 'lr':
                out_json_3['learning_rate'].append(i[1])
            elif i[0] == 'm':
                out_json_3['momentum'].append(i[1])
            elif i[0] == 'wd':
                out_json_3['weight_decay'].append(i[1])
            else:
                out_json_3['gamma'].append(i[1])

        out_json_3['best_save_name'].append('best')
        out_json_3['last_save_name'].append('last')
        out_json_3['csv'].append('/home/nhu.nguyen2/ICCV_2023/classifier_challenge/CSV_Implemented_training/CSV/Efficient Net v2 S/en2s_0.csv')
        out_json_3['log_name'].append(f'test{index}')
        out_json_3['model_to_load'].append('')

        index += 1

    json_string_0 = json.dumps(out_json_0, indent=2)

    with open(os.path.join(save_dir,f'test_0.json'), 'w') as out:
        out.write(json_string_0)

    json_string_1 = json.dumps(out_json_1, indent=2)

    with open(os.path.join(save_dir,f'test_1.json'), 'w') as out:
        out.write(json_string_1)

    json_string_2 = json.dumps(out_json_2, indent=2)

    with open(os.path.join(save_dir,f'test_2.json'), 'w') as out:
        out.write(json_string_2)

    json_string_3 = json.dumps(out_json_3, indent=2)

    with open(os.path.join(save_dir,f'test_3.json'), 'w') as out:
        out.write(json_string_3)
