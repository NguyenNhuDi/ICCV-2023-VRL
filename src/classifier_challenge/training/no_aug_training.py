from model_trainer import ModelTrainer
import argparse
import json
import albumentations as A
import os
import torch


def lambda_transform(x: torch.Tensor, **kwargs) -> torch.Tensor:
    return x / 255


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Model Trainer',
        description='This program will train a model',
        epilog='Vision Research Lab')
    parser.add_argument('-c', '--config', required=True,
                        help='The path to the config file.')
    args = parser.parse_args()

    with open(args.config) as f:
        args = json.load(f)

    yaml_path = args['yaml_path']
    best_save_name = args['best_save_name']
    last_save_name = args['last_save_name']
    save_dir = args['save_dir']

    image_size = args['image_size']
    batch_size = args['batch_size']
    epochs = args['epochs']
    num_processes = args['num_processes']
    learning_rate = args['learning_rate']
    momentum = args['momentum']
    weight_decay = args['weight_decay']
    unfreeze_epoch = args['unfreeze_epoch']
    epoch_step = args['epoch_step']
    gamma = args['gamma']
    csv = args['csv']
    image_dir_20 = args['image_dir_20']
    image_dir_21 = args['image_dir_21']
    model_to_load = args['model_to_load']

    model = args['model']
    model_name = args['model_name']
    out_name = args['log_name']

    months = args['which_months']
    train = args['which_train_set']
    val = args['which_val_set']

    train_transform = A.Compose(
        transforms=[
            A.Resize(image_size, image_size),
            A.Flip(p=0.5),
            A.Rotate(
                limit=(-90, 90),
                interpolation=1,
                border_mode=0,
                value=0,
                mask_value=0,
                always_apply=False,
                p=0.75,
            ),
            A.Lambda(image=lambda_transform)
        ],
        p=1.0,
    )

    val_transform = A.Compose(
        transforms=[
            A.Resize(image_size, image_size),
            A.Lambda(image=lambda_transform)
        ],
        p=1.0,
    )

    submit_json = {
        'test_dir': '',
        'save_path': '',
        'batch_size': '',
        'all_models_paths': [],
        'all_month_sizes': [],
        'all_month_means': [],
        'all_month_stds': [],
        'march_sizes': [],
        'april_sizes': [],
        'may_sizes': [],
        'march_means': [],
        'march_stds': [],
        'april_means': [],
        'april_stds': [],
        'may_means': [],
        'may_stds': [],
        'march_models': [],
        'april_models': [],
        'may_models': [],
        'run_amount': 1

    }

    for i in range(len(csv)):
        trainer = ModelTrainer(yaml_path=yaml_path,
                               best_save_name=os.path.join(save_dir, best_save_name[i]),
                               last_save_name=os.path.join(save_dir, last_save_name[i]),
                               save_dir=save_dir,
                               csv=csv[i],
                               image_dir_20=image_dir_20,
                               image_dir_21=image_dir_21,
                               train_transform=train_transform,
                               val_transform=val_transform,
                               image_size=image_size,
                               submit_json=submit_json,
                               batch_size=batch_size,
                               epochs=epochs,
                               weight_decay=weight_decay[i],
                               num_processes=num_processes,
                               learning_rate=learning_rate[i],
                               momentum=momentum[i],
                               unfreeze_epoch=unfreeze_epoch,
                               epoch_step=epoch_step,
                               gamma=gamma[i],
                               model_to_load=model_to_load[i],
                               months=months,
                               val=val,
                               train=train,
                               model=model,
                               model_name=f'{model_name} -- {i + 1}',
                               out_name=out_name[i]
                               )
        submit_json = trainer()

    with open(f'{model_name}.json', 'w') as json_file:
        json.dump(submit_json, json_file, indent=4)