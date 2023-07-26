from model_trainer import ModelTrainer
import argparse
import json
import albumentations as A
import os


def check_is_none(item):
    if item.lower() == 'none' or item == '':
        return None
    return item


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

    best_save_name = os.path.join(save_dir, best_save_name)
    last_save_name = os.path.join(save_dir, last_save_name)

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
    model_to_load = check_is_none(model_to_load)
    model = args['model']
    model_name = args['model_name']

    months = args['which_months']
    train = args['which_train_set']
    val = args['which_val_set']

    CROP_SIZE = 750

    val_transform = A.Compose(
        transforms=[
            A.RandomCrop(height=CROP_SIZE, width=CROP_SIZE, always_apply=True),
            A.Resize(image_size, image_size),
            A.Normalize(mean=((0.4680, 0.4038, 0.2885)), std=(0.2476, 0.2107, 0.1931))
        ],
        p=1.0,
    )

    train_transform = A.Compose(
        transforms=[
            A.RandomCrop(height=CROP_SIZE, width=CROP_SIZE, always_apply=True),
            A.Resize(image_size, image_size),
            # A.GaussNoise(),
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

            A.OneOf(transforms=[
                A.RandomFog(0.1, 0.3, 0.5, p=0.2),
                A.RandomRain(-20, 20, 50, 1, rain_type='drizzle', p=0.2),
                A.RandomRain(-20, 20, 50, 1, rain_type='heavy', p=0.2),
                A.RandomRain(-20, 20, 50, 1, rain_type='torrential', p=0.2),
                A.RandomShadow((0, 0, 1, 1), 5, 10, p=0.2),
            ], p=0.3),

            A.OneOf(transforms=[
                A.Sharpen(p=0.25),
                A.RandomBrightnessContrast((-0.2, 0.2), (0.0), p=0.25),
                A.RandomBrightnessContrast((0, 0), (-0.2, 0.2), p=0.25),
                A.Blur(p=0.125),
                A.MotionBlur(p=0.125)
            ], p=0.3),

            A.OneOf(transforms=[
                A.ImageCompression(50, 100, p=0.25),
                A.GaussNoise((0, 0.02), p=0.25),
                A.ISONoise((0.01, 0.1), p=0.25),
                A.RandomGamma((75, 200), p=0.25),
            ], p=0.3),

            A.Normalize(mean=((0.4680, 0.4038, 0.2885)), std=(0.2476, 0.2107, 0.1931))

            # ],p=0.2),

        ],
        p=1.0,
    )

    trainer = ModelTrainer(yaml_path,
                           best_save_name,
                           last_save_name,
                           save_dir,
                           csv,
                           image_dir_20,
                           image_dir_21,
                           train_transform=train_transform,
                           val_transform=val_transform,
                           batch_size=batch_size,
                           epochs=epochs,
                           weight_decay=weight_decay,
                           num_processes=num_processes,
                           learning_rate=learning_rate,
                           momentum=momentum,
                           unfreeze_epoch=unfreeze_epoch,
                           epoch_step=epoch_step,
                           gamma=gamma,
                           model_to_load=model_to_load,
                           months=months,
                           val=val,
                           train=train,
                           model=model,
                           model_name=model_name
                           )
    trainer()
