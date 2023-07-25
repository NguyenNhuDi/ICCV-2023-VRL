from model_trainer import ModelTrainer
import argparse
import json
import albumentations as A

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
    image_size = args['image_size']
    batch_size = args['batch_size']
    epochs = args['epochs']
    num_processes = args['num_processes']
    learning_rate = args['learning_rate']
    momentum = args['momentum']
    unfreeze_epoch = args['unfreeze_epoch']
    epoch_step = args['epoch_step']
    gamma = args['gamma']
    csv_20 = args['csv_20']
    csv_21 = args['csv_21']
    image_dir_20 = args['image_dir_20']
    image_dir_21 = args['image_dir_21']
    model_to_load = args['model_to_load']

    if model_to_load.lower() == 'none' or model_to_load == '':
        model_to_load = None

    model = args['model']
    model_name = args['model_name']

    CROP_SIZE = 750

    val_transform = A.Compose(
        transforms=[
            A.RandomCrop(height=CROP_SIZE, width=CROP_SIZE, always_apply=True),
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.4680, 0.4038, 0.2885), std=(0.2476, 0.2107, 0.1931))
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
            A.Normalize(mean=(0.4680, 0.4038, 0.2885), std=(0.2476, 0.2107, 0.1931))
            # A.OneOf(transforms=[
            #     # A.RandomFog(0.1,0.3,0.5,p=0.05),
            #     # A.RandomRain(-20, 20, 50, 1, rain_type='drizzle', p =0.05),
            #     # A.RandomRain(-20, 20, 50, 1, rain_type='heavy', p = 0.05),
            #     # A.RandomRain(-20, 20, 50, 1, rain_type='torrential', p = 0.05),
            #     # A.RandomShadow ((0,0,1,1), 5, 10, p =0.05),
            #     # A.Sharpen(p=0.05),
            #     # A.RandomBrightnessContrast((-0.2,0.2), (0.0), p=0.05),
            #     # A.RandomBrightnessContrast((0,0), (-0.2,0.2), p=0.05),
            #     # A.ImageCompression(50,100, p=0.05),
            #     # # A.GaussNoise((0, 0.02), p=0.06),
            #     # A.ISONoise((0.01,0.1), p=0.07),
            #     # A.RandomGamma((75,200), p=0.07),
            #     # A.Blur(p=0.1),
            #     # A.MotionBlur(p=0.1)

            # ],p=0.2),

        ],
        p=1.0,
    )

    trainer = ModelTrainer(yaml_path,
                           best_save_name,
                           last_save_name,
                           train_transform,
                           val_transform,
                           batch_size,
                           epochs,
                           num_processes,
                           learning_rate,
                           momentum,
                           unfreeze_epoch,
                           epoch_step,
                           gamma,
                           csv_20,
                           csv_21,
                           image_dir_20,
                           image_dir_21,
                           model_to_load,
                           model,
                           model_name
                           )
    trainer()
