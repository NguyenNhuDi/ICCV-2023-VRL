train_transform = A.Compose(
        transforms=[
            A.RandomCrop(height=CROP_SIZE, width=CROP_SIZE, always_apply=True),
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
            A.OneOf(transforms=[
                A.RandomFog(0.1,0.3,0.5,p=0.5),
                A.RandomShadow ((0,0,1,1), 1, 1, p=0.5),
            ],p=4),

            A.OneOf(transforms=[
                A.Sharpen(alpha=(0.0, 0.1), lightness=(0,0.1), p=0.25),
                A.RandomBrightnessContrast((-0.05,0.05), (0.0), p=0.25),
                A.RandomBrightnessContrast((0,0), (-0.25,0.25), p=0.25),
                A.ImageCompression(65,100, p=0.25),
            ], p=0.35),


            A.OneOf(transforms=[
                A.GaussNoise((0, 0.02), p=0.2125),
                A.ISONoise((0.01,0.1), p=0.2125),
                A.RandomGamma((80,110), p=0.2125),
                A.Blur(blur_limit=(1,2), p=0.2125),
                A.MotionBlur(blur_limit=3, p=0.15)
            ], p=0.45),


            A.Normalize(mean=(0.4680, 0.4038, 0.2885), std=(0.2476, 0.2107, 0.1931))

        ],
        p=1.0,

    )
