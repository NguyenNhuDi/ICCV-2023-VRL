
#!/bin/bash

python training_entry.py -d /home/student/andrew/Documents/Seg3D/datasets/main -m /home/student/andrew/Documents/Seg3D/src/models/model_definitions/exactly_nnunet_dsc_CBAM_middler4.json -folds 0 -log /home/student/andrew/Documents/Seg3D/logs/exactly_nnunet_dsc_CBAM_middler4__ADAM.log --_2d

wait

python training_entry.py -d /home/student/andrew/Documents/Seg3D/datasets/main -m /home/student/andrew/Documents/Seg3D/src/models/model_definitions/exactly_nnunet_dsc_CBAM_middler2.json -folds 0 -log /home/student/andrew/Documents/Seg3D/logs/exactly_nnunet_dsc_CBAM_middler2__ADAM.log --_2d

wait

python training_entry.py -d /home/student/andrew/Documents/Seg3D/datasets/main -m /home/student/andrew/Documents/Seg3D/src/models/model_definitions/nnunet_small_with_dsc_CBAM_encoderr2.json -folds 0 -log /home/student/andrew/Documents/Seg3D/logs/nnunet_small_with_dsc_CBAM_encoderr2__ADAM.log --_2d

wait

python training_entry.py -d /home/student/andrew/Documents/Seg3D/datasets/main -m /home/student/andrew/Documents/Seg3D/src/models/model_definitions/nnunet_small_with_dsc_CBAM_middler2.json -folds 0 -log /home/student/andrew/Documents/Seg3D/logs/nnunet_small_with_dsc_CBAM_middler2__ADAM.log --_2d

wait

python training_entry.py -d /home/student/andrew/Documents/Seg3D/datasets/main -m /home/student/andrew/Documents/Seg3D/src/models/model_definitions/unet_dsc_with_pooling_CBAM_middler2.json -folds 0 -log /home/student/andrew/Documents/Seg3D/logs/unet_dsc_with_pooling_CBAM_middler2__ADAM.log --_2d
