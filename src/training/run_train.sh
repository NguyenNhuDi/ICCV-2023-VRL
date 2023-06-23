
#!/bin/bash

python training_entry.py -d /home/student/andrew/Documents/Seg3D/datasets/main -m /home/student/andrew/Documents/Seg3D/src/models/model_definitions/nnunet_small_with_dsc.json -folds 0 -log /home/student/andrew/Documents/Seg3D/logs/nnunet_small_with_dsc__ADAM.log --_2d

wait

python training_entry.py -d /home/student/andrew/Documents/Seg3D/datasets/main -m /home/student/andrew/Documents/Seg3D/src/models/model_definitions/nnunet_with_xmodule__37.json -folds 0 -log /home/student/andrew/Documents/Seg3D/logs/nnunet_with_xmodule__37__ADAM.log --_2d

wait

python training_entry.py -d /home/student/andrew/Documents/Seg3D/datasets/main -m /home/student/andrew/Documents/Seg3D/src/models/model_definitions/unet_dsc_with_pooling.json -folds 0 -log /home/student/andrew/Documents/Seg3D/logs/unet_dsc_with_pooling__maxpool__ADAM.log --_2d

wait 

python training_entry.py -d /home/student/andrew/Documents/Seg3D/datasets/main -m /home/student/andrew/Documents/Seg3D/src/models/model_definitions/nnunet_small_with_dsc.json -folds 0 -log /home/student/andrew/Documents/Seg3D/logs/nnunet_small_with_dsc__SGD.log --_2d --sgd

wait

python training_entry.py -d /home/student/andrew/Documents/Seg3D/datasets/main -m /home/student/andrew/Documents/Seg3D/src/models/model_definitions/nnunet_with_xmodule__37.json -folds 0 -log /home/student/andrew/Documents/Seg3D/logs/nnunet_with_xmodule__37__SGD.log --_2d --sgd

wait

python training_entry.py -d /home/student/andrew/Documents/Seg3D/datasets/main -m /home/student/andrew/Documents/Seg3D/src/models/model_definitions/nnunet_with_xmodule__samekernel[3-7].json -folds 0 -log /home/student/andrew/Documents/Seg3D/logs/nnunet_with_xmodule_samekernel[3-7]__ADAM.log --_2d

wait

python training_entry.py -d /home/student/andrew/Documents/Seg3D/datasets/main -m /home/student/andrew/Documents/Seg3D/src/models/model_definitions/exactly_nnunet_dsc.json -folds 0 -log /home/student/andrew/Documents/Seg3D/logs/exactly_nnunet_dsc__SGD.log --_2d --sgd

wait

python training_entry.py -d /home/student/andrew/Documents/Seg3D/datasets/main -m /home/student/andrew/Documents/Seg3D/src/models/model_definitions/nnunet_with_xmodule__samekernel[3-7].json -folds 0 -log /home/student/andrew/Documents/Seg3D/logs/nnunet_with_xmodule_samekernel[3-7]__SGD.log --_2d --sgd

wait

python training_entry.py -d /home/student/andrew/Documents/Seg3D/datasets/main -m nnUNet -folds 0 -log /home/student/andrew/Documents/Seg3D/logs/nnUNet_SGD.log --_2d --sgd

wait

python training_entry.py -d /home/student/andrew/Documents/Seg3D/datasets/main -m /home/student/andrew/Documents/Seg3D/src/models/model_definitions/nnunet_with_xmodule__357.json  -folds 0 -log /home/student/andrew/Documents/Seg3D/logs/nnunet_with_xmodule__357__ADAM.log --_2d

wait

python training_entry.py -d /home/student/andrew/Documents/Seg3D/datasets/main -m /home/student/andrew/Documents/Seg3D/src/models/model_definitions/nnunet_with_xmodule__357.json  -folds 0 -log /home/student/andrew/Documents/Seg3D/logs/nnunet_with_xmodule__357__SGD.log --_2d --sgd

wait

python training_entry.py -d /home/student/andrew/Documents/Seg3D/datasets/main -m /home/student/andrew/Documents/Seg3D/src/models/model_definitions/nnunet_with_xmodule__35.json  -folds 0 -log /home/student/andrew/Documents/Seg3D/logs/nnunet_with_xmodule__35__ADAM.log --_2d

wait

python training_entry.py -d /home/student/andrew/Documents/Seg3D/datasets/main -m /home/student/andrew/Documents/Seg3D/src/models/model_definitions/nnunet_with_xmodule__samekernel[3-7]_relubetween.json -folds 0 -log /home/student/andrew/Documents/Seg3D/logs/nnunet_with_xmodule__samekernel[3-7]_relubetween__ADAM.log --_2d
