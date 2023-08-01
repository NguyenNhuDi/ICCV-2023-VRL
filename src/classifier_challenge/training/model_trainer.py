import os
import yaml
from model_chooser import ModelChooser
from DSAL import DSAL
import numpy as np
from PIL import Image
import torch
from torch import nn
import pandas as pd
import warnings
from tqdm import tqdm
import math
import albumentations as A

warnings.filterwarnings("ignore")


class ModelTrainer:

    def __init__(self, yaml_path,
                 best_save_name,
                 last_save_name,
                 save_dir,
                 csv,
                 image_dir_20,
                 image_dir_21,
                 train_transform=None,
                 val_transform=None,
                 weight_decay=0,
                 batch_size=32,
                 epochs=20,
                 num_processes=20,
                 learning_rate=0.01,
                 momentum=0.9,
                 unfreeze_epoch=3,
                 epoch_step=10,
                 gamma=0.85,
                 model_to_load=None,
                 months=[3, 4, 5],
                 train=[0, 1],
                 val=[0, 1],
                 model='efficientnet_b6',
                 model_name='',
                 out_name='out.log'):

        self.image_dir_20 = image_dir_20
        self.image_dir_21 = image_dir_21
        self.best_save_name = best_save_name
        self.last_save_name = last_save_name
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.num_processes = num_processes
        self.unfreeze_epoch = unfreeze_epoch
        self.epoch_step = epoch_step
        self.gamma = gamma
        self.csv = csv
        self.model_name = model_name
        self.months = months
        self.train = train
        self.val = val
        self.out_name = out_name
        self.momentum = momentum
        self.learning_rate = learning_rate

        with open(yaml_path, 'r') as yaml_file:
            self.labels = yaml.safe_load(yaml_file)

        if model_to_load != '':
            self.model = torch.load(model_to_load)
        else:
            model_chooser = ModelChooser(model)
            self.model = model_chooser()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum,
                                         weight_decay=weight_decay)

        self.model.to(self.device)

        print(
            f'momentum: {momentum} --- gamma: {gamma} --- learning rate: {learning_rate} --- weight decay: {weight_decay}')

    def __call__(self):

        print('Program starting...')

        val_set = []
        train_set = []

        df = pd.read_csv(self.csv)
        data_dict = df.to_dict(orient='list')

        for image in data_dict['val']:
            image = str(image)
            if image != 'nan':
                if int(image[5]) in self.months and int(image[3]) in self.val:

                    if image[3] == '0':
                        val_set.append(os.path.join(self.image_dir_20, image))
                    else:
                        val_set.append(os.path.join(self.image_dir_21, image))

        for image in data_dict['train']:
            image = str(image)
            if image != 'nan':
                if int(image[5]) in self.months and int(image[3]) in self.val:
                    if image[3] == '0':
                        train_set.append(os.path.join(self.image_dir_20, image))
                    else:
                        train_set.append(os.path.join(self.image_dir_21, image))

        val_test_dsal = DSAL(val_set,
                             self.labels,
                             ModelTrainer.transform_image_label,
                             batch_size=self.batch_size,
                             epochs=1,
                             num_processes=self.num_processes,
                             max_queue_size=self.num_processes * 2,
                             transform=self.val_transform)

        val_test_dsal.start()

        val_mean, val_std = ModelTrainer.find_mean_std(val_test_dsal)

        val_test_dsal.join()

        print(f'{val_mean} --- {val_std}')

        train_test_dsal = DSAL(train_set,
                               self.labels,
                               ModelTrainer.transform_image_label,
                               batch_size=self.batch_size,
                               epochs=1,
                               num_processes=self.num_processes,
                               max_queue_size=self.num_processes * 2,
                               transform=self.train_transform)

        train_test_dsal.start()
        train_mean, train_std = ModelTrainer.find_mean_std(train_test_dsal)
        train_test_dsal.join()

        print(f'{train_mean} --- {train_std}')

        val_dsal = DSAL(val_set,
                        self.labels,
                        ModelTrainer.transform_image_label,
                        batch_size=self.batch_size,
                        epochs=1,
                        num_processes=self.num_processes,
                        max_queue_size=self.num_processes * 2,
                        transform=self.val_transform,
                        mean=val_mean,
                        std=val_std)

        val_batches = []
        val_dsal.start()

        for i in tqdm(range(val_dsal.num_batches)):
            val_batches.append(val_dsal.get_item())

        val_dsal.join()

        train_dsal = DSAL(train_set,
                          self.labels,
                          ModelTrainer.transform_image_label,
                          batch_size=self.batch_size,
                          epochs=self.epochs,
                          num_processes=self.num_processes,
                          max_queue_size=self.num_processes * 2,
                          transform=self.train_transform,
                          mean=train_mean,
                          std=train_std)

        f = open(os.path.join(self.save_dir, self.out_name), 'w')

        f.write(
            f'momentum: {self.momentum} --- gamma: {self.gamma} --- learning rate: {self.learning_rate} --- weight decay: {self.weight_decay}')

        print('starting pathing...')
        train_dsal.start()
        print('pathing finished')

        print(f'\n\n\n------------{self.model_name}--------------\n\n\n')
        f.write(f'\n\n\n------------{self.model_name}--------------\n\n\n')

        self.freeze()

        counter = 0
        batches_per_epoch = train_dsal.num_batches // self.epochs
        epoch = 0

        total = 0
        total_correct = 0
        total_loss = 0

        best_loss = 1000
        best_accuracy = 0
        best_epoch = 0

        torch.set_grad_enabled(True)

        # scheduler: optimizer, step size, gamma
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.epoch_step, self.gamma)

        print('start training')
        for i in tqdm(range(train_dsal.num_batches)):

            if counter == batches_per_epoch:
                total_loss = total_loss / total
                accuracy = total_correct / total
                message = f'Training --- Epoch: {epoch}, Loss: {total_loss:6.8f}, Accuracy: {accuracy:6.8f}\n'
                current_loss, current_accuracy, print_output = self.evaluate(val_batches, epoch)
                print(message)
                print(print_output)
                f.write(message)
                f.write(print_output)

                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_epoch = epoch

                    torch.save(self.model, self.best_save_name)

                if current_loss < best_loss:
                    best_loss = current_loss

                message = f'Best epoch: {best_epoch}, Best Loss: {best_loss:6.8f}, Best Accuracy: {best_accuracy:6.8f}\n'
                print(message)
                f.write(message)
                # get_last_lr()
                self.model.train()

                total = 0
                total_correct = 0
                total_loss = 0
                epoch += 1
                counter = 0
                scheduler.step()

            if epoch == self.unfreeze_epoch:
                self.unfreeze()

            image, label = train_dsal.get_item()
            label = label.type(torch.int64)
            image, label = image.to(self.device), label.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(image)
            loss = self.criterion(outputs, label)

            if epoch < self.unfreeze_epoch:
                loss.requires_grad = True

            loss.backward()
            self.optimizer.step()
            total += image.size(0)
            _, predictions = outputs.max(1)
            total_correct += (predictions == label).sum()
            total_loss += loss.item() * image.size(0)

            counter += 1

        total_loss = total_loss / total
        accuracy = total_correct / total

        message = f'Training --- Epoch: {epoch}, Loss: {total_loss:6.8f}, Accuracy: {accuracy:6.8f}\n'
        _, _, eval_message = self.evaluate(val_batches, epoch)

        print(message)
        print(eval_message)

        f.write(message)
        f.write(eval_message)

        f.close()

        train_dsal.join()

        torch.save(self.model, self.last_save_name)

    def freeze(self):
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def unfreeze(self):
        for parameter in self.model.parameters():
            parameter.requires_grad = True

    def evaluate(self, val_batches, epoch):
        self.model.eval()
        total_correct = 0
        total_loss = 0
        total = 0

        for batch in val_batches:
            image, label = batch
            label = label.type(torch.int64)
            image, label = image.to(self.device), label.to(self.device)

            with torch.no_grad():
                outputs = self.model(image)
                outputs = outputs.type(torch.float32)
                loss = self.criterion(outputs, label)

                total_loss += loss.item() * image.size(0)
                total += image.size(0)
                _, prediction = outputs.max(1)

                total_correct += (label == prediction).sum()

        loss = total_loss / total
        accuracy = total_correct / total
        return loss, accuracy, f'Evaluate --- Epoch: {epoch}, Loss: {loss:6.8f}, Accuracy: {accuracy:6.8f}\n'

    @staticmethod
    def find_mean_std(test_dsal):
        sum_ = torch.zeros(3)
        sq_sum = torch.zeros(3)
        num_images = 0

        print(f'---finding mean and std ---')

        for _ in tqdm(range(test_dsal.num_batches)):
            image, _ = test_dsal.get_item()
            batch = image.size(0)
            sum_ += torch.mean(image, dim=[0, 2, 3]) * batch
            sq_sum += torch.mean(image ** 2, dim=[0, 2, 3]) * batch
            num_images += batch

        mean = sum_ / num_images
        std = ((sq_sum / num_images) - mean ** 2) ** 0.5

        return mean, std

    @staticmethod
    def transform_image_label(image_path, label, transform, mean=None, std=None):
        out_image = np.array(Image.open(image_path), dtype='uint8')

        if label == 'unfertilized':
            out_label = 0
        elif label == '_PKCa':
            out_label = 1
        elif label == 'N_KCa':
            out_label = 2
        elif label == 'NP_Ca':
            out_label = 3
        elif label == 'NPK_':
            out_label = 4
        elif label == 'NPKCa':
            out_label = 5
        else:
            out_label = 6

        if transform is not None:
            augmented = transform(image=out_image)
            out_image = augmented['image']

        if mean is not None and std is not None:
            temp = A.Compose(
                transforms=[
                    A.Normalize(mean=mean, std=std, max_pixel_value=1.0)
                ],
                p=1.0
            )

            augmented = temp(image=out_image)
            out_image = augmented['image']

        # converting the image and mask into tensors

        out_image = torch.from_numpy(out_image).permute(2, 0, 1)
        out_label = torch.tensor(out_label)

        return out_image, out_label
