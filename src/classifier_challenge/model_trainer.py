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

warnings.filterwarnings("ignore")


class ModelTrainer:

    def __init__(self, yaml_path,
                 best_save_name,
                 last_save_name,
                 train_transform=None,
                 val_transform=None,
                 batch_size=32,
                 epochs=20,
                 num_processes=20,
                 learning_rate=0.01,
                 momentum=0.9,
                 unfreeze_epoch=3,
                 epoch_step=10,
                 gamma=0.85,
                 csv_20=None,
                 csv_21=None,
                 image_dir_20=None,
                 image_dir_21=None,
                 model_to_load=None,
                 model='efficientnet_b6',
                 model_name=''):

        assert csv_20 is not None or csv_21 is not None, 'At least one csv dataset must be passed in'

        if csv_20 is not None:
            assert image_dir_20 is not None, 'csv_20 is given but not its image directory'

        if csv_21 is not None:
            assert image_dir_21 is not None, 'csv_21 is given but not its image directory'

        self.image_dir_20 = image_dir_20
        self.image_dir_21 = image_dir_21
        self.best_save_name = best_save_name
        self.last_save_name = last_save_name
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_processes = num_processes
        self.unfreeze_epoch = unfreeze_epoch
        self.epoch_step = epoch_step
        self.gamma = gamma
        self.csv_20 = csv_20
        self.csv_21 = csv_21
        self.model_name = model_name

        with open(yaml_path, 'r') as yaml_file:
            self.labels = yaml.safe_load(yaml_file)

        if model_to_load is not None:
            self.model = torch.load(model_to_load)
        else:
            model_chooser = ModelChooser(model)
            self.model = model_chooser()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)

        self.model.to(self.device)

    def __call__(self):

        val_set = []
        train_set = []

        # get the 2020 train and val set
        if self.csv_20 is not None:
            df = pd.read_csv(self.csv_20)
            data_dict_20 = df.to_dict(orient='list')

            for image in data_dict_20['val']:
                image = str(image)
                if image != 'nan':
                    val_set.append(os.path.join(self.image_dir_20, image))

            for image in data_dict_20['train']:
                image = str(image)
                if image != 'nan':
                    train_set.append(os.path.join(self.image_dir_20, image))

        # get the 2021 train and val set
        if self.csv_21 is not None:
            df = pd.read_csv(self.csv_21)
            data_dict_21 = df.to_dict(orient='list')

            for image in data_dict_21['val']:
                image = str(image)
                if image != 'nan':
                    val_set.append(os.path.join(self.image_dir_21, image))

            for image in data_dict_21['train']:
                image = str(image)
                if image != 'nan':
                    train_set.append(os.path.join(self.image_dir_21, image))

        val_dsal = DSAL(val_set,
                        self.labels,
                        ModelTrainer.transform_image_label,
                        batch_size=self.batch_size,
                        epochs=1,
                        num_processes=self.num_processes,
                        max_queue_size=self.num_processes * 2,
                        transform=self.val_transform)

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
                          transform=self.train_transform)

        print('starting pathing...')
        train_dsal.start()
        print('pathing finished')

        print(f'\n\n\n------------{self.model_name}--------------\n\n\n')

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
                print(f'Training --- Epoch: {epoch}, Loss: {total_loss:6.4f}, Accuracy: {accuracy:6.4f}')
                current_loss, current_accuracy = self.evaluate(val_batches, epoch)

                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_epoch = epoch

                    torch.save(self.model, self.best_save_name)

                if current_loss < best_loss:
                    best_loss = current_loss

                print(f'Best epoch: {best_epoch}, Best Loss: {best_loss:6.4f}, Best Accuracy: {best_accuracy:6.4f}')
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

        print(f'Training --- Epoch: {epoch}, Loss: {total_loss:6.4f}, Accuracy: {accuracy:6.4f}')
        self.evaluate(val_batches, epoch)

        train_dsal.join()

        torch.save(self.model, self.last_save_name)

    @staticmethod
    def transform_image_label(image_path, label, transform):
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

        # converting the image and mask into tensors

        out_image = torch.from_numpy(out_image).permute(2, 0, 1)
        out_label = torch.tensor(out_label)

        return out_image, out_label

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

        print(f'Evaluate --- Epoch: {epoch}, Loss: {loss:6.4f}, Accuracy: {accuracy:6.4f}')
        return loss, accuracy
