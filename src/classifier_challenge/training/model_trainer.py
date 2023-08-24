import os
from model_chooser import ModelChooser, SplittedModel
from DSAL import DSAL
import torch
from torch import nn
import pandas as pd
import warnings
import albumentations as A
import random
import time
from tqdm import tqdm

warnings.filterwarnings("ignore")


class ModelTrainer:

    def __init__(self,
                 current_train_dict,
                 labels,
                 csv,
                 best_save_name,
                 last_save_name,
                 save_dir,
                 images,
                 train_transform=None,
                 val_transform=None,
                 plant_index=None,
                 weight_decay=0,
                 batch_size=32,
                 epochs=20,
                 submit_json={},
                 image_size=224,
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
                 out_name='out.log',
                 tile_size=256,
                 cutmix=True,
                 month_embedding_length=8,
                 year_embedding_length=8,
                 plant_embedding_length=8):

        self.images = images
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
        self.image_size = image_size
        self.tile_size = tile_size
        self.current_train_dict = current_train_dict
        self.cut_mix = cutmix
        self.month_embedding_length = month_embedding_length
        self.year_embedding_length = year_embedding_length
        self.plant_index = plant_index

        # making json to submit
        self.submit_json = submit_json

        self.labels = labels

        if model_to_load != '':
            self.model = torch.load(model_to_load)
        else:
            model_chooser = ModelChooser(model, month_embedding_length=month_embedding_length,
                                         year_embedding_length=year_embedding_length,
                                         plant_embedding_length=plant_embedding_length)
            self.model = model_chooser()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum,
                                         weight_decay=weight_decay)

        print(
            f'momentum: {momentum} --- gamma: {gamma} --- learning rate: {learning_rate} --- weight decay: {weight_decay}')

    def __call__(self):
        print('Program starting...')

        f = open(os.path.join(self.save_dir, self.out_name), 'w')

        val_set = []
        train_set = []

        df = pd.read_csv(self.csv)
        data_dict = df.to_dict(orient='list')

        for image_name in data_dict['val']:
            image_name = str(image_name)
            if image_name != 'nan':
                if int(image_name[5]) in self.months and int(image_name[3]) in self.val:
                    index = ModelTrainer.__search__(image_name, 0, len(self.images), self.images)

                    image = self.images[index][0]

                    if int(image_name[5]) == 4:
                        val_set.append((image, image_name))
                    val_set.append((image, image_name))

        for image_name in data_dict['train']:
            image_name = str(image_name)
            if image_name != 'nan':
                if int(image_name[5]) in self.months and int(image_name[3]) in self.val:
                    index = ModelTrainer.__search__(image_name, 0, len(self.images), self.images)
                    image = self.images[index][0]

                    if int(image_name[5]) == 4:
                        train_set.append((image, image_name))

                    train_set.append((image, image_name))

        train_test_dsal = DSAL(images=train_set,
                               yml=self.labels,
                               read_and_transform_function=ModelTrainer.transform_image_label,
                               cut_mix_function=None,
                               batch_size=self.batch_size,
                               epochs=1,
                               num_processes=self.num_processes,
                               max_queue_size=self.num_processes * 2,
                               transform=self.train_transform,
                               plant_index=self.plant_index)

        train_test_dsal.start()
        train_mean, train_std = ModelTrainer.find_mean_std(train_test_dsal)
        train_test_dsal.join()

        print(f'{train_mean} --- {train_std}')
        f.write(f'train---{train_mean} --- {train_std}\n')

        val_dsal = DSAL(images=val_set,
                        yml=self.labels,
                        read_and_transform_function=ModelTrainer.transform_image_label,
                        cut_mix_function=None,
                        batch_size=self.batch_size,
                        epochs=1,
                        num_processes=self.num_processes,
                        max_queue_size=self.num_processes * 2,
                        transform=self.val_transform,
                        mean=train_mean,
                        std=train_std,
                        plant_index=self.plant_index)

        val_batches = []
        val_dsal.start()

        # start = time.time()

        for _ in tqdm(range(val_dsal.num_batches)):
            # print(f'time taken: {int(time.time() - start)}.....', end='\r')
            val_batches.append(val_dsal.get_item())

        val_dsal.join()

        cut_mix = self.cut_mix_function if self.cut_mix else None

        train_dsal = DSAL(images=train_set,
                          yml=self.labels,
                          plant_index=self.plant_index,
                          read_and_transform_function=ModelTrainer.transform_image_label,
                          cut_mix_function=cut_mix,
                          batch_size=self.batch_size,
                          epochs=self.epochs,
                          num_processes=self.num_processes,
                          max_queue_size=self.num_processes * 2,
                          transform=self.train_transform,
                          mean=train_mean,
                          std=train_std)

        f.write(
            f'momentum: {self.momentum} --- gamma: {self.gamma} --- learning rate: {self.learning_rate} --- weight decay: {self.weight_decay}')

        print('                                                                                            ', end='\r')
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
        for _ in tqdm(range(train_dsal.num_batches)):
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
                self.model.train()

                total = 0
                total_correct = 0
                total_loss = 0
                epoch += 1
                counter = 0
                scheduler.step()

            if epoch == self.unfreeze_epoch:
                self.unfreeze()

            image, label, month, year, plant_index = train_dsal.get_item()
            label = label.type(torch.int64)
            image, label = image.to(self.device), label.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model.forward(image, month, year, plant_index)
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
        loss, accuracy, eval_message = self.evaluate(val_batches, epoch)

        print(message)
        print(eval_message)

        f.write(message)
        f.write(eval_message)

        f.close()

        train_dsal.join()

        torch.save(self.model, self.last_save_name)

        model_to_use = self.best_save_name if best_accuracy >= accuracy else self.last_save_name

        self.store_submit_json(means=train_mean, stds=train_std, model_path=model_to_use)

        return self.submit_json

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
            image, label, m, y, p = batch
            label = label.type(torch.int64)
            image, label = image.to(self.device), label.to(self.device)

            with torch.no_grad():
                outputs = self.model.forward(image, m, y, p)
                outputs = outputs.type(torch.float32)
                loss = self.criterion(outputs, label)

                total_loss += loss.item() * image.size(0)
                total += image.size(0)
                _, prediction = outputs.max(1)

                total_correct += (label == prediction).sum()

        loss = total_loss / total
        accuracy = total_correct / total
        return loss, accuracy, f'Evaluate --- Epoch: {epoch}, Loss: {loss:6.8f}, Accuracy: {accuracy:6.8f}\n'

    def store_submit_json(self, means, stds, model_path):

        means = means.numpy().tolist()
        stds = stds.numpy().tolist()

        if 3 in self.months and 4 in self.months and 5 in self.months:
            self.submit_json['all_month_sizes'].append(self.image_size)
            self.submit_json['all_month_means'].append(means)
            self.submit_json['all_month_stds'].append(stds)
            self.submit_json['all_models_paths'].append(model_path)
        else:
            if 3 in self.months:
                self.submit_json['march_sizes'].append(self.image_size)
                self.submit_json['march_means'].append(means)
                self.submit_json['march_stds'].append(stds)
                self.submit_json['march_models'].append(model_path)

            if 4 in self.months:
                self.submit_json['april_sizes'].append(self.image_size)
                self.submit_json['april_means'].append(means)
                self.submit_json['april_stds'].append(stds)
                self.submit_json['april_models'].append(model_path)

            if 5 in self.months:
                self.submit_json['may_sizes'].append(self.image_size)
                self.submit_json['may_means'].append(means)
                self.submit_json['may_stds'].append(stds)
                self.submit_json['may_models'].append(model_path)

    @staticmethod
    def find_mean_std(test_dsal):
        sum_ = torch.zeros(3)
        sq_sum = torch.zeros(3)
        num_images = 0

        print(f'---finding mean and std ---')

        for _ in tqdm(range(test_dsal.num_batches)):
            item = test_dsal.get_item()
            image = item[0]
            batch = image.size(0)
            sum_ += torch.mean(image, dim=[0, 2, 3]) * batch
            sq_sum += torch.mean(image ** 2, dim=[0, 2, 3]) * batch
            num_images += batch

        mean = sum_ / num_images
        std = ((sq_sum / num_images) - mean ** 2) ** 0.5

        return mean, std

    @staticmethod
    def __search__(x, l, r, arr):
        if l >= r:
            return -1

        m = (l + r) // 2

        if x == arr[m][1]:
            return m

        # item is to the left
        elif x < arr[m][1]:
            return ModelTrainer.__search__(x, l, m, arr)
        # item is to the right
        else:
            return ModelTrainer.__search__(x, m + 1, r, arr)

    @staticmethod
    def get_label(label):
        if label == 'unfertilized':
            return 0
        elif label == '_PKCa':
            return 1
        elif label == 'N_KCa':
            return 2
        elif label == 'NP_Ca':
            return 3
        elif label == 'NPK_':
            return 4
        elif label == 'NPKCa':
            return 5
        else:
            return 6

    def cut_mix_function(self, image, image_name, label, transform, mean=None, std=None):
        year = image_name[3]
        month = image_name[5]

        a = image.copy()

        out_label = ModelTrainer.get_label(label)

        # cut mixing it

        curr_class_arr = self.current_train_dict[year][month][label]

        random_index = random.randint(0, len(curr_class_arr) - 1)

        b = curr_class_arr[random_index].copy()

        temp = A.Compose(
            transforms=[
                A.Resize(1024, 1024)
            ],
            p=1.0,
        )

        augmented = temp(image=a)
        out_image = augmented['image']

        augmented = temp(image=b)
        b = augmented['image']

        i = 0
        while i < (1024 // self.tile_size) - 1:
            j = 0
            while j < (1024 // self.tile_size) - 1:
                new_cell_0 = b[i * self.tile_size: (i + 1) * self.tile_size,
                             j * self.tile_size: (j + 1) * self.tile_size, :]
                new_cell_1 = b[(i + 1) * self.tile_size: (i + 2) * self.tile_size,
                             (j + 1) * self.tile_size: (j + 2) * self.tile_size, :]
                out_image[i * self.tile_size: (i + 1) * self.tile_size, j * self.tile_size: (j + 1) * self.tile_size,
                :] = new_cell_0
                out_image[(i + 1) * self.tile_size: (i + 2) * self.tile_size,
                (j + 1) * self.tile_size: (j + 2) * self.tile_size, :] = new_cell_1
                j += 2
            i += 2

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

    @staticmethod
    def transform_image_label(image, label, transform, mean=None, std=None):
        out_image = image.copy()

        out_label = ModelTrainer.get_label(label)

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
