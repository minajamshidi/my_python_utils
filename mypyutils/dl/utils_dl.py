
# region packages

import os.path as op
from colorama import Fore
import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset
from PIL import Image

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

from ..utils_general import *

# endregion


class ImageDataset(Dataset):
    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    def __init__(self, x, y, transform=None, xpath=True):
        """

        :param x: a list/ndarray of the names of the image files
        :param y: a list/ndarray of the labels of the images
        :param transform:The transformation object to be applied to the images
        """
        super().__init__()
        self.image_labels = y
        self.xpath = xpath
        self.images = x
        self.transform = transform

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        """
        get the image of index idx
        :param idx:
        :return:
        """
        image = Image.open(self.images[idx]) if self.xpath else self.images[idx]
        # image = plt.imread(self.img_path[idx])
        label = self.image_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class AddNoise(object):
    def __init__(self, noise_std=0.5):
        self.noise_std = noise_std

    def __call__(self, data):
        return data + self.noise_std * torch.randn(data.size())


class MaxNormalize(object):
    def __init__(self, norm_max=255):
        self.norm_max = norm_max

    def __call__(self, img):
        return img / self.norm_max

class Clip(object):
    def __init__(self, min_=0, max_=1):
        self.min_ = min_
        self.max_ = max_

    def __call__(self, img):
        return torch.clip(img, self.min_, self.max_)

def lr_schedule(optimizer, current_epoch_num, lr_init):
    lr = lr_init / 10 ** np.floor(current_epoch_num / 10)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_cnn_2class_classification(model, loss_func, optimizer, n_epochs, lr_init, device, dataloader_train,
                                    dataloader_val, min_epoch_num=0, lr_schedule1=True, n_epoch_val=1,
                                    early_stopping=True, patience=None, stop_criterion='loss',
                                    save_path=None):
    # todo: measure performance with accuracy
    n_train_loader = len(dataloader_train)
    n_val_loader = len(dataloader_val)

    patience = n_epoch_val if patience is None else patience

    # region containers for the results ---------------------------

    # all mini-batch train and val AUC and loss values
    train_auc_all_minibatches = []
    val_auc_all_minibatches = []
    loss_all_minibatches = []

    # average of the auc and loss in each epoch
    train_auc_epoch = []
    val_auc_epoch = []

    train_loss_epoch = []
    val_loss_epoch = []

    train_auc_epoch_2 = []  # like a cumsum
    val_auc_epoch_2 = []

    # to find the best epoch - in case of early stopping
    min_loss = np.Inf
    max_auc = 0
    best_epoch = 0

    stop_cr_counter = 0
    best_model = copy.deepcopy(model)
    # endregion

    for e in range(n_epochs):
        print(f'epoch #{e} and stop criterion is {stop_cr_counter} *****')
        if lr_schedule1:
            lr_schedule(optimizer, e, lr_init)
        model.train()
        train_loss = 0
        train_auc_e = 0
        val_loss_e = 0
        val_auc_e = 0

        # region mini-batch training -------------------
        # i = 0
        for i_img1, (images1, labels1) in enumerate(dataloader_train):
            # if i == 0:
            #     break
            labels1 = labels1.to(torch.int32)
            images1 = images1.to(device)
            labels1 = labels1.to(device)

            labels_pred1 = model(images1)
            loss = loss_func(labels_pred1, labels1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_all_minibatches.append(loss.item())
            lbls_gt1 = labels1.cpu().numpy()

            train_loss += (loss.item() / n_train_loader)
            lbl_pred1 = labels_pred1[:, -1].detach().cpu().numpy()
            auc_train = roc_auc_score(lbls_gt1, lbl_pred1)
            train_auc_all_minibatches.append(auc_train)
            train_auc_e += auc_train / n_train_loader
            if not i_img1 % 50:
                print(f'batch{i_img1} / {n_train_loader} of train')

        # collect the epoch training metrics ----------------
        train_loss_epoch.append(train_loss)
        train_auc_epoch.append(train_auc_e)
        train_auc_epoch_2.append(np.mean(train_auc_all_minibatches))
        print(f'train AUC={train_auc_e}')
        # endregion -------------------

        # if save_path:
        #     torch.save(model.state_dict(), op.join(save_path, 'model_epoch_' + str(e) + '.pt'))

        # region mini-batch validation ---------------------------
        if not e % n_epoch_val:  # validation for every {n_epoch_val} epochs
            model.eval()
            # with torch.no_grad():
            # i = 0
            for i_img, (images, labels) in enumerate(dataloader_val):
                # if i == 0:
                #     break
                lbls_gt = labels.numpy()

                labels = labels.to(torch.int32)
                images = images.to(device)
                labels = labels.to(device)

                labels_pred = model(images)
                lbl_pred = labels_pred[:, -1].detach().cpu().numpy()
                auc_val = roc_auc_score(lbls_gt, lbl_pred)
                val_auc_all_minibatches.append(auc_val)
                val_auc_e += auc_val / n_val_loader

                loss = loss_func(labels_pred, labels)
                val_loss_e += loss.item() / n_val_loader

                if not i_img % 50:
                    print(f'batch{i_img} / {n_val_loader} of validation')

            # collect the validation metrics ---------------------------
            val_loss_epoch.append(val_loss_e)
            val_auc_epoch.append(val_auc_e)
            val_auc_epoch_2.append(np.mean(val_auc_all_minibatches))

            # region collect the best model ---------------------------
            if e >= min_epoch_num and early_stopping:
                if stop_criterion == 'loss' and val_loss_e <= min_loss:
                    stop_cr_counter = 0
                    min_loss = val_loss_e
                    best_epoch = e
                    best_model = copy.deepcopy(model)
                elif stop_criterion == 'auc' and val_auc_e >= max_auc:
                    stop_cr_counter = 0
                    max_auc = val_auc_e
                    best_epoch = e
                    best_model = copy.deepcopy(model)
                else:
                    stop_cr_counter += 1

            if stop_cr_counter >= patience and early_stopping:
                break
            # endregion
        # endregion

        # print(f'Epoch {e + 1} with train loss={train_loss} and validation loss={val_loss_e}')

    # region organize results and model params ---------------------------
    training_performance = dict()
    training_performance['train_auc_all_minibatches'] = train_auc_all_minibatches
    training_performance['val_auc_all_minibatches'] = val_auc_all_minibatches
    training_performance['loss_all_minibatches'] = loss_all_minibatches

    training_performance['train_auc_epoch'] = train_auc_epoch
    training_performance['val_auc_epoch'] = val_auc_epoch
    training_performance['train_loss_epoch'] = train_loss_epoch
    training_performance['val_loss_epoch'] = val_loss_epoch
    training_performance['train_auc_epoch_2'] = train_auc_epoch_2
    training_performance['val_auc_epoch_2'] = val_auc_epoch_2

    training_performance['best_epoch'] = best_epoch

    # model parameters
    training_performance['min_epoch_num'] = min_epoch_num
    training_performance['lr_schedule'] = lr_schedule1
    training_performance['n_epoch_val'] = n_epoch_val
    training_performance['early_stopping'] = early_stopping
    training_performance['patience'] = patience
    training_performance['stop_criterion'] = stop_criterion

    # endregion

    if save_path:
        save_pickle(op.join(save_path, 'training_performance'), training_performance)

    return training_performance, model, best_model

def train_model(model, loss_func, optimizer, n_epochs, lr_init, device, dataloader_train,
                dataloader_val, min_epoch_num=0, lr_schedule1=True, n_epoch_val=1,
                early_stopping=True, patience=None, stop_criterion='loss',
                save_path=None):
    # todo: measure performance with accuracy
    n_train_loader = len(dataloader_train)
    n_val_loader = len(dataloader_val)

    patience = n_epoch_val if patience is None else patience

    # region containers for the results ---------------------------

    # all mini-batch train and val AUC and loss values
    loss_all_minibatches = []

    # average of the loss in each epoch
    train_loss_epoch = []
    val_loss_epoch = []

    # to find the best epoch - in case of early stopping
    min_loss = np.Inf
    max_auc = 0
    best_epoch = 0

    stop_cr_counter = 0
    best_model = copy.deepcopy(model)
    # endregion

    for e in range(n_epochs):
        print(f'epoch #{e} and stop criterion is {stop_cr_counter} *****')
        if lr_schedule1:
            lr_schedule(optimizer, e, lr_init)
        model.train()
        train_loss = 0
        train_auc_e = 0
        val_loss_e = 0
        val_auc_e = 0

        # region mini-batch training -------------------
        # i = 0
        for i_img1, (images1, labels1) in enumerate(dataloader_train):
            # if i == 0:
            #     break
            labels1 = labels1.to(torch.float32)
            images1 = images1.to(device)
            labels1 = labels1.to(device)

            labels_pred1 = model(images1)
            loss = loss_func(labels_pred1, labels1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_all_minibatches.append(loss.item())
            train_loss += (loss.item() / n_train_loader)

            if not i_img1 % 50:
                print(f'batch{i_img1} / {n_train_loader} of train')

        # collect the epoch training metrics ----------------
        train_loss_epoch.append(train_loss)
        # endregion -------------------

        # if save_path:
        #     torch.save(model.state_dict(), op.join(save_path, 'model_epoch_' + str(e) + '.pt'))
        print(f'the train loss in this epoch = {train_loss_epoch[e]}')

        # region mini-batch validation ---------------------------
        if not e % n_epoch_val:  # validation for every {n_epoch_val} epochs
            model.eval()
            # with torch.no_grad():
            # i = 0
            for i_img, (images, labels) in enumerate(dataloader_val):
                # if i == 0:
                #     break
                labels = labels.to(torch.float32)
                images = images.to(device)
                labels = labels.to(device)

                labels_pred = model(images)

                loss = loss_func(labels_pred, labels)
                val_loss_e += loss.item() / n_val_loader

                if not i_img % 50:
                    print(f'batch{i_img} / {n_val_loader} of validation')

            # collect the validation metrics ---------------------------
            val_loss_epoch.append(val_loss_e)
            print(f'the validation loss in this epoch = {val_loss_epoch[e]}')

            # region collect the best model ---------------------------
            if e >= min_epoch_num and early_stopping:
                if stop_criterion == 'loss' and val_loss_e <= min_loss:
                    stop_cr_counter = 0
                    min_loss = val_loss_e
                    best_epoch = e
                    best_model = copy.deepcopy(model)
                else:
                    stop_cr_counter += 1

            if stop_cr_counter >= patience and early_stopping:
                break
            # endregion
        # endregion



        # print(f'Epoch {e + 1} with train loss={train_loss} and validation loss={val_loss_e}')

    # region organize results and model params ---------------------------
    training_performance = dict()
    training_performance['loss_all_minibatches'] = loss_all_minibatches

    training_performance['train_loss_epoch'] = train_loss_epoch
    training_performance['val_loss_epoch'] = val_loss_epoch

    training_performance['best_epoch'] = best_epoch

    # model parameters
    training_performance['min_epoch_num'] = min_epoch_num
    training_performance['lr_schedule'] = lr_schedule1
    training_performance['n_epoch_val'] = n_epoch_val
    training_performance['early_stopping'] = early_stopping
    training_performance['patience'] = patience
    training_performance['stop_criterion'] = stop_criterion

    # endregion

    if save_path:
        save_pickle(op.join(save_path, 'training_performance'), training_performance)

    return training_performance, model, best_model


def train_autoencoder(model, loss_func, optimizer, n_epochs, lr_init, device, dataloader_train,
                      dataloader_val, min_epoch_num=0, lr_schedule1=True, n_epoch_val=1,
                      early_stopping=True, patience=None, stop_criterion='loss',
                      save_path=None):
    # ToDo: give access to the encoded features
    # TODO: measure performance with accuracy
    n_train_loader = len(dataloader_train)
    n_val_loader = len(dataloader_val)

    patience = n_epoch_val if patience is None else patience

    # region containers for the results ---------------------------

    # all mini-batch train and val AUC and loss values
    loss_all_minibatches = []

    train_loss_epoch = []
    val_loss_epoch = []

    # to find the best epoch - in case of early stopping
    min_loss = np.Inf
    best_epoch = 0

    stop_cr_counter = 0
    best_model = copy.deepcopy(model)
    # endregion

    for e in range(n_epochs):
        print(f'epoch #{e} and stop criterion is {stop_cr_counter} *****')
        if lr_schedule1:
            lr_schedule(optimizer, e, lr_init)
        model.train()
        train_loss = 0
        val_loss_e = 0

        # region mini-batch training -------------------
        # i = 0
        for i_img1, (data1, _) in enumerate(dataloader_train):
            # if i == 0:
            #     break
            data1 = data1.to(device)

            model_output = model(data1)
            loss = loss_func(model_output, data1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_all_minibatches.append(loss.item())

            train_loss += (loss.item() / n_train_loader)
            if not i_img1 % 50:
                print(f'batch{i_img1} / {n_train_loader} of train')

        # collect the epoch training metrics ----------------
        train_loss_epoch.append(train_loss)

        # endregion -------------------

        # if save_path:
        #     torch.save(model.state_dict(), op.join(save_path, 'model_epoch_' + str(e) + '.pt'))

        # region mini-batch validation ---------------------------
        if not e % n_epoch_val:  # validation for every {n_epoch_val} epochs
            model.eval()
            # with torch.no_grad():
            # i = 0
            for i_img, (data, _) in enumerate(dataloader_val):
                # if i == 0:
                #     break
                data = data.to(device)

                model_out_val = model(data)
                loss = loss_func(model_out_val, data)
                val_loss_e += loss.item() / n_val_loader

                if not i_img % 50:
                    print(f'batch{i_img} / {n_val_loader} of validation')

            # collect the validation metrics ---------------------------
            val_loss_epoch.append(val_loss_e)

            # region collect the best model ---------------------------
            if e >= min_epoch_num and early_stopping:
                if stop_criterion == 'loss' and val_loss_e <= min_loss:
                    stop_cr_counter = 0
                    min_loss = val_loss_e
                    best_epoch = e
                    best_model = copy.deepcopy(model)
                else:
                    stop_cr_counter += 1

            if stop_cr_counter >= patience and early_stopping:
                break
            # endregion
        # endregion

        # print(f'Epoch {e + 1} with train loss={train_loss} and validation loss={val_loss_e}')

    # region organize results and model params ---------------------------
    training_performance = dict()
    training_performance['loss_all_minibatches'] = loss_all_minibatches

    training_performance['train_loss_epoch'] = train_loss_epoch
    training_performance['val_loss_epoch'] = val_loss_epoch

    training_performance['best_epoch'] = best_epoch

    # model parameters
    training_performance['min_epoch_num'] = min_epoch_num
    training_performance['lr_schedule'] = lr_schedule1
    training_performance['n_epoch_val'] = n_epoch_val
    training_performance['early_stopping'] = early_stopping
    training_performance['patience'] = patience
    training_performance['stop_criterion'] = stop_criterion

    # endregion

    if save_path:
        save_pickle(op.join(save_path, 'training_performance'), training_performance)

    return training_performance, model, best_model


def train_denoiser_autoencoder(model, loss_func, optimizer, n_epochs, lr_init, device, dataloader_train,
                      dataloader_val, min_epoch_num=0, lr_schedule1=True, n_epoch_val=1,
                      early_stopping=True, patience=None, stop_criterion='loss',
                      save_path=None):
    # todo: measure performance with accuracy
    n_train_loader = len(dataloader_train)
    n_val_loader = len(dataloader_val)

    patience = n_epoch_val if patience is None else patience

    # region containers for the results ---------------------------

    # all mini-batch train and val AUC and loss values
    loss_all_minibatches = []

    train_loss_epoch = []
    val_loss_epoch = []

    # to find the best epoch - in case of early stopping
    min_loss = np.Inf
    best_epoch = 0

    stop_cr_counter = 0
    best_model = copy.deepcopy(model)
    # endregion
    transforms1 = transforms.Compose(AddNoise(0.5), Clip(0, 1))
    for e in range(n_epochs):
        print(f'epoch #{e} and stop criterion is {stop_cr_counter} *****')
        if lr_schedule1:
            lr_schedule(optimizer, e, lr_init)
        model.train()
        train_loss = 0
        val_loss_e = 0

        # region mini-batch training -------------------
        # i = 0
        for i_img1, (data1, _) in enumerate(dataloader_train):
            # if i == 0:
            #     break

            data1 = data1.to(device)
            data1_noisy = transforms1(data1)

            model_output = model(data1_noisy)
            loss = loss_func(model_output, data1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_all_minibatches.append(loss.item())

            train_loss += (loss.item() / n_train_loader)
            if not i_img1 % 50:
                print(f'batch{i_img1} / {n_train_loader} of train')

        # collect the epoch training metrics ----------------
        train_loss_epoch.append(train_loss)

        # endregion -------------------

        # if save_path:
        #     torch.save(model.state_dict(), op.join(save_path, 'model_epoch_' + str(e) + '.pt'))

        # region mini-batch validation ---------------------------
        if not e % n_epoch_val:  # validation for every {n_epoch_val} epochs
            model.eval()
            # with torch.no_grad():
            # i = 0
            for i_img, (data, _) in enumerate(dataloader_val):
                # if i == 0:
                #     break
                data = data.to(device)
                data_noisy = transforms1(data)

                model_out_val = model(data_noisy)
                loss = loss_func(model_out_val, data)
                val_loss_e += loss.item() / n_val_loader

                if not i_img % 50:
                    print(f'batch{i_img} / {n_val_loader} of validation')

            # collect the validation metrics ---------------------------
            val_loss_epoch.append(val_loss_e)

            # region collect the best model ---------------------------
            if e >= min_epoch_num and early_stopping:
                if stop_criterion == 'loss' and val_loss_e <= min_loss:
                    stop_cr_counter = 0
                    min_loss = val_loss_e
                    best_epoch = e
                    best_model = copy.deepcopy(model)
                else:
                    stop_cr_counter += 1

            if stop_cr_counter >= patience and early_stopping:
                break
            # endregion
        # endregion

        # print(f'Epoch {e + 1} with train loss={train_loss} and validation loss={val_loss_e}')

    # region organize results and model params ---------------------------
    training_performance = dict()
    training_performance['loss_all_minibatches'] = loss_all_minibatches

    training_performance['train_loss_epoch'] = train_loss_epoch
    training_performance['val_loss_epoch'] = val_loss_epoch

    training_performance['best_epoch'] = best_epoch

    # model parameters
    training_performance['min_epoch_num'] = min_epoch_num
    training_performance['lr_schedule'] = lr_schedule1
    training_performance['n_epoch_val'] = n_epoch_val
    training_performance['early_stopping'] = early_stopping
    training_performance['patience'] = patience
    training_performance['stop_criterion'] = stop_criterion

    # endregion

    if save_path:
        save_pickle(op.join(save_path, 'training_performance'), training_performance)

    return training_performance, model, best_model

def freeze_layers(model, layers_unfrozen, printing=False):
    """

    :param model:
    :param layers_unfrozen: a list of strings with the names of the layers to stay unfrozen, other layers will be frozen
    :param printing:
    :return:
    """
    for name, child in model.named_children():
        if name in layers_unfrozen:
            print(name + ' is unfrozen') if printing else None
            for param in child.parameters():
                param.requires_grad = True
        else:
            print(name + ' is frozen') if printing else None
            for param in child.parameters():
                param.requires_grad = False


def ensemble_voting_classification(models_, data, strategy='plurality'):
    for model in models_:
        labels_pred = model(data).detach().cpu().numpy()
        lbl_pred = np.argmax(labels_pred, axis=-1)
        
