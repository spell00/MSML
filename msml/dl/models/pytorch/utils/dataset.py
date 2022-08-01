#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

import os
import csv
from PIL import Image
import torch
import itertools
import numpy as np
from torch.utils.data import Dataset
import random
import pandas as pd
from sklearn.preprocessing import minmax_scale as scale
from sklearn.preprocessing import MinMaxScaler, RobustScaler, Normalizer, StandardScaler
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def read_csv(path):
    with open(path, 'r', encoding='utf-8') as csv_file:
        rows = csv.DictReader(csv_file)
        data = []
        labels = []
        row = {}
        for i, row in enumerate(rows):
            labels += [list(row.values())[0]]
            try:
                data += [np.array(list(row.values())[1:], dtype=np.float)]
            except:
                tmp = np.array(list(row.values())[1:])
                tmp = np.array([x if x is not None else 0. for x in tmp])
                tmp = np.array([float(x) if x != '' else 0. for x in tmp])
                data += [tmp]

    data = np.stack(data)
    if data.shape[0] < 35:
        pass
    data[np.isnan(data)] = 0
    return pd.DataFrame(data, index=labels, columns=list(row.keys())[1:])


# scaling data
def ms_data(path):
    mat_datas = []
    labels = []
    for fname in os.listdir(path)[:120]:
        mat_data = read_csv(f"{path}/{fname}")
        labels += [fname.split('.csv')[0]]
        mat_data = np.asarray(mat_data)
        mat_data = scale(mat_data)
        mat_datas += [mat_data]
    return mat_datas, labels


class MSData:
    def __init__(self, path, test=False):
        self.path = path
        if not test:
            self.fnames = os.listdir(path)
        else:
            tmp = os.listdir(path)
            np.random.shuffle(tmp)
            self.fnames = tmp[:57]

    def process(self, i):
        fname = self.fnames[i]
        print(f"Processing sample #{i}: {fname}")
        mat_data = read_csv(f"{self.path}/{fname}")
        label = fname.split('_')[0]
        mat_data = np.asarray(mat_data)
        return mat_data, label

    def __len__(self):
        return len(self.fnames)


class MSImages:
    def __init__(self, path, binarize=False, resize=True, test=False):
        self.path = path
        self.binarize = binarize
        self.resize = resize
        if not test:
            self.fnames = os.listdir(path)
        else:
            tmp = os.listdir(path)
            np.random.shuffle(tmp)
            self.fnames = tmp[:57]

    def process(self, i):
        fname = self.fnames[i]
        print(f"Processing sample #{i}: {fname}")
        png = Image.open(f"{self.path}/{fname}")
        if self.resize:
            png = transforms.Resize((299, 299))(png)
        png.load()  # required for png.split()

        new_png = Image.new("RGB", png.size, (255, 255, 255))
        new_png.paste(png, mask=png.split()[3])  # 3 is the alpha channel
        label = fname.split('_')[0]
        mat_data = np.asarray(new_png)
        if self.binarize:
            mat_data[mat_data > 0] = 1

        return mat_data[:, :, 0], label, new_png

    def __len__(self):
        return len(self.fnames)


class MSCSV:
    def __init__(self, path, scaler, resize=False, test=False):
        self.path = path
        self.resize = resize
        self.scaler = scaler
        self.fnames = []
        if not test:
            self.fnames.extend(os.listdir(f"{path}"))
        else:
            tmp = os.listdir(f"{path}")
            np.random.shuffle(tmp)
            self.fnames = tmp[:57]

    def process(self, i):
        fname = self.fnames[i]
        if 'l' in fname.split('_')[1]:
            low = 1
        else:
            low = 0
        b_list = ["kox", "sau", "blk", "pae", "sep"]
        batch = 0 if fname.split('_')[0] in b_list and 'blk_p' not in fname else 1
        label = fname.split('_')[0]
        print(f"Processing sample #{i}: {fname}")
        mat_data = read_csv(f"{self.path}/{fname}")
        if self.scaler == 'binarize':
            mat_data[mat_data.values > 0] = 1
        elif 'efd' in self.scaler:
            from sklearn.preprocessing import KBinsDiscretizer
            n_bins = int(self.scaler.split('_')[1])
            mat_data = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform').fit_transform(mat_data)
            mat_data = MinMaxScaler().fit_transform(mat_data)
        elif 'ewd' in self.scaler:
            from sklearn.preprocessing import KBinsDiscretizer
            n_bins = int(self.scaler.split('_')[1])
            mat_data = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile').fit_transform(mat_data)
            mat_data = MinMaxScaler().fit_transform(mat_data)
        elif 'kmd' in self.scaler:
            from sklearn.preprocessing import KBinsDiscretizer
            n_bins = int(self.scaler.split('_')[1])
            mat_data = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans').fit_transform(mat_data)
            mat_data = MinMaxScaler().fit_transform(mat_data)

        elif 'cut' in self.scaler:
            n_bins = int(self.scaler.split('_')[1])
            mat_data = pd.DataFrame(
                np.stack([pd.cut(row, n_bins, labels=False, duplicates='drop', include_lowest=True) for row in
                          mat_data.values.T]).T
            )
            mat_data /= mat_data.max()
        elif 'discretizeq' in self.scaler:
            n_bins = int(self.scaler.split('_')[1])
            mat_data = pd.DataFrame(
                np.stack([pd.qcut(row, n_bins, labels=False, duplicates='drop') for row in mat_data.values.T]).T
            )

            mat_data = MinMaxScaler().fit_transform(mat_data)
        elif self.scaler == 'l2':
            mat_data = Normalizer().fit_transform(mat_data)
        elif self.scaler == 'l1':
            mat_data = Normalizer('l1').fit_transform(mat_data)
        elif self.scaler == 'minmax':
            mat_data = MinMaxScaler().fit_transform(mat_data)
        elif self.scaler == 'max':
            mat_data = Normalizer('max').fit_transform(mat_data)
        elif self.scaler == 'maxmax':
            mat_data /= mat_data.max().max()

        if self.resize:
            mat_data = transforms.Resize((299, 299))(torch.Tensor(mat_data.values).unsqueeze(0)).squeeze().detach().cpu().numpy()

        return mat_data.astype('float'), label, low, batch

    def __len__(self):
        return len(self.fnames)


class MSCSV2:
    def __init__(self, path, scaler, batches, resize=False, test=False):
        self.batches = batches
        self.path = path
        self.resize = resize
        self.scaler = scaler
        self.fnames = []
        if not test:
            self.fnames.extend(os.listdir(f"{path}"))
        else:
            tmp = os.listdir(f"{path}")
            np.random.shuffle(tmp)
            self.fnames = tmp[:57]
        tmp = []
        for fname in self.fnames:
            batch = [i for i, b in enumerate(self.batches) if b == fname.split('_')[0]]
            if len(batch) > 0:
                tmp += [fname]
        self.fnames = tmp
        self.fnames = np.array([x for x in self.fnames if 'pool' not in x])

    def process(self, i):
        fname = self.fnames[i]
        if 'l' in fname.split('_')[2]:
            low = 1
        else:
            low = 0

        batch = [i for i, b in enumerate(self.batches) if b == fname.split('_')[0]][0]
        label = fname.split('_')[1]
        print(f"Processing sample #{i}: {fname}")
        mat_data = read_csv(f"{self.path}/{fname}").values
        if self.scaler == 'binarize':
            mat_data[mat_data > 0] = 1
        elif 'efd' in self.scaler:
            from sklearn.preprocessing import KBinsDiscretizer
            n_bins = int(self.scaler.split('_')[1])
            mat_data = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform').fit_transform(mat_data)
            mat_data = MinMaxScaler().fit_transform(mat_data)
        elif 'ewd' in self.scaler:
            from sklearn.preprocessing import KBinsDiscretizer
            n_bins = int(self.scaler.split('_')[1])
            mat_data = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile').fit_transform(mat_data)
            mat_data = MinMaxScaler().fit_transform(mat_data)
        elif 'kmd' in self.scaler:
            from sklearn.preprocessing import KBinsDiscretizer
            n_bins = int(self.scaler.split('_')[1])
            mat_data = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans').fit_transform(mat_data)
            mat_data = MinMaxScaler().fit_transform(mat_data)

        elif 'cut' in self.scaler:
            n_bins = int(self.scaler.split('_')[1])
            mat_data = pd.DataFrame(
                np.stack([pd.cut(row, n_bins, labels=False, duplicates='drop', include_lowest=True) for row in
                          mat_data.values.T]).T
            )
            mat_data /= mat_data.max()
        elif 'discretizeq' in self.scaler:
            n_bins = int(self.scaler.split('_')[1])
            mat_data = pd.DataFrame(
                np.stack([pd.qcut(row, n_bins, labels=False, duplicates='drop') for row in mat_data.values.T]).T
            )

            mat_data = MinMaxScaler().fit_transform(mat_data)
        elif self.scaler == 'l2':
            mat_data = Normalizer().fit_transform(mat_data)
        elif self.scaler == 'l1':
            mat_data = Normalizer('l1').fit_transform(mat_data)
        elif self.scaler == 'minmax':
            mat_data = MinMaxScaler().fit_transform(mat_data)
        elif self.scaler == 'max':
            mat_data = Normalizer('max').fit_transform(mat_data)
        elif self.scaler == 'maxmax':
            mat_data /= mat_data.max().max()

        if self.resize:
            mat_data = transforms.Resize((299, 299))(torch.Tensor(mat_data).unsqueeze(0)).squeeze().detach().cpu().numpy()

        return mat_data.astype('float'), label, low, batch, fname

    def __len__(self):
        return len(self.fnames)


class MSCSV3D:
    def __init__(self, path, scaler, binarize=False, resize=True, test=False):
        self.path = path
        self.binarize = binarize
        self.resize = resize
        self.scaler = scaler
        if not test:
            self.fnames = os.listdir(path)
        else:
            tmp = os.listdir(path)
            np.random.shuffle(tmp)
            self.fnames = tmp[:57]

    def process(self, i):
        fname = self.fnames[i]
        if 'l' in fname.split('_')[1]:
            low = 1
        else:
            low = 0
        b_list = ["kox", "sau", "blk", "pae", "sep"]
        batch = fname.split('_')[0]
        label = fname.split('_')[1]
        print(f"Processing sample #{i}: {fname}")
        tensor = []
        for name in os.listdir(f"{self.path}/{fname}"):
            mat_data = read_csv(f"{self.path}/{fname}/{name}")
            if self.scaler == 'binarize':
                mat_data[mat_data.values > 0] = 1
            elif self.scaler == 'l2':
                mat_data = Normalizer().fit_transform(mat_data)
            elif 'discretize' in self.scaler:
                n_bins = self.scaler.split('_')
                mat_data = pd.cut(mat_data, n_bins)
            elif self.scaler == 'l1':
                mat_data = Normalizer('l1').fit_transform(mat_data)
            elif self.scaler == 'minmax':
                mat_data = MinMaxScaler().fit_transform(mat_data)
            elif self.scaler == 'max':
                mat_data = Normalizer('max').fit_transform(mat_data)
            elif self.scaler == 'maxmax':
                mat_data /= mat_data.max().max()
            tensor += [mat_data.astype('float32')]

        assert len(tensor) == 30
        try:
            tensor = np.stack(tensor)
        except:
            pass
        return tensor, label, low, batch

    def __len__(self):
        return len(self.fnames)


def resize_data_1d(data, new_size=(160,)):
    initial_size_x = data.shape[0]

    new_size_x = new_size[0]

    delta_x = initial_size_x / new_size_x

    new_data = np.zeros((new_size_x))

    for x, y, z in itertools.product(range(new_size_x)):
        new_data[x][y][z] = data[int(x * delta_x)]

    return new_data


def get_first_col_not_zeros(matrix):
    col = 0
    for col in range(matrix.shape[1]):
        if np.sum(matrix[:, col]) > 0:
            break
    return col - 1


def get_last_col_not_zeros(matrix):
    col = 0
    for col in reversed(range(matrix.shape[1])):
        if np.sum(matrix[:, col]) > 0:
            break
    return col + 1


def stretch_and_squeeze(x, ratio=0.1):
    x = torch.Tensor(x)
    pil = transforms.ToPILImage()(x)
    x_t = transforms.Resize(
        (x.shape[0], x.shape[1] + np.random.randint(-int(x.shape[1] * ratio), int(x.shape[1] * ratio)))
    )(pil)
    x_t = transforms.ToTensor()(x_t)[0]
    if x_t.shape[1] < x.shape[1]:
        dim_to_pad = x.shape[1] - x_t.shape[1]
        x_t = torch.concat((x_t, torch.zeros((x.shape[0], dim_to_pad))), 1)
    else:
        x_t = x_t[:, :x.shape[1]]

    return x_t.detach().cpu().numpy()


class MSDataset(Dataset):
    def __init__(self, data, training=False, labels=None, cats=None, lows=None, batches=None, transform=None, rec_transform=None, quantize=False,
                 remove_paddings=False, crop_size=-1, add_noise=False, random_recs=True, device='cuda'):
        self.random_recs = random_recs
        self.device = device
        self.training = training
        self.crop_size = crop_size
        self.samples = data
        self.add_noise = add_noise

        self.rec_transform = rec_transform
        self.transform = transform
        self.crop_size = crop_size
        self.labels = labels
        self.cats = cats
        self.lows = lows
        self.batches = batches
        self.quantize = quantize
        self.remove_paddings = remove_paddings
        labels_inds = {label: [i for i, x in enumerate(labels) if x == label] for label in labels}
        self.labels_data = {label: data[labels_inds[label]].copy() for label in labels}
        self.n_labels = {label: len(self.labels_data[label]) for label in labels}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        # x = x[80:, :1700]
        if self.crop_size != -1:
            max_start_crop = x.shape[1] - self.crop_size
            ran = np.random.randint(0, max_start_crop)
            x = torch.Tensor(x)[:, ran:ran + self.crop_size]  # .to(self.device)
        if self.labels is not None:
            label = self.labels[idx]
            batch = self.batches[idx]
            cat = self.cats[idx]
            if self.lows is not None:
                low = self.lows[idx]
            else:
                low = np.array([])
        else:
            label = None
            cat = None
            low = None
        if self.random_recs:
            # print(self.pool.shape, len(self.pool))
            to_rec = self.labels_data[label][np.random.randint(0, self.n_labels[label])].copy()
        else:
            to_rec = self.samples[idx].copy()
        if self.remove_paddings:
            x = x[:, get_first_col_not_zeros(x):]
            x = x[:, :get_last_col_not_zeros(x)]
        if self.transform:
            # rand = np.random.random()
            # if rand > 0.5:
            #     x[x > 0.9] = 1
            if x.shape[0] == 30:
                x = x.reshape(x.shape[1], x.shape[2], x.shape[0])
            # if self.training and np.random.randint(0, 2) == 1:
            #     x = stretch_and_squeeze(x)
            random.seed(42)
            x = self.transform(x)
            to_rec = self.rec_transform(to_rec)
        # if self.quantize:
        #     x = pd.qcut(x, 4, labels=False)
        if self.add_noise:
            if np.random.random() > 0.5:
                x = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
        return x, label, cat, low, batch, to_rec


class MSDataset2(Dataset):
    def __init__(self, data, training=False, labels=None, batches=None, concs=None, transform=None, quantize=False,
                 remove_paddings=False, crop_size=-1, add_noise=False, random_recs=False, device='cuda'):
        self.random_recs = random_recs
        device = device
        self.training = training
        self.crop_size = crop_size
        self.samples = data
        self.add_noise = add_noise

        self.concs = concs
        self.transform = transform
        self.crop_size = crop_size
        self.labels = labels
        self.batches = batches
        self.quantize = quantize
        self.remove_paddings = remove_paddings
        labels_inds = {label: [i for i, x in enumerate(labels) if x == label] for label in labels}
        self.labels_data = {label: data[labels_inds[label]].copy() for label in labels}
        self.n_labels = {label: len(self.labels_data[label]) for label in labels}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        concs = {}
        if self.labels is not None:
            label = self.labels[idx]
            batch = self.batches[idx]
            concs['lows'] = self.concs['lows'][idx]
            concs['highs'] = self.concs['highs'][idx]
            concs['vhighs'] = self.concs['vhighs'][idx]
            if concs['vhighs'] != -1 and concs['highs'] != -1 and concs['lows'] != -1:
                pass
        else:
            label = None
        if self.random_recs:
            # print(self.pool.shape, len(self.pool))
            to_rec = self.labels_data[label][np.random.randint(0, self.n_labels[label])].copy()
        else:
            to_rec = self.samples[idx].copy()
        x = self.samples[idx]
        if self.crop_size != -1:
            max_start_crop = x.shape[1] - self.crop_size
            ran = np.random.randint(0, max_start_crop)
            x = torch.Tensor(x)[:, ran:ran + self.crop_size]  # .to(device)
        if self.transform:
            x = self.transform(np.expand_dims(x, 0)).squeeze()
        # if self.quantize:
        #     x = pd.qcut(x, 4, labels=False)
        if self.add_noise:
            if np.random.random() > 0.5:
                x = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
        return x, label, batch, to_rec, concs


class MSDataset3(Dataset):
    def __init__(self, data, labels=None, batches=None, concs=None, transform=None, quantize=False,
                 remove_paddings=False, crop_size=-1, add_noise=False, random_recs=False, device='cuda'):
        self.random_recs = random_recs
        self.crop_size = crop_size
        self.samples = data
        self.add_noise = add_noise

        self.concs = concs
        self.transform = transform
        self.crop_size = crop_size
        self.labels = labels
        self.unique_labels = np.unique(labels)
        self.batches = batches
        self.quantize = quantize
        self.remove_paddings = remove_paddings
        labels_inds = {label: [i for i, x in enumerate(labels) if x == label] for label in labels}
        # try:
        self.labels_data = {label: data[labels_inds[label]] for label in labels}
        # except:
        #     print(labels)
        self.n_labels = {label: len(self.labels_data[label]) for label in labels}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        concs = {}
        if self.labels is not None:
            label = self.labels[idx]
            batch = self.batches[idx]
            for k in list(self.concs.keys()):
                concs[k] = self.concs[k][idx]
            # concs['highs'] = self.concs['highs'][idx]
            # concs['vhighs'] = self.concs['vhighs'][idx]
            # if concs['vhighs'] != -1 and concs['highs'] != -1 and concs['lows'] != -1:
            #     pass
        else:
            label = None
        if self.random_recs:
            # print(self.pool.shape, len(self.pool))
            to_rec = self.labels_data[label][np.random.randint(0, self.n_labels[label])].copy()
            not_label = None
            while not_label == label or not_label is None:
                not_label = self.unique_labels[np.random.randint(0, len(self.unique_labels))].copy()
            not_to_rec = self.labels_data[not_label][np.random.randint(0, self.n_labels[not_label])].copy()
        else:
            to_rec = self.samples[idx].copy()
            not_to_rec = torch.Tensor([0])
        x = self.samples[idx]
        if self.crop_size != -1:
            max_start_crop = x.shape[1] - self.crop_size
            ran = np.random.randint(0, max_start_crop)
            x = torch.Tensor(x)[:, ran:ran + self.crop_size]  # .to(device)
        if self.transform:
            x = self.transform(np.expand_dims(x, 0)).squeeze()
        # if self.quantize:
        #     x = pd.qcut(x, 4, labels=False)
        if self.add_noise:
            if np.random.random() > 0.5:
                x = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
        return x, label, batch, to_rec, not_to_rec, concs


class MSDataset3D(Dataset):
    def __init__(self, data, training, labels=None, cats=None, lows=None, batches=None, transform=None, binarize=False,
                 remove_paddings=False, crop_size=-1, device='cuda'):
        self.device = device
        self.training = training
        self.crop_size = crop_size
        self.samples = data

        self.transform = transform
        self.crop_size = crop_size
        self.labels = labels
        self.cats = cats
        self.lows = lows
        self.batches = batches
        self.binarize = binarize
        self.remove_paddings = remove_paddings

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx][np.random.randint(0, len(self.samples[idx]))]
        # x = x[80:, :1700]
        if self.crop_size != -1:
            max_start_crop = x.shape[1] - self.crop_size
            ran = np.random.randint(0, max_start_crop)
            x = torch.Tensor(x)[:, ran:ran + self.crop_size]  # .to(self.device)
        if self.labels is not None:
            label = self.labels[idx]
            batch = self.batches[idx]
            cat = self.cats[idx]
            if self.lows is not None:
                low = self.lows[idx]
            else:
                low = np.array([])
        else:
            label = None
            cat = None
            low = None
        if self.remove_paddings:
            x = x[:, get_first_col_not_zeros(x):]
            x = x[:, :get_last_col_not_zeros(x)]
        if self.transform:
            # rand = np.random.random()
            # if rand > 0.5:
            #     x[x > 0.9] = 1
            if x.shape[0] == 30:
                x = x.reshape(x.shape[1], x.shape[2], x.shape[0])
            # if self.training and np.random.randint(0, 2) == 1:
            #     x = stretch_and_squeeze(x)
            x = self.transform(x)
        if self.binarize:
            x[x > 0] = 1
        # if self.add_noise:
        #     x = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
        return x, label, cat, low, batch


# This function is much faster than using pd.read_csv
def load_data(path):
    cols = csv.DictReader(open(path))
    data = []
    names = []
    for i, row in enumerate(cols):
        names += [list(row.values())[0]]
        data += [np.array(list(row.values())[1:], dtype=float)]
    data = np.stack(data)
    # data = pd.DataFrame(data, index=labels, columns=list(row.keys())[1:])
    labels = np.array([d.split('_')[1].split('-')[0] for d in names])
    batches = np.array([d.split('_')[0] for d in names])
    # data = get_normalized(torch.Tensor(np.array(data.values, dtype='float')))
    data[np.isnan(data)] = 0
    columns = list(row.keys())[1:]
    return data.T, names, labels, batches, columns


def load_checkpoint(checkpoint_path,
                    model,
                    optimizer,
                    name,
                    ):
    losses = {
        "train": [],
        "valid": [],
    }
    if name not in os.listdir(checkpoint_path):
        print("checkpoint not found...")
        return model, None, 1, losses, None, None, {'valid_loss': -1}
    checkpoint_dict = torch.load(checkpoint_path + '/' + name, map_location='cpu')
    epoch = checkpoint_dict['epoch']
    best_values = checkpoint_dict['best_values']
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    state_dict = checkpoint_dict['model']
    model.load_state_dict(state_dict)
    try:
        losses_recon = checkpoint_dict['losses_recon']
        kl_divs = checkpoint_dict['kl_divs']
    except:
        losses_recon = None
        kl_divs = None

    losses = checkpoint_dict['losses']
    print("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, epoch))
    return model, optimizer, epoch, losses, kl_divs, losses_recon, best_values, checkpoint_dict['finished']


def final_save_checkpoint(checkpoint_path, model, optimizer, name):
    model, optimizer, epoch, losses, _, _, best_values, _ = load_checkpoint(checkpoint_path, model, optimizer, name)
    torch.save({'model': model.state_dict(),
                'losses': losses,
                'best_values': best_values,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                # 'learning_rate': learning_rate,
                'finished': True},
               checkpoint_path + '/' + name)


def save_checkpoint(model,
                    optimizer,
                    # learning_rate,
                    epoch,
                    checkpoint_path,
                    losses,
                    best_values,
                    name="cnn",
                    ):
    # model_for_saving = model_name(input_shape=input_shape, nb_classes=nb_classes, variant=variant,
    #                               activation=activation)
    # model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model.state_dict(),
                'losses': losses,
                'best_values': best_values,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                # 'learning_rate': learning_rate,
                'finished': False},
               checkpoint_path + '/' + name)


class PartialDataset(torch.utils.data.Dataset):
    def __init__(self, parent_ds, offset, length):
        self.parent_ds = parent_ds
        self.offset = offset
        self.length = length
        assert len(parent_ds) >= offset + length, Exception("Parent Dataset not long enough")
        super(PartialDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.parent_ds[i + self.offset]


class validation_spliter:
    def __init__(self, dataset, cv):
        self.cv = cv
        self.dataset = dataset
        self.current_cv = 0
        self.val_offset = int(np.floor(len(self.dataset) / self.cv))
        self.current_pos = 0

    def __next__(self):
        self.current_cv += 1
        # if self.current_cv == self.cv:
        #     val_offset = len(self.dataset) - self.current_pos
        # else:
        #     val_offset = self.val_offset
        partial_dataset = PartialDataset(self.dataset, 0, self.val_offset), \
                          PartialDataset(self.dataset, self.val_offset, len(self.dataset) - self.val_offset)

        # Move the samples currently used for the validation set at the end for the next split
        tmp = self.dataset.samples[:self.val_offset]
        self.dataset.samples = np.concatenate([self.dataset.samples[self.val_offset:], tmp], 0)

        return partial_dataset


def get_loaders(data, inputs, random_recs, ae=None, classifier=None, device='cuda'):
    """

    Args:
        data:
        ae:
        classifier:

    Returns:

    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(all_data.mean(), all_data.std()),
    ])

    # TODO Change the order of dicts: set (train, valid or test) first, the, inputs, labels etc
    train_set = MSDataset3(inputs['train'], data['cats']['train'],
                           [x for x in data['batches']['train']], data['subs']['train'],
                           transform=transform, crop_size=-1, random_recs=random_recs,
                           quantize=False, device=device)
    valid_set = MSDataset3(inputs['valid'], data['cats']['valid'],
                           [x for x in data['batches']['valid']], data['subs']['valid'],
                           transform=transform, crop_size=-1, random_recs=False,
                           quantize=False, device=device)
    valid_set2 = MSDataset3(inputs['valid'], data['cats']['valid'],
                            [x for x in data['batches']['valid']], data['subs']['valid'],
                            transform=transform, crop_size=-1, random_recs=random_recs,
                            quantize=False, device=device)
    test_set = MSDataset3(inputs['test'], data['cats']['test'],
                          [x for x in data['batches']['test']], data['subs']['test'],
                          transform=transform, crop_size=-1, random_recs=False,
                          quantize=False, device=device)
    test_set2 = MSDataset3(inputs['test'], data['cats']['test'],
                           [x for x in data['batches']['test']], data['subs']['test'],
                           transform=transform, crop_size=-1, random_recs=random_recs,
                           quantize=False, device=device)

    loaders = {
        'train': DataLoader(train_set,
                            num_workers=0,
                            shuffle=True,
                            batch_size=8,
                            pin_memory=False,
                            drop_last=True),

        'test': DataLoader(test_set,
                           num_workers=0,
                           shuffle=False,
                           batch_size=1,
                           pin_memory=False,
                           drop_last=False),
        'valid': DataLoader(valid_set,
                            num_workers=0,
                            shuffle=False,
                            batch_size=1,
                            pin_memory=False,
                            drop_last=False),
        'test2': DataLoader(test_set2,
                            num_workers=0,
                            shuffle=False,
                            batch_size=8,
                            pin_memory=False,
                            drop_last=True),
        'valid2': DataLoader(valid_set2,
                             num_workers=0,
                             shuffle=False,
                             batch_size=8,
                             pin_memory=False,
                             drop_last=True)
    }
    if ae is not None:
        valid_cats = []
        test_cats = []
        ae.eval()
        classifier.eval()
        for i, batch in enumerate(loaders['valid']):
            # optimizer_ae.zero_grad()
            input, labels, domain, to_rec, not_to_rec, concs = batch
            input[torch.isnan(input)] = 0
            input = input.to(device).float()
            enc, rec, _, kld = ae(input, None, 1, sampling=False)
            preds = ae.classifier(enc)
            domain_preds = ae.dann_discriminator(enc)
            valid_cats += [preds.detach().cpu().numpy().argmax(1)]
        for i, batch in enumerate(loaders['test']):
            # optimizer_ae.zero_grad()
            input, labels, domain, to_rec, not_to_rec, concs = batch
            input[torch.isnan(input)] = 0
            input = input.to(device).float()
            # to_rec = to_rec.to(device).float()
            enc, rec, _, kld = ae(input, None, 1, sampling=False)
            # if self.one_model:
            preds = ae.classifier(enc)
            domain_preds = ae.dann_discriminator(enc)
            # else:
            #     preds = classifier(enc)
            test_cats += [preds.detach().cpu().numpy().argmax(1)]

        valid_set2 = MSDataset3(inputs['valid'], np.concatenate(valid_cats),
                                [x for x in data['batches']['valid']], data['subs']['valid'],
                                transform=transform, crop_size=-1, random_recs=random_recs,
                                quantize=False, device=device)
        test_set2 = MSDataset3(inputs['test'], np.concatenate(test_cats),
                               [x for x in data['batches']['test']], data['subs']['test'],
                               transform=transform, crop_size=-1, random_recs=random_recs,
                               quantize=False, device=device)
        loaders['valid2'] = DataLoader(valid_set2,
                                       num_workers=0,
                                       shuffle=True,
                                       batch_size=8,
                                       pin_memory=False,
                                       drop_last=True)
        loaders['test2'] = DataLoader(test_set2,
                                      num_workers=0,
                                      shuffle=True,
                                      batch_size=8,
                                      pin_memory=False,
                                      drop_last=True)
        all_cats = np.concatenate(
            (data['cats']['train'], np.stack(valid_cats).reshape(-1), np.stack(test_cats).reshape(-1)))
        all_set = MSDataset3(inputs['all'], all_cats, [x for x in data['batches']['all']],
                             data['subs']['all'], transform=transform, crop_size=-1,
                             random_recs=random_recs, quantize=False, device=device)

    else:
        all_set = MSDataset3(inputs['all'], data['cats']['all'], [x for x in data['batches']['all']],
                             data['subs']['all'], transform=transform, crop_size=-1,
                             random_recs=random_recs, quantize=False, device=device)

    loaders['all'] = DataLoader(all_set,
                                num_workers=0,
                                shuffle=True,
                                batch_size=8,
                                pin_memory=False,
                                drop_last=True)

    return loaders


