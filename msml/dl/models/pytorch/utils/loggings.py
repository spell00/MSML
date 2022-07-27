import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# CUDA_VISIBLE_DEVICES = ""

from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
from msml.dl.utils.metrics import rKBET, rLISI
import numpy as np
import pandas as pd
import matplotlib

from msml.utils.utils import get_unique_labels

matplotlib.use('Agg')
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler
from matplotlib.lines import Line2D
from msml.dl.models.pytorch.utils.plotting import confidence_ellipse
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from msml.dl.utils.metrics import batch_f1_score
from msml.dl.utils.utils import log_confusion_matrix, save_roc_curve, save_precision_recall_curve
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from umap import UMAP
from sklearn.manifold import TSNE

# physical_devices = tf.config.list_physical_devices('CPU')
# tf.config.set_visible_devices(physical_devices)


# It is useless to run tensorflow on GPU and it takes a lot of GPU RAM for nothing
class TensorboardLoggingDann:
    def __init__(self, hparams_filepath, params):
        self.params = params
        self.hparams_filepath = hparams_filepath
        HPARAMS = [
            hp.HParam('dropout', hp.RealInterval(0.0, 1.0)),
            hp.HParam('gamma', hp.RealInterval(0., 100.0)),
            hp.HParam('beta', hp.RealInterval(0., 100.0)),
            hp.HParam('zeta', hp.RealInterval(0., 100.0)),
            hp.HParam('layer1', hp.IntInterval(0, 256)),
            hp.HParam('layer2', hp.IntInterval(0, 1024)),
            # hp.HParam('ncols', hp.IntInterval(0, 10000)),
            # hp.HParam('min_lr', hp.RealInterval(1e-15, 1e-1)),
            # hp.HParam('l1', hp.RealInterval(1e-15, 1e-1)),
            # hp.HParam('lr', hp.RealInterval(1e-8, 1e-2)),
            # hp.HParam('lr_dann', hp.RealInterval(1e-8, 1e-2)),
            # hp.HParam('lr_classif', hp.RealInterval(1e-8, 1e-2)),
            # hp.HParam('wd', hp.RealInterval(1e-15, 1e-1)),
            # hp.HParam('wd_dann', hp.RealInterval(1e-15, 1e-1)),
            # hp.HParam('wd_classif', hp.RealInterval(1e-15, 1e-1))
        ]

        with tf.summary.create_file_writer(hparams_filepath).as_default():
            hp.hparams_config(
                hparams=HPARAMS,
                metrics=[
                    hp.Metric('dom_loss', display_name='Domain Loss'),
                    hp.Metric('dom_acc', display_name='Domain Accuracy'),
                    hp.Metric('train_acc', display_name='Train Accuracy'),
                    hp.Metric('valid_acc', display_name='Valid Accuracy'),
                    hp.Metric('test_acc', display_name='Test Accuracy'),
                    hp.Metric('train_acc_l', display_name='Train Accuracy Lows'),
                    hp.Metric('valid_acc_l', display_name='Valid Accuracy Lows'),
                    hp.Metric('test_acc_l', display_name='Test Accuracy Lows'),
                    hp.Metric('train_acc_h', display_name='Train Accuracy Highs'),
                    hp.Metric('valid_acc_h', display_name='Valid Accuracy Highs'),
                    hp.Metric('test_acc_h', display_name='Test Accuracy Highs'),
                    hp.Metric('train_acc_v', display_name='Train Accuracy VHighs'),
                    hp.Metric('valid_acc_v', display_name='Valid Accuracy VHighs'),
                    hp.Metric('test_acc_v', display_name='Test Accuracy VHighs'),
                    hp.Metric('train_loss', display_name='Train Loss'),
                    hp.Metric('valid_loss', display_name='Valid Loss'),
                    hp.Metric('test_loss', display_name='Test Loss'),
                    hp.Metric('train_mcc', display_name='Train MCC'),
                    hp.Metric('valid_mcc', display_name='Valid MCC'),
                    hp.Metric('test_mcc', display_name='Test MCC'),
                    hp.Metric('train_mcc_h', display_name='Train MCC High'),
                    hp.Metric('valid_mcc_h', display_name='Valid MCC High'),
                    hp.Metric('test_mcc_h', display_name='Test MCC High'),
                    hp.Metric('train_mcc_v', display_name='Train MCC VHigh'),
                    hp.Metric('valid_mcc_v', display_name='Valid MCC VHigh'),
                    hp.Metric('test_mcc_v', display_name='Test MCC VHigh'),
                    hp.Metric('train_mcc_l', display_name='Train MCC Low'),
                    hp.Metric('valid_mcc_l', display_name='Valid MCC Low'),
                    hp.Metric('test_mcc_l', display_name='Test MCC Low')
                ],
            )

    def logging(self, traces):
        with tf.summary.create_file_writer(self.hparams_filepath).as_default():
            hp.hparams({
                'dropout': self.params['dropout'],
                # 'lr': self.params['lr'],
                # 'lr_dann': self.params['lr_dann'],
                # 'lr_classif': self.params['lr_classif'],
                # 'wd': self.params['wd'],
                # 'wd_dann': self.params['wd_dann'],
                # 'wd_classif': self.params['wd_classif'],
                'layer1': self.params['layer1'],
                'layer2': self.params['layer2'],
                'gamma': self.params['gamma'],
                'beta': self.params['beta'],
                'zeta': self.params['zeta'],
                # 'ncols': self.params['ncols'],
            })  # record the values used in this trial
            tf.summary.scalar('dom_loss', traces['dom_loss'], step=1)
            tf.summary.scalar('dom_acc', traces['dom_acc'], step=1)
            tf.summary.scalar('train_loss', traces['train_loss'], step=1)
            tf.summary.scalar('train_acc', traces['train_acc'], step=1)
            tf.summary.scalar('train_acc_h', traces['train_acc_h'], step=1)
            tf.summary.scalar('train_acc_v', traces['train_acc_v'], step=1)
            tf.summary.scalar('train_acc_l', traces['train_acc_l'], step=1)
            tf.summary.scalar('train_mcc', traces['train_mcc'], step=1)
            tf.summary.scalar('train_mcc_h', traces['train_mcc_h'], step=1)
            tf.summary.scalar('train_mcc_v', traces['train_mcc_v'], step=1)
            tf.summary.scalar('train_mcc_l', traces['train_mcc_l'], step=1)
            tf.summary.scalar('valid_loss', traces['valid_loss'], step=1)
            tf.summary.scalar('valid_acc', traces['valid_acc'], step=1)
            tf.summary.scalar('valid_acc_h', traces['valid_acc_h'], step=1)
            tf.summary.scalar('valid_acc_v', traces['valid_acc_v'], step=1)
            tf.summary.scalar('valid_acc_l', traces['valid_acc_l'], step=1)
            tf.summary.scalar('valid_mcc', traces['valid_mcc'], step=1)
            tf.summary.scalar('valid_mcc_h', traces['valid_mcc_h'], step=1)
            tf.summary.scalar('valid_mcc_v', traces['valid_mcc_v'], step=1)
            tf.summary.scalar('valid_mcc_l', traces['valid_mcc_l'], step=1)
            tf.summary.scalar('test_loss', traces['test_loss'], step=1)
            tf.summary.scalar('test_acc', traces['test_acc'], step=1)
            tf.summary.scalar('test_acc_h', traces['test_acc_h'], step=1)
            tf.summary.scalar('test_acc_v', traces['test_acc_v'], step=1)
            tf.summary.scalar('test_acc_l', traces['test_acc_l'], step=1)
            tf.summary.scalar('test_mcc', traces['test_mcc'], step=1)
            tf.summary.scalar('test_mcc_h', traces['test_mcc_h'], step=1)
            tf.summary.scalar('test_mcc_v', traces['test_mcc_v'], step=1)
            tf.summary.scalar('test_mcc_l', traces['test_mcc_l'], step=1)


class TensorboardLoggingAEDANN:
    def __init__(self, hparams_filepath, params):
        self.params = params
        self.hparams_filepath = hparams_filepath
        HPARAMS = [
            hp.HParam('dropout', hp.RealInterval(0.0, 1.0)),
            hp.HParam('smoothing', hp.RealInterval(0.0, 1.0)),
            hp.HParam('gamma', hp.RealInterval(0., 100.0)),
            hp.HParam('beta', hp.RealInterval(0., 100.0)),
            hp.HParam('zeta', hp.RealInterval(0., 100.0)),
            hp.HParam('layer1', hp.IntInterval(0, 256)),
            hp.HParam('layer2', hp.IntInterval(0, 1024)),
            hp.HParam('ncols', hp.IntInterval(0, 10000)),
            # hp.HParam('min_lr', hp.RealInterval(1e-15, 1e-1)),
            # hp.HParam('l1', hp.RealInterval(1e-15, 1e-1)),
            hp.HParam('lr', hp.RealInterval(1e-8, 1e-2)),
            hp.HParam('lr_dann', hp.RealInterval(1e-8, 1e-2)),
            hp.HParam('lr_classif', hp.RealInterval(1e-8, 1e-2)),
            hp.HParam('wd', hp.RealInterval(1e-15, 1e-1)),
            hp.HParam('wd_dann', hp.RealInterval(1e-15, 1e-1)),
            hp.HParam('wd_classif', hp.RealInterval(1e-15, 1e-1)),
            hp.HParam('scale', hp.Discrete(['binarize', 'minmax', 'standard', 'robust'])),
            hp.HParam('dann_sets', hp.Discrete([0, 1])),
            hp.HParam('dann_plates', hp.Discrete([0, 1])),
            hp.HParam('zinb', hp.Discrete([0, 1])),
            hp.HParam('variational', hp.Discrete([0, 1])),

        ]

        with tf.summary.create_file_writer(hparams_filepath).as_default():

            hp.hparams_config(
                hparams=HPARAMS,
                metrics=[
                    hp.Metric('train_acc', display_name='Train Accuracy'),
                    hp.Metric('valid_acc', display_name='Valid Accuracy'),
                    hp.Metric('test_acc', display_name='Test Accuracy'),
                    hp.Metric('train_acc_l', display_name='Train Accuracy Lows'),
                    hp.Metric('valid_acc_l', display_name='Valid Accuracy Lows'),
                    hp.Metric('test_acc_l', display_name='Test Accuracy Lows'),
                    hp.Metric('train_acc_h', display_name='Train Accuracy Highs'),
                    hp.Metric('valid_acc_h', display_name='Valid Accuracy Highs'),
                    hp.Metric('test_acc_h', display_name='Test Accuracy Highs'),
                    hp.Metric('train_acc_v', display_name='Train Accuracy VHighs'),
                    hp.Metric('valid_acc_v', display_name='Valid Accuracy VHighs'),
                    hp.Metric('test_acc_v', display_name='Test Accuracy VHighs'),
                    hp.Metric('train_loss', display_name='Train Loss'),
                    hp.Metric('valid_loss', display_name='Valid Loss'),
                    hp.Metric('test_loss', display_name='Test Loss'),
                    hp.Metric('train_mcc', display_name='Train MCC'),
                    hp.Metric('valid_mcc', display_name='Valid MCC'),
                    hp.Metric('test_mcc', display_name='Test MCC'),
                    hp.Metric('train_mcc_h', display_name='Train MCC High'),
                    hp.Metric('valid_mcc_h', display_name='Valid MCC High'),
                    hp.Metric('test_mcc_h', display_name='Test MCC High'),
                    hp.Metric('train_mcc_v', display_name='Train MCC VHigh'),
                    hp.Metric('valid_mcc_v', display_name='Valid MCC VHigh'),
                    hp.Metric('test_mcc_v', display_name='Test MCC VHigh'),
                    hp.Metric('train_mcc_l', display_name='Train MCC Low'),
                    hp.Metric('valid_mcc_l', display_name='Valid MCC Low'),
                    hp.Metric('test_mcc_l', display_name='Test MCC Low')
                ],
            )

    def logging(self, traces):
        with tf.summary.create_file_writer(self.hparams_filepath).as_default():
            hp.hparams({
                'gamma': self.params['gamma'],
                'beta': self.params['beta'],
                'zeta': self.params['zeta'],
                'dropout': self.params['dropout'],
                'smoothing': self.params['smoothing'],
                'lr': self.params['lr'],
                'lr_dann': self.params['lr_dann'],
                'lr_classif': self.params['lr_classif'],
                'wd': self.params['wd'],
                'wd_dann': self.params['wd_dann'],
                'wd_classif': self.params['wd_classif'],
                'layer1': self.params['layer1'],
                'layer2': self.params['layer2'],
                'scale': self.params['scaler'],
                'ncols': self.params['ncols'],
            })  # record the values used in this trial
            tf.summary.scalar('train_loss', traces['train_loss'], step=1)
            tf.summary.scalar('train_acc', traces['train_acc'], step=1)
            tf.summary.scalar('train_acc_h', traces['train_acc_h'], step=1)
            tf.summary.scalar('train_acc_v', traces['train_acc_v'], step=1)
            tf.summary.scalar('train_acc_l', traces['train_acc_l'], step=1)
            tf.summary.scalar('train_mcc', traces['train_mcc'], step=1)
            tf.summary.scalar('train_mcc_h', traces['train_mcc_h'], step=1)
            tf.summary.scalar('train_mcc_v', traces['train_mcc_v'], step=1)
            tf.summary.scalar('train_mcc_l', traces['train_mcc_l'], step=1)
            tf.summary.scalar('valid_loss', traces['valid_loss'], step=1)
            tf.summary.scalar('valid_acc', traces['valid_acc'], step=1)
            tf.summary.scalar('valid_acc_h', traces['valid_acc_h'], step=1)
            tf.summary.scalar('valid_acc_v', traces['valid_acc_v'], step=1)
            tf.summary.scalar('valid_acc_l', traces['valid_acc_l'], step=1)
            tf.summary.scalar('valid_mcc', traces['valid_mcc'], step=1)
            tf.summary.scalar('valid_mcc_h', traces['valid_mcc_h'], step=1)
            tf.summary.scalar('valid_mcc_v', traces['valid_mcc_v'], step=1)
            tf.summary.scalar('valid_mcc_l', traces['valid_mcc_l'], step=1)
            tf.summary.scalar('test_loss', traces['test_loss'], step=1)
            tf.summary.scalar('test/acc', traces['test_acc'], step=1)
            tf.summary.scalar('test/acc/h', traces['test_acc_h'], step=1)
            tf.summary.scalar('test/acc/v', traces['test/acc/v'], step=1)
            tf.summary.scalar('test/acc/l', traces['test/acc/l'], step=1)
            tf.summary.scalar('test_mcc', traces['test_mcc'], step=1)
            tf.summary.scalar('test_mcc_h', traces['test_mcc_h'], step=1)
            tf.summary.scalar('test_mcc_v', traces['test_mcc_v'], step=1)
            tf.summary.scalar('test_mcc_l', traces['test_mcc_l'], step=1)


class TensorboardLoggingAE:
    def __init__(self, hparams_filepath, params, variational, zinb, dann_sets, dann_plates, tw, tl, pseudo,
                 train_after_warmup, berm, subs):
        self.subs = subs  # subsets
        self.params = params
        self.train_after_warmup = train_after_warmup
        self.tw = tw
        self.berm = berm
        self.tl = tl
        self.pseudo = pseudo
        self.variational = variational
        self.zinb = zinb
        self.dann_sets = dann_sets
        self.dann_plates = dann_plates
        self.hparams_filepath = hparams_filepath
        HPARAMS = [
            hp.HParam('thres', hp.RealInterval(0.0, 1.0)),
            hp.HParam('dropout', hp.RealInterval(0.0, 1.0)),
            hp.HParam('smoothing', hp.RealInterval(0.0, 1.0)),
            hp.HParam('margin', hp.RealInterval(0.0, 10.0)),
            hp.HParam('gamma', hp.RealInterval(0., 100.0)),
            hp.HParam('beta', hp.RealInterval(0., 100.0)),
            hp.HParam('zeta', hp.RealInterval(0., 100.0)),
            hp.HParam('nu', hp.RealInterval(0., 100.0)),
            hp.HParam('layer1', hp.IntInterval(0, 256)),
            hp.HParam('layer2', hp.IntInterval(0, 1024)),
            hp.HParam('ncols', hp.IntInterval(0, 10000)),
            hp.HParam('lr', hp.RealInterval(1e-8, 1e-2)),
            hp.HParam('wd', hp.RealInterval(1e-15, 1e-1)),
            hp.HParam('scale', hp.Discrete(['binarize', 'minmax', 'standard', 'robust'])),
            hp.HParam('dann_sets', hp.Discrete([0, 1])),
            hp.HParam('dann_plates', hp.Discrete([0, 1])),
            hp.HParam('zinb', hp.Discrete([0, 1])),
            hp.HParam('variational', hp.Discrete([0, 1])),
            hp.HParam('tied_w', hp.Discrete([0, 1])),
            hp.HParam('tripletloss', hp.Discrete([0, 1])),
            hp.HParam('train_after_warmup', hp.Discrete([0, 1])),
            hp.HParam('train_after_warmup', hp.Discrete([0, 1])),

        ]

        metrics = [
            hp.Metric('rec_loss', display_name='Rec Loss'),
            hp.Metric('dom_loss', display_name='Domain Loss'),
            hp.Metric('dom_acc', display_name='Domain Accuracy'),
            hp.Metric('train/loss', display_name='Train Loss'),
            hp.Metric('valid/loss', display_name='Valid Loss'),
            hp.Metric('test/loss', display_name='Test Loss'),
            hp.Metric('train/acc', display_name='Train Accuracy'),
            hp.Metric('valid/acc', display_name='Valid Accuracy'),
            hp.Metric('test/acc', display_name='Test Accuracy'),
            hp.Metric('train/mcc', display_name='Train MCC'),
            hp.Metric('valid/mcc', display_name='Valid MCC'),
            hp.Metric('test/mcc', display_name='Test MCC'),
        ]
        for k in list(subs):
            new_metrics = [
                hp.Metric(f'train/loss/{k}', display_name=f'Train Loss {k}'),
                hp.Metric(f'valid/loss/{k}', display_name=f'Valid Loss {k}'),
                hp.Metric(f'test/loss/{k}', display_name=f'Test Loss {k}'),
                hp.Metric(f'train/acc/{k}', display_name=f'Train Accuracy {k}'),
                hp.Metric(f'valid/acc/{k}', display_name=f'Valid Accuracy {k}'),
                hp.Metric(f'test/acc/{k}', display_name=f'Test Accuracy {k}'),
                hp.Metric(f'train/mcc/{k}', display_name=f'Train MCC {k}'),
                hp.Metric(f'valid/mcc/{k}', display_name=f'Valid MCC {k}'),
                hp.Metric(f'test/mcc/{k}', display_name=f'Test MCC {k}'),
            ]
            metrics.extend(new_metrics)

        with tf.summary.create_file_writer(hparams_filepath).as_default():
            hp.hparams_config(
                hparams=HPARAMS,
                metrics=metrics,
            )

    def logging(self, traces):
        with tf.summary.create_file_writer(self.hparams_filepath).as_default():
            hp.hparams({
                'thres': self.params['thres'],
                'gamma': self.params['gamma'],
                'beta': self.params['beta'],
                'zeta': self.params['zeta'],
                'nu': self.params['nu'],
                'dropout': self.params['dropout'],
                'smoothing': self.params['smoothing'],
                'lr': self.params['lr'],
                'margin': self.params['margin'],
                # 'lr_classif': self.params['lr_classif'],
                'wd': self.params['wd'],
                # 'wd_classif': self.params['wd_classif'],
                'layer1': self.params['layer1'],
                'layer2': self.params['layer2'],
                'scale': self.params['scaler'],
                'ncols': self.params['ncols'],
                'tied_w': self.tw,
                'tripletloss': self.tl,
                'variational': self.variational,
                'zinb': self.zinb,
                'dann_sets': self.dann_sets,
                'dann_plates': self.dann_plates,
                'pseudo': self.pseudo,
                'train_after_warmup': self.train_after_warmup
            })  # record the values used in this trial
            tf.summary.scalar('rec_loss', traces['rec_loss'], step=1)
            tf.summary.scalar('dom_loss', traces['dom_loss'], step=1)
            try:
                tf.summary.scalar('dom_acc', traces['dom_acc'], step=1)
            except:
                tf.summary.scalar('dom_acc', traces['dom_acc'][0], step=1)

            for g in ['train', 'valid', 'test']:
                tf.summary.scalar(f'{g}/loss', traces[f'{g}_loss'], step=1)
                tf.summary.scalar(f'{g}/acc', traces[f'{g}_acc'], step=1)
                tf.summary.scalar(f'{g}/mcc', traces[f'{g}_mcc'], step=1)
                for k in self.subs:
                    tf.summary.scalar(f'{g}/acc/{k}', traces[f'{g}_acc_{k}'], step=1)
                    tf.summary.scalar(f'{g}/mcc/{k}', traces[f'{g}_mcc_{k}'], step=1)


class TensorboardLoggingVAE:
    def __init__(self, hparams_filepath, params):
        self.params = params
        self.hparams_filepath = hparams_filepath
        HPARAMS = [
            hp.HParam('dropout', hp.RealInterval(0.0, 1.0)),
            hp.HParam('smoothing', hp.RealInterval(0.0, 1.0)),
            hp.HParam('gamma', hp.RealInterval(0., 100.0)),
            hp.HParam('beta', hp.RealInterval(0., 100.0)),
            hp.HParam('zeta', hp.RealInterval(0., 100.0)),
            hp.HParam('layer1', hp.IntInterval(0, 256)),
            hp.HParam('layer2', hp.IntInterval(0, 1024)),
            hp.HParam('ncols', hp.IntInterval(0, 10000)),
            # hp.HParam('min_lr', hp.RealInterval(1e-15, 1e-1)),
            # hp.HParam('l1', hp.RealInterval(1e-15, 1e-1)),
            hp.HParam('lr', hp.RealInterval(1e-8, 1e-2)),
            # hp.HParam('lr_dann', hp.RealInterval(1e-8, 1e-2)),
            # hp.HParam('lr_classif', hp.RealInterval(1e-8, 1e-2)),
            hp.HParam('wd', hp.RealInterval(1e-15, 1e-1)),
            # hp.HParam('wd_dann', hp.RealInterval(1e-15, 1e-1)),
            # hp.HParam('wd_classif', hp.RealInterval(1e-15, 1e-1)),
            hp.HParam('scale', hp.Discrete(['binarize', 'minmax', 'standard', 'robust']))

        ]

        with tf.summary.create_file_writer(hparams_filepath).as_default():
            hp.hparams_config(
                hparams=HPARAMS,
                metrics=[
                    hp.Metric('train/acc', display_name='Train Accuracy'),
                    hp.Metric('valid/acc', display_name='Valid Accuracy'),
                    hp.Metric('test/acc', display_name='Test Accuracy'),
                    hp.Metric('train/acc/l', display_name='Train Accuracy Lows'),
                    hp.Metric('valid/acc/l', display_name='Valid Accuracy Lows'),
                    hp.Metric('test/acc/l', display_name='Test Accuracy Lows'),
                    hp.Metric('train/acc/h', display_name='Train Accuracy Highs'),
                    hp.Metric('valid/acc/h', display_name='Valid Accuracy Highs'),
                    hp.Metric('test/acc/h', display_name='Test Accuracy Highs'),
                    hp.Metric('train/acc/v', display_name='Train Accuracy VHighs'),
                    hp.Metric('valid/acc/v', display_name='Valid Accuracy VHighs'),
                    hp.Metric('test/acc/v', display_name='Test Accuracy VHighs'),
                    hp.Metric('train/loss', display_name='Train Loss'),
                    hp.Metric('valid/loss', display_name='Valid Loss'),
                    hp.Metric('test/loss', display_name='Test Loss'),
                    hp.Metric('train/mcc', display_name='Train MCC'),
                    hp.Metric('valid/mcc', display_name='Valid MCC'),
                    hp.Metric('test/mcc', display_name='Test MCC'),
                    hp.Metric('train/mcc/h', display_name='Train MCC High'),
                    hp.Metric('valid/mcc/h', display_name='Valid MCC High'),
                    hp.Metric('test/mcc/h', display_name='Test MCC High'),
                    hp.Metric('train/mcc/v', display_name='Train MCC VHigh'),
                    hp.Metric('valid/mcc/v', display_name='Valid MCC VHigh'),
                    hp.Metric('test/mcc/v', display_name='Test MCC VHigh'),
                    hp.Metric('train/mcc/l', display_name='Train MCC Low'),
                    hp.Metric('valid/mcc/l', display_name='Valid MCC Low'),
                    hp.Metric('test/mcc/l', display_name='Test MCC Low')
                ],
            )

    def logging(self, traces):
        with tf.summary.create_file_writer(self.hparams_filepath).as_default():
            hp.hparams({
                'gamma': self.params['gamma'],
                'beta': self.params['beta'],
                'zeta': self.params['zeta'],
                'dropout': self.params['dropout'],
                'smoothing': self.params['smoothing'],
                'lr': self.params['lr'],
                'lr_dann': self.params['lr_dann'],
                'lr_classif': self.params['lr_classif'],
                'wd': self.params['wd'],
                'wd_dann': self.params['wd_dann'],
                'wd_classif': self.params['wd_classif'],
                'layer1': self.params['layer1'],
                'layer2': self.params['layer2'],
                'scaler': self.params['scaler'],
                'ncols': self.params['ncols'],
            })  # record the values used in this trial
            tf.summary.scalar('train_loss', traces['train_loss'], step=1)
            tf.summary.scalar('train_acc', traces['train_acc'], step=1)
            tf.summary.scalar('train_acc_h', traces['train_acc_h'], step=1)
            tf.summary.scalar('train_acc_v', traces['train_acc_v'], step=1)
            tf.summary.scalar('train_acc_l', traces['train_acc_l'], step=1)
            tf.summary.scalar('train_mcc', traces['train_mcc'], step=1)
            tf.summary.scalar('train_mcc_h', traces['train_mcc_h'], step=1)
            tf.summary.scalar('train_mcc_v', traces['train_mcc_v'], step=1)
            tf.summary.scalar('train_mcc_l', traces['train_mcc_l'], step=1)
            tf.summary.scalar('valid_loss', traces['valid_loss'], step=1)
            tf.summary.scalar('valid_acc', traces['valid_acc'], step=1)
            tf.summary.scalar('valid_acc_h', traces['valid_acc_h'], step=1)
            tf.summary.scalar('valid_acc_v', traces['valid_acc_v'], step=1)
            tf.summary.scalar('valid_acc_l', traces['valid_acc_l'], step=1)
            tf.summary.scalar('valid_mcc', traces['valid_mcc'], step=1)
            tf.summary.scalar('valid_mcc_h', traces['valid_mcc_h'], step=1)
            tf.summary.scalar('valid_mcc_v', traces['valid_mcc_v'], step=1)
            tf.summary.scalar('valid_mcc_l', traces['valid_mcc_l'], step=1)
            tf.summary.scalar('test_loss', traces['test_loss'], step=1)
            tf.summary.scalar('test_acc', traces['test_acc'], step=1)
            tf.summary.scalar('test_acc_h', traces['test_acc_h'], step=1)
            tf.summary.scalar('test_acc_v', traces['test_acc_v'], step=1)
            tf.summary.scalar('test_acc_l', traces['test_acc_l'], step=1)
            tf.summary.scalar('test_mcc', traces['test_mcc'], step=1)
            tf.summary.scalar('test_mcc_h', traces['test_mcc_h'], step=1)
            tf.summary.scalar('test_mcc_v', traces['test_mcc_v'], step=1)
            tf.summary.scalar('test_mcc_l', traces['test_mcc_l'], step=1)


class TensorboardLogging:
    def __init__(self, hparams_filepath, params):
        self.params = params
        self.hparams_filepath = hparams_filepath
        HPARAMS = [
            hp.HParam('dropout', hp.RealInterval(0.0, 1.0)),
            hp.HParam('smoothing', hp.RealInterval(0.0, 1.0)),
            hp.HParam('gamma', hp.RealInterval(0., 100.0)),
            hp.HParam('beta', hp.RealInterval(0., 100.0)),
            hp.HParam('zeta', hp.RealInterval(0., 100.0)),
            hp.HParam('hidden', hp.IntInterval(0, 256)),
            hp.HParam('ncols', hp.IntInterval(0, 10000)),
            # hp.HParam('min_lr', hp.RealInterval(1e-15, 1e-1)),
            # hp.HParam('l1', hp.RealInterval(1e-15, 1e-1)),
            hp.HParam('lr', hp.RealInterval(1e-8, 1e-2)),
            hp.HParam('lr_dann', hp.RealInterval(1e-8, 1e-2)),
            # hp.HParam('lr_classif', hp.RealInterval(1e-8, 1e-2)),
            hp.HParam('wd', hp.RealInterval(1e-15, 1e-1)),
            hp.HParam('wd_dann', hp.RealInterval(1e-15, 1e-1)),
            # hp.HParam('wd_classif', hp.RealInterval(1e-15, 1e-1)),
            hp.HParam('scale', hp.Discrete(['binarize', 'minmax', 'standard', 'robust']))

        ]

        with tf.summary.create_file_writer(hparams_filepath).as_default():
            hp.hparams_config(
                hparams=HPARAMS,
                metrics=[
                    hp.Metric('train_acc', display_name='Train Accuracy'),
                    hp.Metric('valid_acc', display_name='Valid Accuracy'),
                    hp.Metric('test_acc', display_name='Test Accuracy'),
                    hp.Metric('train_acc_l', display_name='Train Accuracy Lows'),
                    hp.Metric('valid_acc_l', display_name='Valid Accuracy Lows'),
                    hp.Metric('test_acc_l', display_name='Test Accuracy Lows'),
                    hp.Metric('train_acc_h', display_name='Train Accuracy Highs'),
                    hp.Metric('valid_acc_h', display_name='Valid Accuracy Highs'),
                    hp.Metric('test_acc_h', display_name='Test Accuracy Highs'),
                    hp.Metric('train_acc_v', display_name='Train Accuracy VHighs'),
                    hp.Metric('valid_acc_v', display_name='Valid Accuracy VHighs'),
                    hp.Metric('test_acc_v', display_name='Test Accuracy VHighs'),
                    hp.Metric('train_loss', display_name='Train Loss'),
                    hp.Metric('valid_loss', display_name='Valid Loss'),
                    hp.Metric('test_loss', display_name='Test Loss'),
                    hp.Metric('train_mcc', display_name='Train MCC'),
                    hp.Metric('valid_mcc', display_name='Valid MCC'),
                    hp.Metric('test_mcc', display_name='Test MCC'),
                    hp.Metric('train_mcc_h', display_name='Train MCC High'),
                    hp.Metric('valid_mcc_h', display_name='Valid MCC High'),
                    hp.Metric('test_mcc_h', display_name='Test MCC High'),
                    hp.Metric('train_mcc_v', display_name='Train MCC VHigh'),
                    hp.Metric('valid_mcc_v', display_name='Valid MCC VHigh'),
                    hp.Metric('test_mcc_v', display_name='Test MCC VHigh'),
                    hp.Metric('train_mcc_l', display_name='Train MCC Low'),
                    hp.Metric('valid_mcc_l', display_name='Valid MCC Low'),
                    hp.Metric('test_mcc_l', display_name='Test MCC Low')
                ],
            )

    def logging(self, traces):
        with tf.summary.create_file_writer(self.hparams_filepath).as_default():
            hp.hparams({
                'gamma': self.params['gamma'],
                'beta': self.params['beta'],
                'zeta': self.params['zeta'],
                'dropout': self.params['dropout'],
                'smoothing': self.params['smoothing'],
                'lr': self.params['lr'],
                'lr_dann': self.params['lr_dann'],
                # 'lr_classif': self.params['lr_classif'],
                'wd': self.params['wd'],
                'wd_dann': self.params['wd_dann'],
                # 'wd_classif': self.params['wd_classif'],
                # 'layer1': self.params['layer1'],
                'hidden': self.params['hidden'],
                'scaler': self.params['scaler'],
                'ncols': self.params['ncols'],
            })  # record the values used in this trial
            tf.summary.scalar('train_loss', traces['train_loss'], step=1)
            tf.summary.scalar('train_acc', traces['train_acc'], step=1)
            tf.summary.scalar('train_acc_h', traces['train_acc_h'], step=1)
            tf.summary.scalar('train_acc_v', traces['train_acc_v'], step=1)
            tf.summary.scalar('train_acc_l', traces['train_acc_l'], step=1)
            tf.summary.scalar('train_mcc', traces['train_mcc'], step=1)
            tf.summary.scalar('train_mcc_h', traces['train_mcc_h'], step=1)
            tf.summary.scalar('train_mcc_v', traces['train_mcc_v'], step=1)
            tf.summary.scalar('train_mcc_l', traces['train_mcc_l'], step=1)
            tf.summary.scalar('valid_loss', traces['valid_loss'], step=1)
            tf.summary.scalar('valid_acc', traces['valid_acc'], step=1)
            tf.summary.scalar('valid_acc_h', traces['valid_acc_h'], step=1)
            tf.summary.scalar('valid_acc_v', traces['valid_acc_v'], step=1)
            tf.summary.scalar('valid_acc_l', traces['valid_acc_l'], step=1)
            tf.summary.scalar('valid_mcc', traces['valid_mcc'], step=1)
            tf.summary.scalar('valid_mcc_h', traces['valid_mcc_h'], step=1)
            tf.summary.scalar('valid_mcc_v', traces['valid_mcc_v'], step=1)
            tf.summary.scalar('valid_mcc_l', traces['valid_mcc_l'], step=1)
            tf.summary.scalar('test_loss', traces['test_loss'], step=1)
            tf.summary.scalar('test_acc', traces['test_acc'], step=1)
            tf.summary.scalar('test_acc_h', traces['test_acc_h'], step=1)
            tf.summary.scalar('test_acc_v', traces['test_acc_v'], step=1)
            tf.summary.scalar('test_acc_l', traces['test_acc_l'], step=1)
            tf.summary.scalar('test_mcc', traces['test_mcc'], step=1)
            tf.summary.scalar('test_mcc_h', traces['test_mcc_h'], step=1)
            tf.summary.scalar('test_mcc_v', traces['test_mcc_v'], step=1)
            tf.summary.scalar('test_mcc_l', traces['test_mcc_l'], step=1)


def log_metrics(lists, values):
    # sets are grouped togheter for a single metric
    knns = {info: {repres: KNeighborsClassifier(n_neighbors=10) for repres in ['domains', 'labels']} for info in ['enc', 'rec', 'inputs']}
    classes = []
    sets = []
    encoded_values = []
    rec_values = []
    inputs = []
    for group in ['train', 'valid', 'test']:
        tmp = np.concatenate(lists[group]['set'])
        for s in range(len(tmp)):
            if tmp[s] == 'train':
                tmp[s] = 0
            elif tmp[s] == 'valid':
                tmp[s] = 1
            elif tmp[s] == 'test':
                tmp[s] = 2
            else:
                exit()
        sets += [np.array(tmp, np.int)]
        classes += [np.array(np.concatenate(lists[group]['classes']), np.int)]
        encoded_values += [np.concatenate(lists[group]['encoded_values'])]
        rec_values += [np.concatenate(lists[group]['rec_values'])]
        inputs += [np.concatenate(lists[group]['inputs'])]
        for metric, funct in zip(['lisi', 'silhouette', 'kbet'], [rLISI, silhouette_score, rKBET]):
            try:
                values[group][metric]['enc']['labels'] += [funct(np.concatenate(lists[group]['encoded_values']),
                                                                 np.concatenate(lists[group]['classes']))]
            except:
                values[group][metric]['enc']['labels'] += [-1]
            try:
                values[group][metric]['enc']['domains'] += [funct(np.concatenate(lists[group]['encoded_values']),
                                                                  np.concatenate(lists[group]['domains']))]
            except:
                values[group][metric]['enc']['domains'] += [-1]

            try:
                values[group][metric]['rec']['labels'] += [funct(np.concatenate(lists[group]['rec_values']),
                                                                 np.concatenate(lists[group]['classes']))]
            except:
                values[group][metric]['rec']['domains'] += [-1]

            try:
                values[group][metric]['rec']['domains'] += [funct(np.concatenate(lists[group]['rec_values']),
                                                                  np.concatenate(lists[group]['domains']))]
            except:
                values[group][metric]['rec']['domains'] += [-1]

            try:
                values[group][metric]['inputs']['labels'] += [funct(np.concatenate(lists[group]['inputs']),
                                                                 np.concatenate(lists[group]['classes']))]
            except:
                values[group][metric]['inputs']['labels'] += [-1]

            try:
                values[group][metric]['inputs']['domains'] += [funct(np.concatenate(lists[group]['inputs']),
                                                                  np.concatenate(lists[group]['domains']))]
            except:
                values[group][metric]['inputs']['domains'] += [-1]

        for metric, funct in zip(
                ['adjusted_rand_score', 'adjusted_mutual_info_score'],
                [adjusted_rand_score, adjusted_mutual_info_score]):
            # TODO Should not get the dimain classifier for adversarial learning for this metric. Use K-nearest neighbors
            if group == 'train':
                knns['enc']['domains'].fit(
                    np.concatenate(lists[group]['encoded_values']),
                    np.concatenate(lists[group]['domains'])
                )
                knns['rec']['domains'].fit(
                    np.concatenate(lists[group]['rec_values']),
                    np.concatenate(lists[group]['domains'])
                )
                knns['inputs']['domains'].fit(
                    np.concatenate(lists[group]['inputs']),
                    np.concatenate(lists[group]['domains'])
                )

                knns['enc']['labels'].fit(
                    np.concatenate(lists[group]['encoded_values']),
                    np.concatenate(lists[group]['classes'])
                )
                knns['rec']['labels'].fit(
                    np.concatenate(lists[group]['rec_values']),
                    np.concatenate(lists[group]['classes'])
                )
                knns['inputs']['labels'].fit(
                    np.concatenate(lists[group]['inputs']),
                    np.concatenate(lists[group]['classes'])
                )

            lists[group]['domain_preds'] = knns['enc']['domains'].predict(
                np.concatenate(lists[group]['encoded_values']),
            )

            try:
                values[group][metric]['enc']['domains'] += [funct(np.concatenate(lists[group]['domains']),
                                                           lists[group]['domain_preds'])]
            except:
                values[group][metric]['enc']['domains'] += [-1]

            lists[group]['domain_preds'] = knns['enc']['labels'].predict(
                np.concatenate(lists[group]['encoded_values']),
            )

            try:
                values[group][metric]['enc']['labels'] += [funct(np.concatenate(lists[group]['domains']),
                                                           lists[group]['domain_preds'])]
            except:
                values[group][metric]['enc']['labels'] += [-1]

            # lists[group]['domain_preds'] = knns['enc']['labels'].predict(
            #     np.concatenate(lists[group]['encoded_values']),
            # )


            #################

            lists[group]['domain_preds'] = knns['rec']['domains'].predict(
                np.concatenate(lists[group]['rec_values']),
            )

            try:
                values[group][metric]['rec']['domains'] += [funct(np.concatenate(lists[group]['domains']),
                                                                  lists[group]['domain_preds'])]
            except:
                values[group][metric]['rec']['domains'] += [-1]

            lists[group]['domain_preds'] = knns['rec']['labels'].predict(
                np.concatenate(lists[group]['rec_values']),
            )

            try:
                values[group][metric]['rec']['labels'] += [funct(np.concatenate(lists[group]['labels']),
                                                                  lists[group]['domain_preds'])]
            except:
                values[group][metric]['rec']['labels'] += [-1]

            # lists[group]['domain_preds'] = knns['rec']['labels'].predict(
            #     np.concatenate(lists[group]['rec']),
            # )

            #################

            lists[group]['domain_preds'] = knns['inputs']['domains'].predict(
                np.concatenate(lists[group]['inputs']),
            )

            try:
                values[group][metric]['inputs']['domains'] += [funct(np.concatenate(lists[group]['domains']),
                                                                  lists[group]['domain_preds'])]
            except:
                values[group][metric]['inputs']['domains'] += [-1]

            lists[group]['domain_preds'] = knns['inputs']['labels'].predict(
                np.concatenate(lists[group]['inputs']),
            )

            try:
                values[group][metric]['inputs']['labels'] += [funct(np.concatenate(lists[group]['labels']),
                                                                  lists[group]['domain_preds'])]
            except:
                values[group][metric]['inputs']['labels'] += [-1]

            # lists[group]['domain_preds'] = knns['inputs']['labels'].predict(
            #     np.concatenate(lists[group]['inputs']),
            # )

            # try:
            #     values[group][metric]['inputs']['domains'] += [funct(np.concatenate(lists[group]['domains']),
            #                                                       np.concatenate(lists[group]['domain_preds']).argmax(
            #                                                           1))]
            # except:
            #     values[group][metric]['inputs']['domains'] += [-1]

    for metric, funct in zip(['lisi', 'silhouette', 'kbet'], [rLISI, silhouette_score, rKBET]):
        try:
            values['set_batch_metrics'][metric]['enc']['set'] += [
                funct(np.concatenate(encoded_values), np.concatenate(sets))
            ]
        except:
            values['set_batch_metrics'][metric]['enc']['set'] += [-1]

        try:
            values['set_batch_metrics'][metric]['rec']['set'] += [
                funct(np.concatenate(rec_values), np.concatenate(sets))]
        except:
            values['set_batch_metrics'][metric]['rec']['set'] += [-1]

        try:
            values['set_batch_metrics'][metric]['enc']['labels'] += [
                funct(np.concatenate(encoded_values), np.concatenate(classes))]
        except:
            values['set_batch_metrics'][metric]['enc']['labels'] += [-1]

        try:
            values['set_batch_metrics'][metric]['rec']['labels'] += [
                funct(np.concatenate(rec_values), np.concatenate(classes))]
        except:
            values['set_batch_metrics'][metric]['rec']['labels'] += [-1]

        try:
            values['set_batch_metrics'][metric]['inputs']['labels'] += [
                funct(np.concatenate(inputs), np.concatenate(classes))]
        except:
            values['set_batch_metrics'][metric]['inputs']['labels'] += [-1]

        try:
            values['set_batch_metrics'][metric]['inputs']['set'] += [
                funct(np.concatenate(inputs), np.concatenate(sets))]
        except:
            values['set_batch_metrics'][metric]['inputs']['set'] += [-1]

    return values


def log_ORD(ord, logger, train_data, valid_data, test_data, train_labels,
            valid_labels, test_labels, unique_cats, epoch, transductive=False):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)

    model = ord['model']
    if transductive:
        model.fit(np.concatenate((train_data, valid_data, test_data)))
        pcs_train = model.transform(train_data)
    else:
        pcs_train = model.fit_transform(train_data)
    if valid_data is not None:
        pcs_valid = model.transform(valid_data)
    else:
        pcs_valid = np.array([])
    if test_data is not None:
        pcs_test = model.transform(test_data)
    else:
        pcs_test = np.array([])
        test_labels = np.array([])

    pcs_train_df = pd.DataFrame(data=pcs_train, columns=['PC 1', 'PC 2'])
    pcs_test_df = pd.DataFrame(data=pcs_valid, columns=['PC 1', 'PC 2'])
    pcs_inference_df = pd.DataFrame(data=pcs_test, columns=['PC 1', 'PC 2'])
    try:
        ev = model.explained_variance_ratio_
        pc1 = 'Component_1 : ' + str(np.round(ev[0] * 100, decimals=2)) + "%"
        pc2 = 'Component_2 : ' + str(np.round(ev[1] * 100, decimals=2)) + "%"
    except:
        pc1 = 'Component_1'
        pc2 = 'Component_2'

    ax.set_xlabel(pc1, fontsize=15)
    ax.set_ylabel(pc2, fontsize=15)
    ax.set_title(f"2 component {ord['name']}", fontsize=20)

    num_targets = len(unique_cats)
    cmap = plt.cm.tab20

    cols = cmap(np.linspace(0, 1, len(unique_cats) + 1))
    colors = rcParams['axes.prop_cycle'] = cycler(color=cols)
    colors_list = {name: [] for name in ['train_data', 'test_data', 'inference_data']}
    data1_list = {name: [] for name in ['train_data', 'test_data', 'inference_data']}
    data2_list = {name: [] for name in ['train_data', 'test_data', 'inference_data']}
    new_labels = {name: [] for name in ['train_data', 'test_data', 'inference_data']}
    new_cats = {name: [] for name in ['train_data', 'test_data', 'inference_data']}

    ellipses = []
    unique_cats_train = np.array([])
    for df_name, df, labels in zip(['train_data', 'test_data', 'inference_data'],
                                   [pcs_train_df, pcs_test_df, pcs_inference_df],
                                   [train_labels, valid_labels, test_labels]):
        for t, target in enumerate(unique_cats):
            # final_labels = list(train_labels)
            indices_to_keep = [True if x == target else False for x in
                               list(labels)]  # 0 is the name of the column with target values
            data1 = list(df.loc[indices_to_keep, 'PC 1'])
            new_labels[df_name] += [target for _ in range(len(data1))]
            new_cats[df_name] += [target for _ in range(len(data1))]

            data2 = list(df.loc[indices_to_keep, 'PC 2'])
            data1_list[df_name] += [data1]
            data2_list[df_name] += [data2]
            colors_list[df_name] += [np.array([[cols[t]] * len(data1)])]
            if len(indices_to_keep) > 1 and df_name == 'train_data' or target not in unique_cats_train:
                unique_cats_train = np.unique(np.concatenate((new_labels[df_name], unique_cats_train)))
                try:
                    confidence_ellipses = confidence_ellipse(np.array(data1), np.array(data2), ax, 1.5,
                                                             edgecolor=cols[t],
                                                             train_set=True)
                    ellipses += [confidence_ellipses[1]]
                except:
                    pass

    for df_name, marker in zip(list(data1_list.keys()), ['o', 'x', '*']):
        data1_vector = np.hstack([d for d in data1_list[df_name] if len(d) > 0]).reshape(-1, 1)
        colors_vector = np.hstack([d for d in colors_list[df_name] if d.shape[1] > 0]).squeeze()
        data2_vector = np.hstack(data2_list[df_name]).reshape(-1, 1)
        data_colors_vector = np.concatenate((data1_vector, data2_vector, colors_vector), axis=1)
        data2 = data_colors_vector[:, 1]
        col = data_colors_vector[:, 2:]
        data1 = data_colors_vector[:, 0]

        ax.scatter(data1, data2, s=50, alpha=1.0, c=col, label=new_labels[df_name], marker=marker)
        custom_lines = [Line2D([0], [0], color=cmap(x), lw=4) for x in np.linspace(0, 1, len(unique_cats) + 1)]
        ax.legend(custom_lines, unique_cats.tolist())

    # fig.savefig(f'ord.png')
    logger.add_figure(f'{ord["name"]}', fig, epoch)
    # fig.close()
    pass


def log_CCA(ord, logger, train_data, test_data, inference_data, train_labels,
            test_labels, inference_labels, unique_cats, epoch):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)

    model = ord['model']

    try:
        train_cats = OneHotEncoder().fit_transform(np.stack([np.argwhere(unique_cats == x) for x in train_labels]).reshape(-1, 1)).toarray()
    except:
        pass
    # test_cats = [np.argwhere(unique_cats == x) for x in test_labels]
    # inference_cats = [np.argwhere(unique_cats == x) for x in inference_labels]

    pcs_train, _ = model.fit_transform(train_data, train_cats)
    pcs_test = model.transform(test_data)
    pcs_inference = model.transform(inference_data)

    pcs_train_df = pd.DataFrame(data=pcs_train, columns=['PC 1', 'PC 2'])
    pcs_test_df = pd.DataFrame(data=pcs_test, columns=['PC 1', 'PC 2'])
    pcs_inference_df = pd.DataFrame(data=pcs_inference, columns=['PC 1', 'PC 2'])
    try:
        ev = model.explained_variance_ratio_
        pc1 = 'Component_1 : ' + str(np.round(ev[0] * 100, decimals=2)) + "%"
        pc2 = 'Component_2 : ' + str(np.round(ev[1] * 100, decimals=2)) + "%"
    except:
        pc1 = 'Component_1'
        pc2 = 'Component_2'

    ax.set_xlabel(pc1, fontsize=15)
    ax.set_ylabel(pc2, fontsize=15)
    ax.set_title(f"2 component {ord['name']}", fontsize=20)

    num_targets = len(unique_cats)
    cmap = plt.cm.tab20

    cols = cmap(np.linspace(0, 1, len(unique_cats) + 1))
    colors = rcParams['axes.prop_cycle'] = cycler(color=cols)
    colors_list = {name: [] for name in ['train_data', 'test_data', 'inference_data']}
    data1_list = {name: [] for name in ['train_data', 'test_data', 'inference_data']}
    data2_list = {name: [] for name in ['train_data', 'test_data', 'inference_data']}
    new_labels = {name: [] for name in ['train_data', 'test_data', 'inference_data']}
    new_cats = {name: [] for name in ['train_data', 'test_data', 'inference_data']}

    ellipses = []
    unique_cats_train = np.array([])
    for df_name, df, labels in zip(['train_data', 'test_data', 'inference_data'],
                                   [pcs_train_df, pcs_test_df, pcs_inference_df],
                                   [train_labels, test_labels, inference_labels]):
        for t, target in enumerate(unique_cats):
            # final_labels = list(train_labels)
            indices_to_keep = [True if x == target else False for x in
                               list(labels)]  # 0 is the name of the column with target values
            data1 = list(df.loc[indices_to_keep, 'PC 1'])
            new_labels[df_name] += [target for _ in range(len(data1))]
            new_cats[df_name] += [target for _ in range(len(data1))]

            data2 = list(df.loc[indices_to_keep, 'PC 2'])
            data1_list[df_name] += [data1]
            data2_list[df_name] += [data2]
            colors_list[df_name] += [np.array([[cols[t]] * len(data1)])]
            if len(indices_to_keep) > 1 and df_name == 'train_data' or target not in unique_cats_train:
                unique_cats_train = np.unique(np.concatenate((new_labels[df_name], unique_cats_train)))
                try:
                    confidence_ellipses = confidence_ellipse(np.array(data1), np.array(data2), ax, 1.5,
                                                             edgecolor=cols[t],
                                                             train_set=True)
                    ellipses += [confidence_ellipses[1]]
                except:
                    pass

    for df_name, marker in zip(list(data1_list.keys()), ['o', 'x', '*']):
        data1_vector = np.hstack([d for d in data1_list[df_name] if len(d) > 0]).reshape(-1, 1)
        colors_vector = np.hstack([d for d in colors_list[df_name] if d.shape[1] > 0]).squeeze()
        data2_vector = np.hstack(data2_list[df_name]).reshape(-1, 1)
        data_colors_vector = np.concatenate((data1_vector, data2_vector, colors_vector), axis=1)
        data2 = data_colors_vector[:, 1]
        col = data_colors_vector[:, 2:]
        data1 = data_colors_vector[:, 0]

        ax.scatter(data1, data2, s=50, alpha=1.0, c=col, label=new_labels[df_name], marker=marker)
        custom_lines = [Line2D([0], [0], color=cmap(x), lw=4) for x in np.linspace(0, 1, len(unique_cats) + 1)]
        ax.legend(custom_lines, unique_cats.tolist())

    # fig.savefig(f'ord.png')
    logger.add_figure(f'{ord["name"]}', fig, epoch)
    # fig.close()
    pass


def log_TSNE(logger, name, train_data, test_data, inference_data, train_labels,
             test_labels, inference_labels, unique_cats, epoch):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)

    model = TSNE(n_components=2)
    pcs_all = model.fit_transform(np.concatenate((train_data, test_data, inference_data)))
    pcs_train = pcs_all[:train_data.shape[0], :]
    pcs_test = pcs_all[train_data.shape[0]:train_data.shape[0] + test_data.shape[0], :]
    pcs_inference = pcs_all[train_data.shape[0] + test_data.shape[0]:, :]

    pcs_train_df = pd.DataFrame(data=pcs_train, columns=['PC 1', 'PC 2'])
    pcs_test_df = pd.DataFrame(data=pcs_test, columns=['PC 1', 'PC 2'])
    pcs_inference_df = pd.DataFrame(data=pcs_inference, columns=['PC 1', 'PC 2'])
    ax.set_xlabel('Axis_1', fontsize=15)
    ax.set_ylabel('Axis_2', fontsize=15)

    num_targets = len(unique_cats)
    cmap = plt.cm.tab20

    cols = cmap(np.linspace(0, 1, len(unique_cats) + 1))
    colors = rcParams['axes.prop_cycle'] = cycler(color=cols)
    colors_list = {name: [] for name in ['train_data', 'test_data', 'inference_data']}
    data1_list = {name: [] for name in ['train_data', 'test_data', 'inference_data']}
    data2_list = {name: [] for name in ['train_data', 'test_data', 'inference_data']}
    new_labels = {name: [] for name in ['train_data', 'test_data', 'inference_data']}
    new_cats = {name: [] for name in ['train_data', 'test_data', 'inference_data']}

    data1_blk = {name: [] for name in ['train_data', 'test_data', 'inference_data']}
    data2_blk = {name: [] for name in ['train_data', 'test_data', 'inference_data']}
    ellipses = []
    unique_cats_train = np.array([])
    for df_name, df, labels in zip(['train_data', 'test_data', 'inference_data'],
                                   [pcs_train_df, pcs_test_df, pcs_inference_df],
                                   [train_labels, test_labels, inference_labels]):
        for t, target in enumerate(unique_cats):
            # final_labels = list(train_labels)
            indices_to_keep = [True if x == target else False for x in
                               list(labels)]  # 0 is the name of the column with target values
            data1 = list(df.loc[indices_to_keep, 'PC 1'])
            new_labels[df_name] += [target for _ in range(len(data1))]
            new_cats[df_name] += [target for _ in range(len(data1))]

            data2 = list(df.loc[indices_to_keep, 'PC 2'])
            data1_list[df_name] += [data1]
            data2_list[df_name] += [data2]
            colors_list[df_name] += [np.array([[cols[t]] * len(data1)])]
            if len(indices_to_keep) > 1 and df_name == 'train_data' or target not in unique_cats_train:
                try:
                    unique_cats_train = np.unique(np.concatenate((new_labels[df_name], unique_cats_train)))
                    confidence_ellipses = confidence_ellipse(np.array(data1), np.array(data2), ax, 1.5,
                                                             edgecolor=cols[t],
                                                             train_set=True)
                    ellipses += [confidence_ellipses[1]]
                except:
                    pass
    for df_name, marker in zip(list(data1_list.keys()), ['o', 'x', '*']):
        data1_vector = np.hstack([d for d in data1_list[df_name] if len(d) > 0]).reshape(-1, 1)
        colors_vector = np.hstack([d for d in colors_list[df_name] if d.shape[1] > 0]).squeeze()
        data2_vector = np.hstack(data2_list[df_name]).reshape(-1, 1)
        data_colors_vector = np.concatenate((data1_vector, data2_vector, colors_vector), axis=1)
        data2 = data_colors_vector[:, 1]
        col = data_colors_vector[:, 2:]
        data1 = data_colors_vector[:, 0]

        ax.scatter(data1, data2, s=50, alpha=1.0, c=col, label=new_labels[df_name], marker=marker)
        custom_lines = [Line2D([0], [0], color=cmap(x), lw=4) for x in np.linspace(0, 1, len(unique_cats) + 1)]
        ax.legend(custom_lines, unique_cats.tolist())

    fig.savefig('tsne.png')
    logger.add_figure(f"{name}", fig, epoch)
    pass


def log_stuff(logger_images, logger, lists, values, traces, classifier, unique_labels, plates, epoch, device='cuda'):
    if len(unique_labels) > 2:
        bout = 0
    else:
        bout = 1
    values = log_metrics(lists, values)
    for repres in ['enc', 'rec', 'inputs']:
        for metric in ['silhouette', 'kbet', 'lisi']:
            for info in ['labels', 'domains']:
                for group in ['train', 'valid', 'test']:
                    if metric == 'lisi':
                        try:
                            logger.add_scalar(f'{metric}/{group}/{repres}/{info}',
                                              values[group][metric][repres][info][0][0], epoch)
                        except:
                            logger.add_scalar(f'{metric}/{group}/{repres}/{info}',
                                              values[group][metric][repres][info][0], epoch)

                    elif metric == 'silhouette':
                        logger.add_scalar(f'{metric}/{group}/{repres}/{info}',
                                          values[group][metric][repres][info][0], epoch)
                    elif metric == 'kbet':
                        try:
                            logger.add_scalar(f'{metric}/{group}/{repres}/{info}',
                                              values[group][metric][repres][info][0], epoch)
                        except:
                            pass
            logger.add_scalar(f'{metric}/set/{repres}/labels',
                              values['set_batch_metrics'][metric][repres]['labels'][0], epoch)
            logger.add_scalar(f'{metric}/set/{repres}/set',
                              values['set_batch_metrics'][metric][repres]['set'][0], epoch)
            if metric != 'lisi':
                logger.add_scalar(f'F1_{metric}/set/{repres}',
                                  batch_f1_score(
                                      batch_score=values['set_batch_metrics'][metric][repres]['set'][0],
                                      class_score=values['set_batch_metrics'][metric][repres]['labels'][0],
                                  ),
                                  # values['set_batch_metrics'][metric][repres]['set'][0],
                                  epoch)
            else:
                logger.add_scalar(f'F1_{metric}/set/{repres}',
                                  batch_f1_score(
                                      batch_score=values['set_batch_metrics'][metric][repres]['set'][0]/len(unique_labels),
                                      class_score=values['set_batch_metrics'][metric][repres]['labels'][0]/len(plates),
                                  ),
                                  # values['set_batch_metrics'][metric][repres]['set'][0],
                                  epoch)

            for group in ['train', 'valid', 'test']:
                if metric == 'lisi':
                    try:
                        logger.add_scalar(f'F1_{metric}/{group}/{repres}',
                                          batch_f1_score(
                                              batch_score=values[group][metric][repres]['domains'][0][0]/len(plates),
                                              class_score=values[group][metric][repres]['labels'][0][0]/len(unique_labels),
                                          ),
                                          epoch)
                    except:
                        logger.add_scalar(f'F1_{metric}/{group}/{repres}',
                                          batch_f1_score(
                                              batch_score=values[group][metric][repres]['domains'][0]/len(plates),
                                              class_score=values[group][metric][repres]['labels'][0]/len(unique_labels),
                                          ),
                                          epoch)

                elif metric == 'silhouette':
                    logger.add_scalar(f'F1_{metric}/{group}/{repres}',
                                      batch_f1_score(
                                          batch_score=values[group][metric][repres]['domains'][0],
                                          class_score=values[group][metric][repres]['labels'][0],
                                      ),
                                      epoch)
                elif metric == 'kbet':
                    try:
                        logger.add_scalar(f'F1_{metric}/{group}/{repres}',
                                          batch_f1_score(
                                              batch_score=values[group][metric][repres]['domains'][0],
                                              class_score=values[group][metric][repres]['labels'][0],
                                          ),
                                          epoch)
                    except:
                        pass

    for repres in ['enc', 'rec', 'inputs']:
        for metric in ['adjusted_rand_score', 'adjusted_mutual_info_score']:
            for group in ['train', 'valid', 'test']:
                for info in ['labels', 'domains']:
                    logger.add_scalar(f'{metric}/{group}/{repres}/{info}', values[group][metric][repres][info][0], epoch)
                logger.add_scalar(f'F1_{metric}/{group}/{repres}',
                                  batch_f1_score(
                                      batch_score=values[group][metric][repres]['domains'][0],
                                      class_score=values[group][metric][repres]['labels'][0],
                                  ),
                                  epoch)

    clusters = ['labels', 'domains']
    reps = ['enc', 'rec']
    train_lisi_enc = [values['train']['lisi']['enc'][c][0].reshape(-1) for c in clusters]
    train_lisi_rec = [values['train']['lisi']['rec'][c][0].reshape(-1) for c in clusters]
    valid_lisi_enc = [values['valid']['lisi']['enc'][c][0].reshape(-1) for c in clusters]
    valid_lisi_rec = [values['valid']['lisi']['rec'][c][0].reshape(-1) for c in clusters]
    test_lisi_enc = [values['test']['lisi']['enc'][c][0].reshape(-1) for c in clusters]
    test_lisi_rec = [values['test']['lisi']['rec'][c][0].reshape(-1) for c in clusters]

    set_lisi_enc = [values['set_batch_metrics']['lisi']['enc'][c][0].reshape(-1) for c in ['labels', 'set']]
    set_lisi_rec = [values['set_batch_metrics']['lisi']['rec'][c][0].reshape(-1) for c in ['labels', 'set']]

    lisi_df_sets = pd.DataFrame(np.concatenate((
        np.concatenate((np.concatenate(set_lisi_enc), np.concatenate(set_lisi_rec))).reshape(1, -1),
        np.array(['enc', 'enc', 'rec', 'rec']).reshape(1, -1),
    ), 0).T, columns=['lisi', 'representation'])

    lisi_df_sets['lisi'] = pd.to_numeric(lisi_df_sets['lisi'])
    sns.set_theme(style="whitegrid")
    figure = plt.figure(figsize=(8, 8))
    ax = sns.boxplot(x="representation", y="lisi", data=lisi_df_sets)
    logger_images.add_figure(f"LISI_sets", figure, epoch)

    lisi_set_train_enc = np.concatenate(
        [np.array([s for _ in values]) for s, values in zip(clusters, train_lisi_enc)]
    )
    lisi_set_train_rec = np.concatenate(
        [np.array([s for _ in values]) for s, values in zip(clusters, train_lisi_rec)]
    )
    lisi_train_reps = np.concatenate(
        [np.array([r for _ in np.concatenate(values)]) for r, values in zip(reps, [train_lisi_enc, train_lisi_rec])]
    )
    lisi_set_valid_enc = np.concatenate(
        [np.array([s for _ in values]) for s, values in zip(clusters, valid_lisi_enc)]
    )
    lisi_set_valid_rec = np.concatenate(
        [np.array([s for _ in values]) for s, values in zip(clusters, valid_lisi_rec)]
    )
    lisi_valid_reps = np.concatenate(
        [np.array([r for _ in np.concatenate(values)]) for r, values in zip(reps, [valid_lisi_enc, valid_lisi_rec])]
    )
    lisi_set_test_enc = np.concatenate(
        [np.array([s for _ in values]) for s, values in zip(clusters, test_lisi_enc)]
    )
    lisi_set_test_rec = np.concatenate(
        [np.array([s for _ in values]) for s, values in zip(clusters, test_lisi_rec)]
    )
    lisi_test_reps = np.concatenate(
        [np.array([r for _ in np.concatenate(values)]) for r, values in zip(reps, [test_lisi_enc, test_lisi_rec])]
    )
    lisi_df_train = pd.DataFrame(np.concatenate((
        np.concatenate((np.concatenate(train_lisi_enc), np.concatenate(train_lisi_rec))).reshape(1, -1),
        lisi_train_reps.reshape(1, -1),
        np.concatenate((lisi_set_train_enc, lisi_set_train_rec)).reshape(1, -1),
    ), 0).T, columns=['lisi', 'representation', 'set'])
    lisi_df_train['lisi'] = pd.to_numeric(lisi_df_train['lisi'])
    lisi_df_valid = pd.DataFrame(np.concatenate((
        np.concatenate((valid_lisi_enc, valid_lisi_rec)).reshape(1, -1),
        lisi_valid_reps.reshape(1, -1),
        np.concatenate((lisi_set_valid_enc, lisi_set_valid_rec)).reshape(1, -1),
    ), 0).T, columns=['lisi', 'representation', 'set'])
    lisi_df_valid['lisi'] = pd.to_numeric(lisi_df_valid['lisi'])
    lisi_df_test = pd.DataFrame(np.concatenate((
        np.concatenate((test_lisi_enc, test_lisi_rec)).reshape(1, -1),
        lisi_test_reps.reshape(1, -1),
        np.concatenate((lisi_set_test_enc, lisi_set_test_rec)).reshape(1, -1),
    ), 0).T, columns=['lisi', 'representation', 'set'])
    lisi_df_test['lisi'] = pd.to_numeric(lisi_df_test['lisi'])
    # lisi_means = [values[s]['lisi'][0] for s in ['train', 'valid', 'test']]

    sns.set_theme(style="whitegrid")
    figure = plt.figure(figsize=(8, 8))
    ax = sns.boxplot(x="set", y="lisi", hue='representation', data=lisi_df_train)
    logger_images.add_figure(f"LISI_train", figure, epoch)

    figure = plt.figure(figsize=(8, 8))
    ax = sns.boxplot(x="set", y="lisi", hue='representation', data=lisi_df_valid)
    logger_images.add_figure(f"LISI_valid", figure, epoch)

    figure = plt.figure(figsize=(8, 8))
    ax = sns.boxplot(x="set", y="lisi", hue='representation', data=lisi_df_test)
    logger_images.add_figure(f"LISI_test", figure, epoch)
    sns.set_theme(style="white")
    log_confusion_matrix(logger_images, epoch,
                         lists,
                         unique_labels, traces)
    # save_roc_curve(classifier,
    #                torch.Tensor(np.concatenate(lists['train']['encoded_values'])).to(device),
    #                np.concatenate(lists['train']['classes']),
    #                self.unique_labels, name='./roc_train', binary=self.bout, epoch=epoch,
    #                acc=values['train']['acc'][-1], logger=logger_images)
    save_roc_curve(classifier,
                   torch.Tensor(np.concatenate(lists['valid']['encoded_values'])).to(device),
                   np.concatenate(lists['valid']['classes']),
                   unique_labels, name='./roc_valid', binary=bout, epoch=epoch,
                   acc=values['valid']['acc'][-1], logger=logger_images)
    save_roc_curve(classifier,
                   torch.Tensor(np.concatenate(lists['test']['encoded_values'])).to(device),
                   np.concatenate(lists['test']['classes']),
                   unique_labels, name='./roc_test', binary=bout, epoch=epoch,
                   acc=values['test']['acc'][-1], logger=logger_images)
    # save_precision_recall_curve(classifier,
    #                             torch.Tensor(np.concatenate(lists['train']['encoded_values'])).to(
    #                                 device),
    #                             np.concatenate(lists['train']['classes']),
    #                             self.unique_labels, name='./prc_train', binary=self.bout, epoch=epoch,
    #                             acc=values['train']['acc'][-1], logger=logger_cm)
    save_precision_recall_curve(classifier,
                                torch.Tensor(np.concatenate(lists['valid']['encoded_values'])).to(
                                    device),
                                np.concatenate(lists['valid']['classes']),
                                unique_labels, name='./prc_valid', binary=bout, epoch=epoch,
                                acc=values['valid']['acc'][-1], logger=logger_images)
    save_precision_recall_curve(classifier,
                                torch.Tensor(np.concatenate(lists['test']['encoded_values'])).to(
                                    device),
                                np.concatenate(lists['test']['classes']),
                                unique_labels, name='./prc_test', binary=bout, epoch=epoch,
                                acc=values['test']['acc'][-1], logger=logger_images)
    for labs in ['domains', 'labels']:
        unique_labels = np.unique(np.concatenate((np.concatenate(lists['train'][labs]),
                                                  np.concatenate(lists['valid'][labs]),
                                                  np.concatenate(lists['test'][labs]))))
        log_CCA({'model': CCA(n_components=2), 'name': f'CCA_encs_{labs}'}, logger_images,
                np.concatenate(lists['train']['encoded_values']),
                np.concatenate(lists['valid']['encoded_values']),
                np.concatenate(lists['test']['encoded_values']), np.concatenate(lists['train'][labs]),
                np.concatenate(lists['valid'][labs]), np.concatenate(lists['test'][labs]),
                unique_labels, epoch)
        log_ORD({'model': PCA(n_components=2), 'name': f'PCA_encs_{labs}'}, logger_images,
                np.concatenate(lists['train']['encoded_values']),
                np.concatenate(lists['valid']['encoded_values']),
                np.concatenate(lists['test']['encoded_values']), np.concatenate(lists['train'][labs]),
                np.concatenate(lists['valid'][labs]), np.concatenate(lists['test'][labs]),
                unique_labels, epoch)
        log_ORD({'model': PCA(n_components=2), 'name': f'PCA_encs_{labs}_transductive'}, logger_images,
                np.concatenate(lists['train']['encoded_values']),
                np.concatenate(lists['valid']['encoded_values']),
                np.concatenate(lists['test']['encoded_values']), np.concatenate(lists['train'][labs]),
                np.concatenate(lists['valid'][labs]), np.concatenate(lists['test'][labs]),
                unique_labels, epoch, transductive=True)
        log_TSNE(logger_images, f'TSNE_encs_{labs}', np.concatenate(lists['train']['encoded_values']),
                 np.concatenate(lists['valid']['encoded_values']),
                 np.concatenate(lists['test']['encoded_values']), np.concatenate(lists['train'][labs]),
                 np.concatenate(lists['valid'][labs]), np.concatenate(lists['test'][labs]),
                 unique_labels, epoch)
        log_ORD({'model': UMAP(n_neighbors=5, min_dist=0.3, metric='correlation'), 'name': f'UMAP_encs_{labs}'},
                logger_images, np.concatenate(lists['train']['encoded_values']),
                np.concatenate(lists['valid']['encoded_values']),
                np.concatenate(lists['test']['encoded_values']), np.concatenate(lists['train'][labs]),
                np.concatenate(lists['valid'][labs]), np.concatenate(lists['test'][labs]),
                unique_labels, epoch, transductive=True)
        log_ORD({'model': UMAP(n_neighbors=5, min_dist=0.3, metric='correlation'), 'name': f'UMAP_encs_{labs}'},
                logger_images, np.concatenate(lists['train']['encoded_values']),
                np.concatenate(lists['valid']['encoded_values']),
                np.concatenate(lists['test']['encoded_values']), np.concatenate(lists['train'][labs]),
                np.concatenate(lists['valid'][labs]), np.concatenate(lists['test'][labs]),
                unique_labels, epoch)
        log_CCA({'model': CCA(n_components=2), 'name': f'CCA_recs_{labs}'}, logger_images,
                np.concatenate(lists['train']['rec_values']),
                np.concatenate(lists['valid']['rec_values']), np.concatenate(lists['test']['rec_values']),
                np.concatenate(lists['train'][labs]),
                np.concatenate(lists['valid'][labs]), np.concatenate(lists['test'][labs]),
                unique_labels, epoch)
        log_ORD({'model': PCA(n_components=2), 'name': f'PCA_recs_{labs}'}, logger_images,
                np.concatenate(lists['train']['rec_values']),
                np.concatenate(lists['valid']['rec_values']), np.concatenate(lists['test']['rec_values']),
                np.concatenate(lists['train'][labs]),
                np.concatenate(lists['valid'][labs]), np.concatenate(lists['test'][labs]),
                unique_labels, epoch)
        log_ORD({'model': PCA(n_components=2), 'name': f'PCA_recs_{labs}_transductive'}, logger_images,
                np.concatenate(lists['train']['rec_values']),
                np.concatenate(lists['valid']['rec_values']), np.concatenate(lists['test']['rec_values']),
                np.concatenate(lists['train'][labs]),
                np.concatenate(lists['valid'][labs]), np.concatenate(lists['test'][labs]),
                unique_labels, epoch, transductive=True)
        log_TSNE(logger_images, f'TSNE_recs_{labs}', np.concatenate(lists['train']['rec_values']),
                 np.concatenate(lists['valid']['rec_values']),
                 np.concatenate(lists['test']['rec_values']), np.concatenate(lists['train'][labs]),
                 np.concatenate(lists['valid'][labs]), np.concatenate(lists['test'][labs]),
                 unique_labels, epoch)
        log_ORD({'model': UMAP(n_neighbors=5, min_dist=0.3, metric='correlation'), 'name': f'UMAP_recs_{labs}'},
                logger_images, np.concatenate(lists['train']['rec_values']),
                np.concatenate(lists['valid']['rec_values']),
                np.concatenate(lists['test']['rec_values']), np.concatenate(lists['train'][labs]),
                np.concatenate(lists['valid'][labs]), np.concatenate(lists['test'][labs]),
                unique_labels, epoch)
        log_ORD({'model': UMAP(n_neighbors=5, min_dist=0.3, metric='correlation'), 'name': f'UMAP_recs_{labs}_transductive'},
                logger_images, np.concatenate(lists['train']['rec_values']),
                np.concatenate(lists['valid']['rec_values']),
                np.concatenate(lists['test']['rec_values']), np.concatenate(lists['train'][labs]),
                np.concatenate(lists['valid'][labs]), np.concatenate(lists['test'][labs]),
                unique_labels, epoch, transductive=True)

    # log_CCA({'model': CCA(n_components=2), 'name': f'CCA_encs_set'}, logger_images,
    #         np.concatenate(lists['train']['encoded_values']),
    #         np.concatenate(lists['valid']['encoded_values']),
    #         np.concatenate(lists['test']['encoded_values']), np.concatenate(lists['train']['set']),
    #         np.concatenate(lists['valid']['set']), np.concatenate(lists['test']['set']),
    #         np.array(['train', 'valid', 'test']), epoch)
    log_ORD({'model': PCA(n_components=2), 'name': f'PCA_encs_set'}, logger_images,
            np.concatenate(lists['train']['encoded_values']),
            np.concatenate(lists['valid']['encoded_values']),
            np.concatenate(lists['test']['encoded_values']), np.concatenate(lists['train']['set']),
            np.concatenate(lists['valid']['set']), np.concatenate(lists['test']['set']),
            np.array(['train', 'valid', 'test']), epoch)
    log_ORD({'model': PCA(n_components=2), 'name': f'PCA_encs_set_transductive'}, logger_images,
            np.concatenate(lists['train']['encoded_values']),
            np.concatenate(lists['valid']['encoded_values']),
            np.concatenate(lists['test']['encoded_values']), np.concatenate(lists['train']['set']),
            np.concatenate(lists['valid']['set']), np.concatenate(lists['test']['set']),
            np.array(['train', 'valid', 'test']), epoch, transductive=True)
    log_TSNE(logger_images, f'TSNE_encs_set', np.concatenate(lists['train']['encoded_values']),
             np.concatenate(lists['valid']['encoded_values']),
             np.concatenate(lists['test']['encoded_values']), np.concatenate(lists['train']['set']),
             np.concatenate(lists['valid']['set']), np.concatenate(lists['test']['set']),
             np.array(['train', 'valid', 'test']), epoch)
    log_ORD({'model': UMAP(n_neighbors=5, min_dist=0.3, metric='correlation'), 'name': f'UMAP_encs_set'},
            logger_images, np.concatenate(lists['train']['encoded_values']),
            np.concatenate(lists['valid']['encoded_values']),
            np.concatenate(lists['test']['encoded_values']), np.concatenate(lists['train']['set']),
            np.concatenate(lists['valid']['set']), np.concatenate(lists['test']['set']),
            np.array(['train', 'valid', 'test']), epoch)
    log_ORD({'model': UMAP(n_neighbors=5, min_dist=0.3, metric='correlation'), 'name': f'UMAP_encs_set_transductive'},
            logger_images, np.concatenate(lists['train']['encoded_values']),
            np.concatenate(lists['valid']['encoded_values']),
            np.concatenate(lists['test']['encoded_values']), np.concatenate(lists['train']['set']),
            np.concatenate(lists['valid']['set']), np.concatenate(lists['test']['set']),
            np.array(['train', 'valid', 'test']), epoch, transductive=True)
    log_ORD({'model': PCA(n_components=2), 'name': f'PCA_recs_set'}, logger_images,
            np.concatenate(lists['train']['rec_values']),
            np.concatenate(lists['valid']['rec_values']), np.concatenate(lists['test']['rec_values']),
            np.concatenate(lists['train']['set']),
            np.concatenate(lists['valid']['set']), np.concatenate(lists['test']['set']),
            np.array(['train', 'valid', 'test']), epoch)
    # log_CCA({'model': CCA(n_components=2), 'name': f'CCA_recs_set'}, logger_images,
    #         np.concatenate(lists['train']['rec_values']),
    #         np.concatenate(lists['valid']['rec_values']), np.concatenate(lists['test']['rec_values']),
    #         np.concatenate(lists['train']['set']),
    #         np.concatenate(lists['valid']['set']), np.concatenate(lists['test']['set']),
    #         np.array(['train', 'valid', 'test']), epoch)
    log_ORD({'model': PCA(n_components=2), 'name': f'PCA_recs_set_transductive'}, logger_images,
            np.concatenate(lists['train']['rec_values']),
            np.concatenate(lists['valid']['rec_values']), np.concatenate(lists['test']['rec_values']),
            np.concatenate(lists['train']['set']),
            np.concatenate(lists['valid']['set']), np.concatenate(lists['test']['set']),
            np.array(['train', 'valid', 'test']), epoch, transductive=True)
    log_TSNE(logger_images, f'TSNE_recs_set', np.concatenate(lists['train']['rec_values']),
             np.concatenate(lists['valid']['rec_values']),
             np.concatenate(lists['test']['rec_values']), np.concatenate(lists['train']['set']),
             np.concatenate(lists['valid']['set']), np.concatenate(lists['test']['set']),
             np.array(['train', 'valid', 'test']), epoch)
    log_ORD({'model': UMAP(n_neighbors=5, min_dist=0.3, metric='correlation'), 'name': f'UMAP_recs_set'},
            logger_images, np.concatenate(lists['train']['rec_values']),
            np.concatenate(lists['valid']['rec_values']),
            np.concatenate(lists['test']['rec_values']), np.concatenate(lists['train']['set']),
            np.concatenate(lists['valid']['set']), np.concatenate(lists['test']['set']),
            np.array(['train', 'valid', 'test']), epoch)
    log_ORD({'model': UMAP(n_neighbors=5, min_dist=0.3, metric='correlation'), 'name': f'UMAP_recs_set_transductive'},
            logger_images, np.concatenate(lists['train']['rec_values']),
            np.concatenate(lists['valid']['rec_values']),
            np.concatenate(lists['test']['rec_values']), np.concatenate(lists['train']['set']),
            np.concatenate(lists['valid']['set']), np.concatenate(lists['test']['set']),
            np.array(['train', 'valid', 'test']), epoch, transductive=True)


def log_input_ordination(logger, data, inputs, epoch=0):
    """

    Args:
        logger: SummaryWriter object
        train_data: Training set
        valid_data: Valid set (optional)
        test_data: Test set (optional)
        epoch: Epoch

    Returns:
        Nothing
    """
    unique_labels = get_unique_labels(data['labels']['all'])
    log_CCA({'model': CCA(n_components=2), 'name': 'CCA_inputs_classes'}, logger, inputs['train'], inputs['valid'],
            inputs['test'], data['labels']['train'], data['labels']['valid'], data['labels']['test'],
            unique_labels, epoch)
    log_ORD({'model': PCA(n_components=2), 'name': 'PCA_inputs_classes'}, logger, inputs['train'], inputs['valid'],
            inputs['test'], data['labels']['train'], data['labels']['valid'], data['labels']['test'],
            unique_labels, epoch)
    log_ORD({'model': PCA(n_components=2), 'name': 'PCA_inputs_classes_transductive'}, logger, inputs['train'],
            inputs['valid'],
            inputs['test'], data['labels']['train'], data['labels']['valid'], data['labels']['test'],
            unique_labels, epoch,
            transductive=True)
    log_TSNE(logger, 'TSNE_inputs_classes', inputs['train'], inputs['valid'], inputs['test'],
             data['labels']['train'], data['labels']['valid'],
             data['labels']['test'], unique_labels, epoch)
    log_ORD({'model': UMAP(n_neighbors=5, min_dist=0.3, metric='correlation'), 'name': 'UMAP_inputs_classes'},
            logger, inputs['train'], inputs['valid'], inputs['test'], data['labels']['train'],
            data['labels']['valid'],
            data['labels']['test'], unique_labels, epoch)
    log_ORD({'model': UMAP(n_neighbors=5, min_dist=0.3, metric='correlation'),
             'name': 'UMAP_inputs_classes_transductive'},
            logger, inputs['train'], inputs['valid'], inputs['test'], data['labels']['train'],
            data['labels']['valid'],
            data['labels']['test'], unique_labels, epoch, transductive=True)
    unique_plates = np.unique(
        np.concatenate(
            (data['batches']['train'], data['batches']['valid'], data['batches']['test'])))

    log_CCA({'model': CCA(n_components=2), 'name': 'CCA_inputs_plates'}, logger, inputs['train'], inputs['valid'],
            inputs['test'], data['batches']['train'], data['batches']['valid'],
            data['batches']['test'], unique_plates, epoch)
    log_ORD({'model': PCA(n_components=2), 'name': 'PCA_inputs_plates'}, logger, inputs['train'], inputs['valid'],
            inputs['test'], data['batches']['train'], data['batches']['valid'],
            data['batches']['test'], unique_plates, epoch)
    log_ORD({'model': PCA(n_components=2), 'name': 'PCA_inputs_plates_transductive'}, logger, inputs['train'],
            inputs['valid'],
            inputs['test'], data['batches']['train'], data['batches']['valid'],
            data['batches']['test'], unique_plates, epoch,
            transductive=True)
    log_TSNE(logger, 'TSNE_inputs_plates', inputs['train'], inputs['valid'], inputs['test'],
             data['batches']['train'], data['batches']['valid'],
             data['batches']['test'], unique_plates, epoch)
    log_ORD({'model': UMAP(n_neighbors=5, min_dist=0.3, metric='correlation'), 'name': 'UMAP_inputs_plates'},
            logger, inputs['train'], inputs['valid'], inputs['test'], data['batches']['train'],
            data['batches']['valid'],
            data['batches']['test'], unique_plates, epoch)
    log_ORD({'model': UMAP(n_neighbors=5, min_dist=0.3, metric='correlation'),
             'name': 'UMAP_inputs_plates_transductive'},
            logger, inputs['train'], inputs['valid'], inputs['test'], data['batches']['train'],
            data['batches']['valid'],
            data['batches']['test'], unique_plates, epoch, transductive=True)

    train_sets = ['train' for _ in data['batches']['train']]
    valid_sets = ['valid' for _ in data['batches']['valid']]
    test_sets = ['test' for _ in data['batches']['test']]
    unique_sets = np.array(['train', 'valid', 'test'])

    # Can't do CCA because train_sets only has a single class
    # log_CCA({'model': CCA(n_components=2), 'name': 'CCA_inputs_sets'}, logger, inputs['train'], inputs['valid'], inputs['test'],
    #         train_sets, valid_sets, test_sets, unique_sets, epoch)
    log_ORD({'model': PCA(n_components=2), 'name': 'PCA_inputs_sets'}, logger, inputs['train'], inputs['valid'],
            inputs['test'], train_sets, valid_sets, test_sets, unique_sets, epoch)
    log_ORD({'model': PCA(n_components=2), 'name': 'PCA_inputs_sets_transductive'}, logger, inputs['train'],
            inputs['valid'],
            inputs['test'], train_sets, valid_sets, test_sets, unique_sets, epoch, transductive=True)
    log_TSNE(logger, 'TSNE_inputs_sets', inputs['train'], inputs['valid'], inputs['test'], train_sets, valid_sets,
             test_sets, unique_sets, epoch)
    log_ORD({'model': UMAP(n_neighbors=5, min_dist=0.3, metric='correlation'), 'name': 'UMAP_inputs_sets'},
            logger, inputs['train'], inputs['valid'], inputs['test'], train_sets, valid_sets, test_sets,
            unique_sets, epoch)
    log_ORD(
        {'model': UMAP(n_neighbors=5, min_dist=0.3, metric='correlation'), 'name': 'UMAP_inputs_sets_transductive'},
        logger, inputs['train'], inputs['valid'], inputs['test'], train_sets, valid_sets, test_sets, unique_sets,
        epoch,
        transductive=True)