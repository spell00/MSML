import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
import matplotlib
import seaborn as sns

matplotlib.use('Agg')
CUDA_VISIBLE_DEVICES = ""
from umap import UMAP
import matplotlib.pyplot as plt
import math
import random
import json
import copy
import torch
from itertools import cycle
from torch import nn
import tensorflow as tf
from torch.utils.data import DataLoader
import torchvision
from msml.scikit_learn.utils import get_unique_labels
import os
from msml.dl.models.pytorch.utils.loggings import TensorboardLoggingAE, log_metrics, log_ORD, log_TSNE, log_CCA
from tensorboardX import SummaryWriter
from sklearn.model_selection import StratifiedKFold
from ax.service.managed_loop import optimize
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from msml.dl.models.pytorch.aedann import AutoEncoder, Classifier, to_categorical, ReverseLayerF
from msml.dl.utils.dataset import MSDataset3
from msml.utils.batch_effect_removal import comBatR, harmonyR
from msml.dl.utils.utils import log_confusion_matrix, save_roc_curve, save_precision_recall_curve, \
    get_best_values_from_tb, get_best_values, get_empty_traces, log_traces, get_empty_dicts, \
    add_to_logger, count_labels

import warnings

warnings.filterwarnings("ignore")

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def batch_f1_score(batch_score, class_score):
    return 2 * (1 - batch_score) * (class_score) / (1 - batch_score + class_score)


def get_optimizer(model, learning_rate, weight_decay, optimizer_type, momentum=0.9):
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay,
                                     )
    else:
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay,
                                    momentum=momentum
                                    )
    return optimizer


class Train:
    def __init__(self, path, run_name='all', prescaler='robust', log2='after', early_stop=100,
                 early_warmup_stop=100, warmup=1000, random_recs=False, one_model=0, train_after_warmup=1,
                 balanced_rec_loader=1, variational=0, zinb=0, dann_plates=0, dann_sets=0, predict_tests=0,
                 n_epochs=10000, dop=1, bout=0, combat=0, add_noise=0, scaler='binarize', tl=0,
                 loss='mse', inference_inputs=0, tied_weights=0, alpha_warmup=1000, logging=False,
                 load_tb=True, model_name='ae_classifier', features_selection='f_classif', berm=None):

        self.berm_str = berm
        self.berm = get_berm(berm)
        self.train_pool_data = None
        self.valid_pool_data = None
        self.test_pool_data = None

        self.features_selection = features_selection

        self.train_after_warmup = train_after_warmup
        self.model = model_name
        self.verbose = 1

        self.balanced_rec_loader = balanced_rec_loader
        self.predict_tests = predict_tests
        self.zinb = zinb
        self.load_tb = load_tb
        self.one_model = one_model
        self.dann_plates = dann_plates
        self.dann_sets = dann_sets
        self.tl = tl
        self.variational = variational
        self.random_recs = random_recs
        self.alpha_warmup = alpha_warmup
        self.tied_weights = tied_weights
        self.inference_inputs = inference_inputs
        self.loss = loss
        self.n_epochs = n_epochs
        self.bout = bout  # bout means binary outputs. Should be 1 if output is blank vs bacteria only
        self.dop = dop
        self.add_noise = add_noise
        self.early_stop = early_stop
        self.early_warmup_stop = early_warmup_stop
        self.warmup = warmup
        self.run_name = run_name
        self.prescaler = prescaler  # preprocess scaler
        self.log2 = log2
        self.logging = logging

        self.combat = combat
        self.path = path

        if dann_sets == 1 and dann_plates == 1:
            print("dann_sets and dann_plates are mutually exclusive. dann_sets will be used")

        # data = minmax_scaler.fit_transform(all_df.values)
        # all_batches = [x - 2 for x in all_batches]
        # self.all_plates = np.array([np.argwhere(self.plates == x)[0][0] for x in self.all_plates])

    def train(self, params):
        epoch = 0
        best_loss = 1000
        best_closs = 1000
        best_acc = 0
        if not self.dann_sets and not self.dann_plates:
            params['gamma'] = 0
        if not self.variational:
            params['beta'] = 0
        if not self.zinb:
            params['zeta'] = 0
        optimizer_type = 'adam'
        warmup_counter = 0
        warmup = True
        print(params)
        # params['thres'] = 0.99
        smooth = params['smoothing']
        layer1 = params['layer1']
        layer2 = params['layer2']
        ncols = params['ncols']
        scale = params['scaler']
        dropout = params['dropout']
        margin = params['margin']

        if ncols > self.all_df.shape[1]:
            ncols = self.all_df.shape[1]

        n_cats = len(set(self.all_labels))

        gamma = params['gamma']
        beta = params['beta']
        zeta = params['zeta']
        thres = params['thres']
        wd = params['wd']

        nu = params['nu']
        lr = params['lr']

        # If thres > 0, features that are 0 for a proportion of samples smaller than thres are removed
        all_data, train_data, valid_data, test_data = \
            self.keep_good_features(thres, self.all_df, self.train_data, self.valid_data, self.test_data)

        # Trsnform the data with the chosen scaler
        all_data, train_data, valid_data, test_data = \
            self.scale_data(scale, ncols, all_data, train_data, valid_data, test_data)

        if self.berm is not None:
            df = pd.DataFrame(all_data)
            # df[df.isna()] = 0
            all_data = self.berm(df, self.all_plates)
            train_data = all_data[:train_data.shape[0]]
            valid_data = all_data[train_data.shape[0]:train_data.shape[0]+valid_data.shape[0]]
            test_data = all_data[train_data.shape[0]+valid_data.shape[0]:]

        # Gets all the pytorch dataloaders to train the models
        all_loader, train_loader, train_loader2, valid_loader, test_loader, valid_loader2, test_loader2 = \
            self.get_loaders(all_data, train_data, valid_data, test_data, None, None)

        if self.dann_plates:
            ae = AutoEncoder(all_data.shape[1],
                             n_batches=len(set(self.all_plates)),
                             nb_classes=len(set(self.all_labels)),
                             layer1=layer1, layer2=layer2, dropout=dropout,
                             variational=self.variational, conditional=False, zinb=self.zinb,
                             add_noise=False, tied_weights=self.tied_weights).to(device)
        else:
            ae = AutoEncoder(all_data.shape[1],
                             n_batches=3,
                             nb_classes=len(set(self.all_labels)),
                             layer1=layer1, layer2=layer2, dropout=dropout,
                             variational=self.variational, conditional=False, zinb=self.zinb,
                             add_noise=False, tied_weights=self.tied_weights).to(device)
        best_ae = copy.deepcopy(ae)
        classifier = Classifier(layer2, len(set(self.all_labels))).to(device)
        log_path = f'logs/{self.model}/bout{self.bout}/spd{self.spd}/inference{self.inference_inputs}/' \
                   f'combat{self.combat}/berm{self.berm_str}/{self.run_name}/n{self.add_noise}/tw{self.tied_weights}/taw{self.train_after_warmup}/' \
                   f'tl{self.tl}/pseudo{self.predict_tests}/vae{self.variational}/' \
                   f'zinb{self.zinb}/balanced{self.balanced_rec_loader}/dannset{self.dann_sets}/loss{self.loss}/' \
                   f'scale{self.prescaler}/log{self.log2}/' \
                   f'{scale}/{optimizer_type}/ncols{ncols}/thres{thres}/layers_{layer1}-{layer2}/' \
                   f'd{dropout}/gamma{gamma}/beta{beta}/zeta{zeta}/nu{nu}/smooth{smooth}/lr{lr}/wd{wd}/'

        print(f'See results using: tensorboard --logdir={log_path} --port=6006')

        hparams_filepath = log_path + '/hp'
        os.makedirs(hparams_filepath, exist_ok=True)

        tb_logging = TensorboardLoggingAE(hparams_filepath, params, tw=self.tied_weights, tl=self.tl,
                                          variational=self.variational, zinb=self.zinb,
                                          dann_plates=self.dann_plates, dann_sets=self.dann_sets,
                                          pseudo=self.predict_tests, train_after_warmup=self.train_after_warmup,
                                          berm=self.berm_str)

        event_acc = EventAccumulator(hparams_filepath)
        event_acc.Reload()
        # This verifies if the hparams have already been tested. If they were,
        # the best classification loss is retrieved and we go to the next trial
        if len(event_acc.Tags()['tensors']) > 2 and self.load_tb:
            best_closs = get_best_values_from_tb(event_acc)
        else:
            logger_cm = SummaryWriter(
                f'{log_path}/cm'
            )
            logger = SummaryWriter(
                f'{log_path}/traces'
            )

            sceloss, celoss, mseloss, triplet_loss = self.get_losses(scale, smooth, margin)

            optimizer_ae = get_optimizer(ae, lr, wd, optimizer_type)

            self.log_input_ordination(logger, train_data, valid_data, test_data, epoch)
            values, best_values, best_lists, best_traces = get_empty_dicts()

            early_stop_counter = 0
            best_vals = values
            for epoch in range(epoch, self.n_epochs):
                if early_stop_counter == self.early_stop:
                    if self.verbose > 0:
                        print('EARLY STOPPING.', epoch)
                    break
                lists, traces = get_empty_traces()
                ae.train()
                # classifier.train()

                # with balanced_red_loader, the number of samples to train the autoencoder is balanced
                # between the train set (included the valid data) and the test set
                if self.balanced_rec_loader:
                    iterator = enumerate(zip(train_loader2, cycle(valid_loader2), cycle(test_loader2)))
                else:
                    iterator = enumerate(zip(all_loader, all_loader, all_loader))

                if warmup or self.train_after_warmup:
                    for i, (train_batch, valid_batch, test_batch) in iterator:
                        optimizer_ae.zero_grad()
                        data, labels, domain, to_rec, not_to_rec, concs = train_batch
                        data[torch.isnan(data)] = 0
                        data = data.to(device).float()
                        to_rec = to_rec.to(device).float()
                        enc, rec, zinb_loss, kld = ae(data, None, 1, sampling=True)
                        reverse = ReverseLayerF.apply(enc, 1)
                        domain_preds = ae.dann_discriminator(reverse)

                        if self.dann_sets:
                            domain = torch.zeros(domain_preds.shape[0], 3).float().to(device)
                            domain[:, 0] = 1
                            dloss = celoss(domain_preds, domain)
                            domain = domain.argmax(1)
                        elif self.dann_plates:
                            domain = domain.to(device).long().to(device)
                            dloss = celoss(domain_preds, domain)
                        else:
                            dloss = torch.zeros(1)[0].float().to(device)


                        # rec_loss = triplet_loss(rec, to_rec, not_to_rec)
                        if self.tl and self.balanced_rec_loader and not warmup:
                            not_to_rec = not_to_rec.to(device).float()
                            rec_loss = triplet_loss(rec, to_rec, not_to_rec)
                        else:
                            if scale == 'binarize':
                                rec = torch.sigmoid(rec)
                            rec_loss = mseloss(rec, to_rec)
                        traces['losses'] += [rec_loss.item()]
                        traces['dlosses'] += [dloss.item()]
                        traces['dacc'] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                                    zip(domain_preds.detach().cpu().numpy().argmax(1),
                                                        domain.detach().cpu().numpy())])]

                        (rec_loss + gamma * dloss + beta * kld.mean() + zeta * zinb_loss).backward()
                        optimizer_ae.step()
                        if not self.balanced_rec_loader:
                            break

                        optimizer_ae.zero_grad()
                        data, labels, domain, to_rec, not_to_rec, concs = valid_batch
                        data[torch.isnan(data)] = 0
                        data = data.to(device).float()
                        to_rec = to_rec.to(device).float()
                        enc, rec, zinb_loss, kld = ae(data, None, 1, sampling=True)
                        reverse = ReverseLayerF.apply(enc, 1)

                        domain_preds = ae.dann_discriminator(reverse)

                        if self.dann_sets:
                            domain = torch.zeros(domain_preds.shape[0], 3).float().to(device)
                            domain[:, 1] = 1
                            dloss = celoss(domain_preds, domain)
                            domain = domain.argmax(1)
                        elif self.dann_plates:
                            domain = domain.to(device).long().to(device)
                            dloss = celoss(domain_preds, domain)
                        else:
                            dloss = torch.zeros(1)[0].float().to(device)
                        # rec_loss = ((rec - to_rec) ** 2).sum(axis=0).mean()
                        if scale == 'binarize':
                            rec = torch.sigmoid(rec)
                        rec_loss = mseloss(rec, to_rec)

                        traces['losses'][-1] += rec_loss.item()
                        traces['dlosses'][-1] += dloss.item()
                        traces['dacc'][-1] += np.mean([0 if pred != dom else 1 for pred, dom in
                                                        zip(domain_preds.detach().cpu().numpy().argmax(1),
                                                            domain.detach().cpu().numpy())])

                        (rec_loss + gamma * dloss + beta * kld.mean() + zeta * zinb_loss).backward()
                        optimizer_ae.step()

                        optimizer_ae.zero_grad()
                        data, labels, domain, to_rec, not_to_rec, concs = test_batch
                        data[torch.isnan(data)] = 0
                        data = data.to(device).float()
                        to_rec = to_rec.to(device).float()
                        enc, rec, zinb_loss, kld = ae(data, None, 1, sampling=True)
                        reverse = ReverseLayerF.apply(enc, 1)

                        domain_preds = ae.dann_discriminator(reverse)

                        if self.dann_sets:
                            domain = torch.zeros(domain_preds.shape[0], 3).float().to(device)
                            domain[:, 2] = 1
                            dloss = celoss(domain_preds, domain)
                            domain = domain.argmax(1)
                        elif self.dann_plates:
                            domain = domain.to(device).long().to(device)
                            dloss = celoss(domain_preds, domain)
                        else:
                            dloss = torch.zeros(1)[0].float().to(device)
                        # rec_loss = ((rec - to_rec) ** 2).sum(axis=0).mean()
                        if scale == 'binarize':
                            rec = torch.sigmoid(rec)
                        rec_loss = mseloss(rec, to_rec)

                        traces['losses'][-1] += rec_loss.item()
                        traces['dlosses'][-1] += dloss.item()
                        traces['dacc'][-1] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                                        zip(domain_preds.detach().cpu().numpy().argmax(1),
                                                            domain.detach().cpu().numpy())])]

                        (rec_loss + gamma * dloss + beta * kld.mean() + zeta * zinb_loss).backward()
                        optimizer_ae.step()
                        traces['losses'][-1] /= 3
                        traces['dlosses'][-1] /= 3
                        traces['dacc'][-1] /= 3

                if np.mean(traces['losses']) < best_loss:
                    "Every counters go to 0 when a better reconstruction loss is reached"
                    print(
                        f"Best Loss Epoch {epoch}, Losses: {np.mean(traces['losses'])}, "
                        f"Domain Losses: {np.mean(traces['dlosses'])}, "
                        f"Domain Accuracy: {np.mean(traces['dacc'])}")
                    warmup_counter = 0
                    early_stop_counter = 0
                    best_loss = np.mean(traces['losses'])

                elif warmup_counter == self.early_warmup_stop and warmup:  # or warmup_counter == 100:
                    # When the warnup counter gets to
                    print(f"\n\nWARMUP FINISHED. {epoch}\n\n")
                    warmup = False
                if epoch < self.warmup and warmup:  # and np.mean(traces['losses']) >= best_loss:
                    warmup_counter += 1
                    get_best_values(traces, ae_only=True)
                    # tb_logging.logging(best_values)
                    # if best_loss > 1.0
                    continue
                ae.train()
                if not self.train_after_warmup:
                    for param in ae.dec.parameters():
                        param.requires_grad = False
                    for param in ae.enc.parameters():
                        param.requires_grad = False
                    for param in ae.classifier.parameters():
                        param.requires_grad = True

                for i, batch in enumerate(train_loader):
                    optimizer_ae.zero_grad()
                    data, labels, domain, to_rec, not_to_rec, concs = batch
                    data[torch.isnan(data)] = 0
                    data = data.to(device).float()
                    to_rec = to_rec.to(device).float()
                    not_to_rec = not_to_rec.to(device).float()
                    enc, rec, _, kld = ae(data, None, 1, sampling=True)
                    preds = ae.classifier(enc)
                    domain_preds = ae.dann_discriminator(enc)

                    classif_loss = sceloss(preds, to_categorical(labels.long(), n_cats).to(device).float())
                    if self.tl and not self.predict_tests:
                        rec_loss = triplet_loss(rec, to_rec, not_to_rec)
                    else:
                        rec_loss = torch.zeros(1).to(device)[0]

                    # classif_loss = nllloss(preds, labels.to(device))
                    lists['train']['set'] += [np.array(['train' for _ in range(len(domain))])]
                    lists['train']['domains'] += [domain.detach().cpu().numpy()]
                    lists['train']['domain_preds'] += [domain_preds.detach().cpu().numpy()]
                    lists['train']['preds'] += [preds.detach().cpu().numpy()]
                    lists['train']['classes'] += [labels.detach().cpu().numpy()]
                    lists['train']['concs']['l'] += [concs['lows']]
                    lists['train']['concs']['h'] += [concs['highs']]
                    lists['train']['concs']['v'] += [concs['vhighs']]
                    lists['train']['encoded_values'] += [enc.view(enc.shape[0], -1).detach().cpu().numpy()]
                    lists['train']['inputs'] += [data.view(rec.shape[0], -1).detach().cpu().numpy()]
                    lists['train']['rec_values'] += [rec.view(rec.shape[0], -1).detach().cpu().numpy()]
                    lists['train']['labels'] += [np.array(
                        [self.unique_labels[x] for x in labels.detach().cpu().numpy()])]

                    traces['train']['acc'] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                                        zip(preds.detach().cpu().numpy().argmax(1),
                                                            labels.detach().cpu().numpy())])]

                    if sum(concs['lows']) > -1:
                        traces['train']['acc_l'] += [np.mean([0 if pred != dom else 1 for pred, dom, low in
                                                              zip(preds.detach().cpu().numpy().argmax(1),
                                                                  labels.detach().cpu().numpy(),
                                                                  concs['lows']) if low > -1])]
                    if sum(concs['highs']) > -1:
                        traces['train']['acc_h'] += [np.mean([0 if pred != dom else 1 for pred, dom, high in
                                                              zip(preds.detach().cpu().numpy().argmax(1),
                                                                  labels.detach().cpu().numpy(),
                                                                  concs['highs']) if high > -1])]
                    if sum(concs['vhighs']) > -1:
                        traces['train']['acc_v'] += [np.mean([0 if pred != dom else 1 for pred, dom, vhigh in
                                                              zip(preds.detach().cpu().numpy().argmax(1),
                                                                  labels.detach().cpu().numpy(),
                                                                  concs['vhighs']) if vhigh > -1])]
                    # classif_loss.backward()
                    # (classif_loss + gamma * domain_loss).backward()
                    total_loss = nu * classif_loss
                    if self.train_after_warmup:
                        total_loss += rec_loss
                    total_loss.backward()
                    traces['train']['closs'] += [classif_loss.item()]
                    # optimizer_classif.step()
                    optimizer_ae.step()

                if torch.isnan(classif_loss):
                    break
                ae.eval()
                # classifier.eval()

                for i, batch in enumerate(valid_loader):
                    # optimizer_ae.zero_grad()
                    data, labels, domain, to_rec, not_to_rec, concs = batch
                    data[torch.isnan(data)] = 0
                    data = data.to(device).float()
                    # to_rec = to_rec.to(device).float()
                    enc, rec, _, kld = ae(data, None, 1, sampling=False)
                    # if self.one_model:
                    preds = ae.classifier(enc)
                    domain_preds = ae.dann_discriminator(enc)

                    lists['valid']['set'] += [np.array(['valid' for _ in range(len(domain))])]
                    lists['valid']['domains'] += [domain.detach().cpu().numpy()]
                    lists['valid']['domain_preds'] += [domain_preds.detach().cpu().numpy()]
                    lists['valid']['encoded_values'] += [enc.view(enc.shape[0], -1).detach().cpu().numpy()]
                    lists['valid']['rec_values'] += [rec.view(rec.shape[0], -1).detach().cpu().numpy()]
                    lists['valid']['inputs'] += [data.view(rec.shape[0], -1).detach().cpu().numpy()]
                    lists['valid']['labels'] += [np.array(
                        [self.unique_labels[x] for x in labels.detach().cpu().numpy()])]
                    lists['valid']['classes'] += [labels.detach().cpu().numpy()]
                    lists['valid']['preds'] += [preds.detach().cpu().numpy()]
                    lists['valid']['concs']['l'] += [concs['lows']]
                    lists['valid']['concs']['h'] += [concs['highs']]
                    lists['valid']['concs']['v'] += [concs['vhighs']]

                    # kld_list += [kld.mean().item()]
                    # losses_list += [rec_loss.item()]
                    # domain_loss = 0
                    traces['valid']['acc'] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                                        zip(preds.detach().cpu().numpy().argmax(1),
                                                            labels.detach().cpu().numpy())])]

                    if sum(concs['lows']) > -1:
                        traces['valid']['acc_l'] += [np.mean([0 if pred != dom else 1 for pred, dom, low in
                                                              zip(preds.detach().cpu().numpy().argmax(1),
                                                                  labels.detach().cpu().numpy(),
                                                                  concs['lows']) if low > -1])]
                    if sum(concs['highs']) > -1:
                        traces['valid']['acc_h'] += [np.mean([0 if pred != dom else 1 for pred, dom, high in
                                                              zip(preds.detach().cpu().numpy().argmax(1),
                                                                  labels.detach().cpu().numpy(),
                                                                  concs['highs']) if high > -1])]
                    if sum(concs['vhighs']) > -1:
                        traces['valid']['acc_v'] += [np.mean([0 if pred != dom else 1 for pred, dom, vhigh in
                                                              zip(preds.detach().cpu().numpy().argmax(1),
                                                                  labels.detach().cpu().numpy(),
                                                                  concs['vhighs']) if vhigh > -1])]

                    # domain_loss = celoss(domain_preds, to_categorical(domain.long(), n_domains).to(device).float())
                    # classif_loss = nllloss(preds, labels.to(device))
                    classif_loss = celoss(preds, to_categorical(labels.long(), n_cats).to(device).float())
                    # (rec_loss + gamma * domain_loss).backward()
                    # (gamma * domain_loss).backward()
                    traces['valid']['closs'] += [classif_loss.item()]
                    # domain_losses += [domain_loss.item()]
                    # del data, batch, rec, enc, domain, domain_loss  # , regularization_loss, param

                for i, batch in enumerate(test_loader):

                    data, labels, domain, to_rec, not_to_rec, concs = batch
                    data[torch.isnan(data)] = 0
                    data = data.to(device).float()
                    # to_rec = to_rec.to(device).float()
                    enc, rec, _, kld = ae(data, None, 1, sampling=False)
                    # if self.one_model:
                    preds = ae.classifier(enc)
                    domain_preds = ae.dann_discriminator(enc)
                    lists['test']['set'] += [np.array(['test' for _ in range(len(domain))])]
                    lists['test']['domains'] += [domain.detach().cpu().numpy()]
                    lists['test']['domain_preds'] += [domain_preds.detach().cpu().numpy()]
                    lists['test']['encoded_values'] += [enc.view(enc.shape[0], -1).detach().cpu().numpy()]
                    lists['test']['rec_values'] += [rec.view(rec.shape[0], -1).detach().cpu().numpy()]
                    lists['test']['inputs'] += [data.view(rec.shape[0], -1).detach().cpu().numpy()]
                    lists['test']['labels'] += [np.array(
                        [self.unique_labels[x] for x in labels.detach().cpu().numpy()])]
                    lists['test']['preds'] += [preds.detach().cpu().numpy()]
                    lists['test']['classes'] += [labels.detach().cpu().numpy()]
                    lists['test']['concs']['l'] += [concs['lows']]
                    lists['test']['concs']['h'] += [concs['highs']]
                    lists['test']['concs']['v'] += [concs['vhighs']]

                    traces['test']['acc'] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                                       zip(preds.detach().cpu().numpy().argmax(1),
                                                           labels.detach().cpu().numpy())])]

                    tmp = []
                    if sum(concs['lows']) > -1:
                        traces['test']['acc_l'] += [np.mean([0 if pred != dom else 1 for pred, dom, low in
                                                             zip(preds.detach().cpu().numpy().argmax(1),
                                                                 labels.detach().cpu().numpy(),
                                                                 concs['lows']) if low > -1])]
                        tmp += [traces['test']['acc_l'][-1]]
                    if sum(concs['highs']) > -1:
                        traces['test']['acc_h'] += [np.mean([0 if pred != dom else 1 for pred, dom, high in
                                                             zip(preds.detach().cpu().numpy().argmax(1),
                                                                 labels.detach().cpu().numpy(),
                                                                 concs['highs']) if high > -1])]
                        tmp += [traces['test']['acc_h'][-1]]
                    if sum(concs['vhighs']) > -1:
                        traces['test']['acc_v'] += [np.mean([0 if pred != dom else 1 for pred, dom, vhigh in
                                                             zip(preds.detach().cpu().numpy().argmax(1),
                                                                 labels.detach().cpu().numpy(),
                                                                 concs['vhighs']) if vhigh > -1])]
                        tmp += [traces['test']['acc_v'][-1]]
                    # domain_loss = celoss(domain_preds, to_categorical(domain.long(), n_domains).to(device).float())
                    classif_loss = celoss(preds, to_categorical(labels.long(), n_cats).to(device).float())
                    # classif_loss = nllloss(preds, labels.to(device))
                    # (rec_loss + gamma * domain_loss).backward()
                    traces['test']['closs'] += [classif_loss.item()]
                    # domain_losses += [domain_loss.item()]
                    # del data, batch, rec, enc, domain, domain_loss  # , regularization_loss, param

                # values['train']['ld'] += [torch.mean(kld).item()]
                for group in lists.keys():
                    preds, classes = np.concatenate(lists[group]['preds']).argmax(1), np.concatenate(
                        lists[group]['classes'])
                    traces[group]['mcc'] = MCC(preds, classes)
                    inds = torch.concat(lists[group]['concs']['l']).detach().cpu().numpy()
                    inds = np.array([i for i, x in enumerate(inds) if x > -1])
                    traces[group]['mcc_l'] = MCC(preds[inds], classes[inds])
                    inds = torch.concat(lists[group]['concs']['h']).detach().cpu().numpy()
                    inds = np.array([i for i, x in enumerate(inds) if x > -1])
                    traces[group]['mcc_h'] = MCC(preds[inds], classes[inds])
                    inds = torch.concat(lists[group]['concs']['v']).detach().cpu().numpy()
                    inds = np.array([i for i, x in enumerate(inds) if x > -1])
                    traces[group]['mcc_v'] = MCC(preds[inds], classes[inds])

                values = log_traces(traces, values)
                try:
                    add_to_logger(values, logger, epoch)
                except:
                    pass

                if values['valid']['acc'][-1] > best_acc:
                    print(f"Best Classification Acc Epoch {epoch}, "
                          f"Acc: {values['test']['acc'][-1]}, "
                          f"Lows: {values['test']['acc_l'][-1]}, "
                          f"Highs: {values['test']['acc_h'][-1]}, "
                          f"VHighs: {values['test']['acc_v'][-1]}"
                          f"Classification train loss: {values['train']['closs'][-1]},"
                          f" valid loss: {values['valid']['closs'][-1]},"
                          f" test loss: {values['test']['closs'][-1]}")
                    best_acc = values['valid']['acc'][-1]
                    early_stop_counter = 0

                if values['valid']['closs'][-1] < best_closs:
                    if self.logging:
                        self.log_stuff(logger_cm, logger, lists, values, traces, ae, epoch)

                    print(f"Best Classification Loss Epoch {epoch}, "
                          f"Acc: {values['test']['acc'][-1]}, "
                          f"Lows: {values['test']['acc_l'][-1]}, "
                          f"Highs: {values['test']['acc_h'][-1]}, "
                          f"VHighs: {values['test']['acc_v'][-1]}"
                          f"Classification train loss: {values['train']['closs'][-1]},"
                          f" valid loss: {values['valid']['closs'][-1]},"
                          f" test loss: {values['test']['closs'][-1]}")
                    best_closs = values['valid']['closs'][-1]
                    best_values = get_best_values(values, ae_only=False)
                    # best_acc = values['valid']['acc'][-1]
                    best_vals = values
                    best_ae = copy.deepcopy(ae)
                    best_lists = lists
                    best_traces = traces

                    early_stop_counter = 0
                else:
                    # if epoch > self.warmup:
                    early_stop_counter += 1

                if self.predict_tests:
                    all_loader, train_loader, train_loader2, \
                        valid_loader, test_loader, valid_loader2, \
                        test_loader2 = self.get_loaders(all_data, train_data, valid_data,
                                                        test_data, ae, ae.classifier)
                elif self.tl:
                    all_loader, train_loader, train_loader2, \
                        valid_loader, test_loader, valid_loader2, \
                        test_loader2 = self.get_loaders(all_data, train_data, valid_data,
                                                        test_data, ae, classifier)

            try:
                self.log_stuff(logger_cm, logger, best_lists, best_vals, best_traces, best_ae, epoch)
            except:
                pass
            tb_logging.logging(best_values)

        return best_closs

    def get_data(self, classif):
        # TODO Solve preprocess so this is not necessary
        try:
            train_data = pd.read_csv(
                f"{self.path}/matrices/mz{self.mz_bin}/rt{self.rt_bin}/{self.spd}spd/"
                f"combat{self.combat}/stride{self.stride}/{self.prescaler}/"
                f"log{self.log2}/{self.features_selection}/train/{self.run_name}/"
                f"BACT_train_inputs_gt0.0_{self.run_name}.csv"
            )
        except:
            train_data = pd.read_csv(
                f"{self.path}/matrices/mz{self.mz_bin}/rt{self.rt_bin}/{self.spd}spd/"
                f"combat{self.combat}/stride{self.stride}/{self.prescaler}/"
                f"log{self.log2}/{self.features_selection}/train/{self.run_name}/"
                f"BACT_train_inputs_{self.run_name}.csv"
            )
        try:
            self.train_pool_data = pd.read_csv(
                f"{self.path}/matrices/mz{self.mz_bin}/rt{self.rt_bin}/{self.spd}spd/"
                f"combat{self.combat}/stride{self.stride}/{self.prescaler}/"
                f"log{self.log2}/{self.features_selection}/train/{self.run_name}/"
                f"BACT_pool_inputs_gt0.0_{self.run_name}.csv"
            )
        except:
            self.train_pool_data = pd.read_csv(
                f"{self.path}/matrices/mz{self.mz_bin}/rt{self.rt_bin}/{self.spd}spd/"
                f"combat{self.combat}/stride{self.stride}/{self.prescaler}/"
                f"log{self.log2}/{self.features_selection}/train/{self.run_name}/"
                f"BACT_pool_inputs_{self.run_name}.csv"
            )

        valid_data = pd.read_csv(
            f"{self.path}/matrices/mz{self.mz_bin}/rt{self.rt_bin}/{self.spd}spd/"
            f"combat{self.combat}/stride{self.stride}/{self.prescaler}/"
            f"log{self.log2}/{self.features_selection}/valid/{self.run_name}/"
            f"BACT_valid_inputs_{self.run_name}.csv"
        )
        self.valid_pool_data = pd.read_csv(
            f"{self.path}/matrices/mz{self.mz_bin}/rt{self.rt_bin}/{self.spd}spd/"
            f"combat{self.combat}/stride{self.stride}/{self.prescaler}/"
            f"log{self.log2}/{self.features_selection}/valid/{self.run_name}/"
            f"BACT_valid_pool_inputs_{self.run_name}.csv"
        )
        test_data = pd.read_csv(
            f"{self.path}/matrices/mz{self.mz_bin}/rt{self.rt_bin}/{self.spd}spd/"
            f"combat{self.combat}/stride{self.stride}/{self.prescaler}/"
            f"log{self.log2}/{self.features_selection}/test/{self.run_name}/"
            f"BACT_inference_inputs_{self.run_name}.csv"
        )

        self.test_pool_data = pd.read_csv(
            f"{self.path}/matrices/mz{self.mz_bin}/rt{self.rt_bin}/{self.spd}spd/"
            f"combat{self.combat}/stride{self.stride}/{self.prescaler}/"
            f"log{self.log2}/{self.features_selection}/test/{self.run_name}/"
            f"BACT_inference_pool_inputs_{self.run_name}.csv"
        )

        self.train_names = train_data['ID']
        self.train_labels = np.array([d.split('_')[1].split('-')[0] for d in self.train_names])
        self.train_batches = np.array([int(d.split('_')[0]) for d in self.train_names])

        self.valid_names = valid_data['ID']
        self.valid_labels = np.array([d.split('_')[1].split('-')[0] for d in self.valid_names])
        self.valid_batches = np.array([int(d.split('_')[0]) for d in self.valid_names])

        self.test_names = test_data['ID']
        self.test_labels = np.array([d.split('_')[1].split('-')[0] for d in self.test_names])
        self.test_batches = np.array([int(d.split('_')[0]) for d in self.test_names])

        # Drops the ID column
        self.train_data = train_data.iloc[:, 1:]
        self.valid_data = valid_data.iloc[:, 1:]
        self.test_data = test_data.iloc[:, 1:]

        # TODO Find out why valid and test have 1 column less; all the rest is fine, columns are the same
        if self.train_data.shape[1] > self.valid_data.shape[1]:
            self.train_data = self.train_data.iloc[:, :-1]

        # train_columns = train_data.columns
        self.unique_labels = get_unique_labels(self.train_labels)
        self.train_cats = np.array([np.where(x == self.unique_labels)[0][0] for i, x in enumerate(self.train_labels)])
        self.train_highs = np.array(
            [i if 'h' in x.split('_')[2] or 'blk' in x else -1 for i, x in enumerate(self.train_names)])
        self.train_vhighs = np.array(
            [i if 'v' in x.split('_')[2] or 'blk' in x else -1 for i, x in enumerate(self.train_names)])
        self.train_lows = np.array(
            [i if 'l' in x.split('_')[2] or 'blk' in x else -1 for i, x in enumerate(self.train_names)])

        self.valid_cats = np.array([np.where(x == self.unique_labels)[0][0] for i, x in enumerate(self.valid_labels)])
        self.valid_highs = np.array(
            [i if 'h' in x.split('_')[2] or 'blk' in x else -1 for i, x in enumerate(self.valid_names)])
        self.valid_vhighs = np.array(
            [i if 'v' in x.split('_')[2] or 'blk' in x else -1 for i, x in enumerate(self.valid_names)])
        self.valid_lows = np.array(
            [i if 'l' in x.split('_')[2] or 'blk' in x else -1 for i, x in enumerate(self.valid_names)])

        self.test_cats = np.array([np.where(x == self.unique_labels)[0][0] for i, x in enumerate(self.test_labels)])
        self.test_highs = np.array(
            [i if 'h' in x.split('_')[2] or 'blk' in x else -1 for i, x in enumerate(self.test_names)])
        self.test_vhighs = np.array(
            [i if 'v' in x.split('_')[2] or 'blk' in x else -1 for i, x in enumerate(self.test_names)])
        self.test_lows = np.array(
            [i if 'l' in x.split('_')[2] or 'blk' in x else -1 for i, x in enumerate(self.test_names)])

        train_batches_infos = pd.read_csv(f'{self.path}/RD150_SamplePrep_Juillet_Samples.csv', index_col=0)
        train_batches_infos.index = [x.lower() for x in train_batches_infos.index]
        plates = pd.DataFrame(train_batches_infos['Plate'], index=train_batches_infos.index, columns=['Plate'])
        plates = pd.concat((plates, pd.DataFrame(1, index=['blk'], columns=['Plate'])))
        plates = pd.concat((plates, pd.DataFrame(2, index=['blk_p2'], columns=['Plate'])))
        plates = pd.concat((plates, pd.DataFrame(3, index=['blk_p3'], columns=['Plate'])))
        plates = pd.concat((plates, pd.DataFrame(4, index=['blk_p4'], columns=['Plate'])))
        # Adds 4 because they are not the same plates as in plates
        # test_plates['Plate'] = [5 for _ in test_plates['Plate']]

        train_plates = np.array(
            [plates.loc[x.split('_')[1]].values[0] if 'blk_p' not in x else int(x.split('blk_p')[1].split('-')[0])
             for
             x in self.train_names]
        )

        valid_batches_infos = pd.read_csv(f'{self.path}/RD150_SamplePrep_Novembre_Samples.csv', index_col=0)
        valid_batches_infos.index = [x.lower() for x in valid_batches_infos.index]
        valid_plates = pd.DataFrame(valid_batches_infos['Plate'], index=valid_batches_infos.index, columns=['Plate'])
        valid_plates = pd.concat((valid_plates, pd.DataFrame(8, index=['blk'], columns=['Plate'])))
        valid_plates = np.array([valid_plates.loc[x.split('_')[1]].values[0] for x in self.valid_names])

        test_batches_infos = pd.read_csv(f'{self.path}/RD159_SamplePreparation_Part.1.csv', index_col=0)
        test_batches_infos.index = [x.lower() for x in test_batches_infos.index]
        test_plates = pd.DataFrame(test_batches_infos['Plate'], index=test_batches_infos.index, columns=['Plate'])
        test_plates = pd.concat((test_plates, pd.DataFrame(11, index=['blk'], columns=['Plate'])))
        test_plates = np.array([test_plates.loc[x.split('_')[1]].values[0] for x in self.test_names])
        test_plates[np.argwhere(test_plates == 11)[:, 0][8:16]] = 12
        # TODO test_data seem to have a different number of columns...

        self.plates = np.array(list(set(np.concatenate((train_plates, valid_plates, test_plates)))))
        self.train_plates = np.array([np.argwhere(self.plates == x)[0][0] for x in train_plates])
        self.valid_plates = np.array([np.argwhere(self.plates == x)[0][0] for x in valid_plates])
        self.test_plates = np.array([np.argwhere(self.plates == x)[0][0] for x in test_plates])

        self.all_df = pd.concat((self.train_data, self.valid_data, self.test_data), 0)
        self.train_samples = np.array(
            [1 for _ in self.train_labels] + [0 for _ in self.valid_labels] + [0 for _ in self.test_labels])
        self.all_labels = np.concatenate((self.train_labels, self.valid_labels, self.test_labels))
        self.all_batches = np.concatenate((self.train_batches, self.valid_batches, self.test_batches))
        self.all_plates = np.concatenate((self.train_plates, self.valid_plates, self.test_plates))
        self.all_cats = np.concatenate((self.train_cats, self.valid_cats, self.test_cats))
        self.all_lows = np.concatenate((self.train_lows, self.valid_lows, self.test_lows))
        self.all_highs = np.concatenate((self.train_highs, self.valid_highs, self.test_highs))
        self.all_vhighs = np.concatenate((self.train_vhighs, self.valid_vhighs, self.test_vhighs))

        if classif != 'all':
            # plate = int()
            inds = np.where(self.train_plates == classif)[0]
            self.train_data = self.train_data.iloc[inds]
            self.train_labels = self.train_labels[inds]
            self.train_plates = self.train_plates[inds]
            self.train_names = self.train_names[inds].to_numpy()
            self.train_batches = self.train_batches[inds]
            self.train_cats = self.train_cats[inds]
            self.unique_labels = np.unique(self.train_labels)

            inds = np.where(self.valid_plates == classif)[0]
            self.valid_data = self.valid_data.iloc[inds]
            self.valid_labels = self.valid_labels[inds]
            self.valid_plates = self.valid_plates[inds]
            self.valid_names = self.valid_names[inds].to_numpy()
            self.valid_batches = self.valid_batches[inds]
            self.valid_cats = self.valid_cats[inds]

            inds = np.array([i for i, x in enumerate(self.test_labels) if x in np.unique(self.train_labels)])
            self.test_data = self.test_data.iloc[inds]
            self.test_labels = self.test_labels[inds]
            self.test_plates = self.test_plates[inds]
            # test_names = test_names[inds].to_numpy()
            # test_batches = test_batches[inds]
            self.test_cats = self.test_cats[inds]

        self.valid_highs = [i if 'h' in x.split('_')[2] or 'blk' in x else -1 for i, x in enumerate(self.valid_names)]
        self.valid_vhighs = [i if 'v' in x.split('_')[2] or 'blk' in x else -1 for i, x in enumerate(self.valid_names)]
        self.valid_lows = [i if 'l' in x.split('_')[2] or 'blk' in x else -1 for i, x in enumerate(self.valid_names)]
        train_lows2 = np.concatenate((self.train_lows, self.valid_lows))
        train_highs2 = np.concatenate((self.train_highs, self.valid_highs))
        train_vhighs2 = np.concatenate((self.train_vhighs, self.valid_vhighs))

        self.concs = {
            'all': {
                'lows': self.all_lows,
                'highs': self.all_highs,
                'vhighs': self.all_vhighs,
            },
            'train': {
                'lows': self.train_lows,
                'highs': self.train_highs,
                'vhighs': self.train_vhighs,
            },
            'train2': {
                'lows': train_lows2,
                'highs': train_highs2,
                'vhighs': train_vhighs2,
            },
            'valid': {
                'lows': self.valid_lows,
                'highs': self.valid_highs,
                'vhighs': self.valid_vhighs,
            },
            'test': {
                'lows': self.test_lows,
                'highs': self.test_highs,
                'vhighs': self.test_vhighs,
            },
        }

    def scale_data(self, scale, ncols, data, train_data, valid_data, test_data):
        if scale == 'binarize':
            data = data.values[:, :ncols]
            train_data = train_data.values[:, :ncols]
            valid_data = valid_data.values[:, :ncols]
            test_data = test_data.values[:, :ncols]
            data[data > 0.] = 1
            train_data[train_data > 0.] = 1
            valid_data[valid_data > 0.] = 1
            test_data[test_data > 0.] = 1
        elif scale == 'robust':
            scaler = RobustScaler()

            data = scaler.fit_transform(self.all_df.values[:, :ncols])
            train_data = scaler.transform(self.train_data.values[:, :ncols])
            valid_data = scaler.transform(self.valid_data.values[:, :ncols])
            test_data = scaler.transform(self.test_data.values[:, :ncols])
        elif scale == 'standard':
            scaler = StandardScaler()

            data = scaler.fit_transform(self.all_df.values[:, :ncols])
            train_data = scaler.transform(self.train_data.values[:, :ncols])
            valid_data = scaler.transform(self.valid_data.values[:, :ncols])
            test_data = scaler.transform(self.test_data.values[:, :ncols])
        elif scale == 'l1':
            scaler = Normalizer(norm='l1')

            data = scaler.fit_transform(self.all_df.values[:, :ncols])
            train_data = scaler.transform(self.train_data.values[:, :ncols])
            valid_data = scaler.transform(self.valid_data.values[:, :ncols])
            test_data = scaler.transform(self.test_data.values[:, :ncols])
        elif scale == 'l2':
            scaler = Normalizer(norm='l2')

            data = scaler.fit_transform(self.all_df.values[:, :ncols])
            train_data = scaler.transform(self.train_data.values[:, :ncols])
            valid_data = scaler.transform(self.valid_data.values[:, :ncols])
            test_data = scaler.transform(self.test_data.values[:, :ncols])
        else:
            data = self.all_df.values[:, :ncols]
            train_data = self.train_data.values[:, :ncols]
            valid_data = self.valid_data.values[:, :ncols]
            test_data = self.test_data.values[:, :ncols]
        if scale != 'none':
            scaler = MinMaxScaler()

            data = scaler.fit_transform(data)
            train_data = scaler.transform(train_data)
            valid_data = scaler.transform(valid_data)
            test_data = scaler.transform(test_data)

        return data, train_data, valid_data, test_data

    def get_loaders(self, all_data, train_data, valid_data, test_data, ae=None, classifier=None):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(all_data.mean(), all_data.std()),
        ])
        train_set = MSDataset3(train_data, True, self.train_cats,
                               [x for x in self.train_plates], self.concs['train'],
                               transform=transform, crop_size=-1, random_recs=self.random_recs,
                               quantize=False, device=device)
        # train_data2 = np.concatenate((train_data, valid_data), 0)
        # train_plates2 = np.concatenate((self.train_plates, self.valid_plates))
        # train_cats2 = np.concatenate((self.train_cats, self.valid_cats))

        train_set2 = MSDataset3(train_data, True, self.train_cats,
                                [x for x in self.train_plates], self.concs['train2'],
                                transform=transform, crop_size=-1, random_recs=self.random_recs,
                                quantize=False, device=device)
        valid_set = MSDataset3(valid_data, True, self.valid_cats,
                               [x for x in self.valid_plates], self.concs['valid'],
                               transform=transform, crop_size=-1, random_recs=False,
                               quantize=False, device=device)
        valid_set2 = MSDataset3(valid_data, True, self.valid_cats,
                               [x for x in self.valid_plates], self.concs['valid'],
                               transform=transform, crop_size=-1, random_recs=self.random_recs,
                               quantize=False, device=device)
        test_set = MSDataset3(test_data, True, self.test_cats, [x for x in self.test_plates],
                              self.concs['test'],
                              transform=transform, crop_size=-1, random_recs=False,
                              quantize=False, device=device)
        test_set2 = MSDataset3(test_data, True, self.test_cats, [x for x in self.test_plates],
                              self.concs['test'],
                              transform=transform, crop_size=-1, random_recs=self.random_recs,
                              quantize=False, device=device)
        train_loader = DataLoader(train_set,
                                  num_workers=0,
                                  shuffle=True,
                                  batch_size=8,
                                  pin_memory=False,
                                  drop_last=True)
        train_loader2 = DataLoader(train_set2,
                                   num_workers=0,
                                   shuffle=True,
                                   batch_size=8,
                                   pin_memory=False,
                                   drop_last=True)

        test_loader = DataLoader(test_set,
                                 num_workers=0,
                                 shuffle=False,
                                 batch_size=1,
                                 pin_memory=False,
                                 drop_last=False)
        valid_loader = DataLoader(valid_set,
                                  num_workers=0,
                                  shuffle=False,
                                  batch_size=1,
                                  pin_memory=False,
                                  drop_last=False)
        test_loader2 = DataLoader(test_set2,
                                  num_workers=0,
                                  shuffle=False,
                                  batch_size=8,
                                  pin_memory=False,
                                  drop_last=True)
        valid_loader2 = DataLoader(valid_set2,
                                   num_workers=0,
                                   shuffle=False,
                                   batch_size=8,
                                   pin_memory=False,
                                   drop_last=True)
        if ae is not None:
            valid_cats = []
            test_cats = []
            ae.eval()
            classifier.eval()
            for i, batch in enumerate(valid_loader):
                # optimizer_ae.zero_grad()
                data, labels, domain, to_rec, not_to_rec, concs = batch
                data[torch.isnan(data)] = 0
                data = data.to(device).float()
                # to_rec = to_rec.to(device).float()
                enc, rec, _, kld = ae(data, None, 1, sampling=False)
                # if self.one_model:
                preds = ae.classifier(enc)
                # else:
                #     preds = classifier(enc)
                domain_preds = ae.dann_discriminator(enc)
                valid_cats += [preds.detach().cpu().numpy().argmax(1)]
            for i, batch in enumerate(test_loader):
                # optimizer_ae.zero_grad()
                data, labels, domain, to_rec, not_to_rec, concs = batch
                data[torch.isnan(data)] = 0
                data = data.to(device).float()
                # to_rec = to_rec.to(device).float()
                enc, rec, _, kld = ae(data, None, 1, sampling=False)
                # if self.one_model:
                preds = ae.classifier(enc)
                domain_preds = ae.dann_discriminator(enc)
                # else:
                #     preds = classifier(enc)
                test_cats += [preds.detach().cpu().numpy().argmax(1)]

            all_cats = np.concatenate(
                (self.train_cats, np.stack(valid_cats).reshape(-1), np.stack(test_cats).reshape(-1)))
            all_set = MSDataset3(all_data, True, all_cats, [x for x in self.all_plates],
                                 self.concs['all'], transform=transform, crop_size=-1,
                                 random_recs=self.random_recs, quantize=False, device=device)

        else:
            all_set = MSDataset3(all_data, True, self.all_cats, [x for x in self.all_plates],
                                 self.concs['all'], transform=transform, crop_size=-1,
                                 random_recs=self.random_recs, quantize=False, device=device)

        all_loader = DataLoader(all_set,
                                num_workers=0,
                                shuffle=True,
                                batch_size=8,
                                pin_memory=False,
                                drop_last=True)

        return all_loader, train_loader, train_loader2, valid_loader, test_loader, valid_loader2, test_loader2

    # TODO this function can be improved to be more readable and much shorter
    def log_stuff(self, logger_images, logger, lists, values, traces, classifier, epoch):
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
                                          batch_score=values['set_batch_metrics'][metric][repres]['set'][0]/len(self.unique_labels),
                                          class_score=values['set_batch_metrics'][metric][repres]['labels'][0]/len(self.plates),
                                      ),
                                      # values['set_batch_metrics'][metric][repres]['set'][0],
                                      epoch)
                else:
                    logger.add_scalar(f'F1_{metric}/set/{repres}',
                                      batch_f1_score(
                                          batch_score=values['set_batch_metrics'][metric][repres]['set'][0]/len(self.unique_labels),
                                          class_score=values['set_batch_metrics'][metric][repres]['labels'][0]/len(self.plates),
                                      ),
                                      # values['set_batch_metrics'][metric][repres]['set'][0],
                                      epoch)

                for group in ['train', 'valid', 'test']:
                    if metric == 'lisi':
                        try:
                            logger.add_scalar(f'F1_{metric}/{group}/{repres}',
                                              batch_f1_score(
                                                  batch_score=values[group][metric][repres]['domains'][0][0]/len(self.plates),
                                                  class_score=values[group][metric][repres]['labels'][0][0]/len(self.unique_labels),
                                              ),
                                              epoch)
                        except:
                            logger.add_scalar(f'F1_{metric}/{group}/{repres}',
                                              batch_f1_score(
                                                  batch_score=values[group][metric][repres]['domains'][0]/len(self.plates),
                                                  class_score=values[group][metric][repres]['labels'][0]/len(self.unique_labels),
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

                # try:
                #     logger.add_scalar(f'{metric}/{group}/{info}', values['set_batch_metrics'][metric]['labels'][0], epoch)
                #     logger.add_scalar(f'{metric}/{group}/{info}', values['set_batch_metrics'][metric]['set'][0], epoch)
                # except:
                #     print('There is nothing in array')

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
                             self.unique_labels, traces)
        # save_roc_curve(classifier,
        #                torch.Tensor(np.concatenate(lists['train']['encoded_values'])).to(device),
        #                np.concatenate(lists['train']['classes']),
        #                self.unique_labels, name='./roc_train', binary=self.bout, epoch=epoch,
        #                acc=values['train']['acc'][-1], logger=logger_images)
        save_roc_curve(classifier,
                       torch.Tensor(np.concatenate(lists['valid']['encoded_values'])).to(device),
                       np.concatenate(lists['valid']['classes']),
                       self.unique_labels, name='./roc_valid', binary=self.bout, epoch=epoch,
                       acc=values['valid']['acc'][-1], logger=logger_images)
        save_roc_curve(classifier,
                       torch.Tensor(np.concatenate(lists['test']['encoded_values'])).to(device),
                       np.concatenate(lists['test']['classes']),
                       self.unique_labels, name='./roc_test', binary=self.bout, epoch=epoch,
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
                                    self.unique_labels, name='./prc_valid', binary=self.bout, epoch=epoch,
                                    acc=values['valid']['acc'][-1], logger=logger_images)
        save_precision_recall_curve(classifier,
                                    torch.Tensor(np.concatenate(lists['test']['encoded_values'])).to(
                                        device),
                                    np.concatenate(lists['test']['classes']),
                                    self.unique_labels, name='./prc_test', binary=self.bout, epoch=epoch,
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

    def keep_good_features(self, thres, data, train_data, valid_data, test_data):
        if thres > 0:
            good_features = np.array(
                [i for i in range(self.train_data.shape[1]) if
                 sum(self.train_data.iloc[:, i] != 0) > int(self.train_data.shape[0] * thres)])
            print(f'{data.shape[1] - len(good_features)} features were < {thres} 0s')

            data = data.iloc[:, good_features]
            train_data = train_data.iloc[:, good_features]
            valid_data = valid_data.iloc[:, good_features]
            test_data = test_data.iloc[:, good_features]
        elif thres < 0 or thres >= 1:
            exit('thres value should be: 0 <= thres < 1 ')

        return data, train_data, valid_data, test_data

    def log_input_ordination(self, logger, train_data, valid_data, test_data, epoch):
        log_CCA({'model': CCA(n_components=2), 'name': 'CCA_inputs_classes'}, logger, train_data, valid_data,
                test_data, self.train_labels, self.valid_labels, self.test_labels, self.unique_labels, epoch)
        log_ORD({'model': PCA(n_components=2), 'name': 'PCA_inputs_classes'}, logger, train_data, valid_data,
                test_data, self.train_labels, self.valid_labels, self.test_labels, self.unique_labels, epoch)
        log_ORD({'model': PCA(n_components=2), 'name': 'PCA_inputs_classes_transductive'}, logger, train_data, valid_data,
                test_data, self.train_labels, self.valid_labels, self.test_labels, self.unique_labels, epoch, transductive=True)
        log_TSNE(logger, 'TSNE_inputs_classes', train_data, valid_data, test_data, self.train_labels, self.valid_labels,
                 self.test_labels, self.unique_labels, epoch)
        log_ORD({'model': UMAP(n_neighbors=5, min_dist=0.3, metric='correlation'), 'name': 'UMAP_inputs_classes'},
                logger, train_data, valid_data, test_data, self.train_labels, self.valid_labels,
                self.test_labels, self.unique_labels, epoch)
        log_ORD({'model': UMAP(n_neighbors=5, min_dist=0.3, metric='correlation'), 'name': 'UMAP_inputs_classes_transductive'},
                logger, train_data, valid_data, test_data, self.train_labels, self.valid_labels,
                self.test_labels, self.unique_labels, epoch, transductive=True)
        unique_plates = np.unique(
            np.concatenate((self.train_plates, self.valid_plates, self.test_plates)))

        log_CCA({'model': CCA(n_components=2), 'name': 'CCA_inputs_plates'}, logger, train_data, valid_data,
                test_data, self.train_plates, self.valid_plates, self.test_plates, unique_plates, epoch)
        log_ORD({'model': PCA(n_components=2), 'name': 'PCA_inputs_plates'}, logger, train_data, valid_data,
                test_data, self.train_plates, self.valid_plates, self.test_plates, unique_plates, epoch)
        log_ORD({'model': PCA(n_components=2), 'name': 'PCA_inputs_plates_transductive'}, logger, train_data, valid_data,
                test_data, self.train_plates, self.valid_plates, self.test_plates, unique_plates, epoch, transductive=True)
        log_TSNE(logger, 'TSNE_inputs_plates', train_data, valid_data, test_data, self.train_plates, self.valid_plates,
                 self.test_plates, unique_plates, epoch)
        log_ORD({'model': UMAP(n_neighbors=5, min_dist=0.3, metric='correlation'), 'name': 'UMAP_inputs_plates'},
                logger, train_data, valid_data, test_data, self.train_plates, self.valid_plates,
                self.test_plates, unique_plates, epoch)
        log_ORD({'model': UMAP(n_neighbors=5, min_dist=0.3, metric='correlation'), 'name': 'UMAP_inputs_plates_transductive'},
                logger, train_data, valid_data, test_data, self.train_plates, self.valid_plates,
                self.test_plates, unique_plates, epoch, transductive=True)

        unique_plates = np.unique(
            np.concatenate((self.train_plates, self.valid_plates, self.test_plates)))

        train_sets = ['train' for _ in self.train_plates]
        valid_sets = ['valid' for _ in self.valid_plates]
        test_sets = ['test' for _ in self.test_plates]
        unique_sets = np.array(['train', 'valid', 'test'])

        # Can't do CCA because train_sets only has a single class
        # log_CCA({'model': CCA(n_components=2), 'name': 'CCA_inputs_sets'}, logger, train_data, valid_data, test_data,
        #         train_sets, valid_sets, test_sets, unique_sets, epoch)
        log_ORD({'model': PCA(n_components=2), 'name': 'PCA_inputs_sets'}, logger, train_data, valid_data,
                test_data, train_sets, valid_sets, test_sets, unique_sets, epoch)
        log_ORD({'model': PCA(n_components=2), 'name': 'PCA_inputs_sets_transductive'}, logger, train_data, valid_data,
                test_data, train_sets, valid_sets, test_sets, unique_sets, epoch, transductive=True)
        log_TSNE(logger, 'TSNE_inputs_sets', train_data, valid_data, test_data, train_sets, valid_sets,
                 test_sets, unique_sets, epoch)
        log_ORD({'model': UMAP(n_neighbors=5, min_dist=0.3, metric='correlation'), 'name': 'UMAP_inputs_sets'},
                logger, train_data, valid_data, test_data, train_sets, valid_sets, test_sets, unique_sets, epoch)
        log_ORD({'model': UMAP(n_neighbors=5, min_dist=0.3, metric='correlation'), 'name': 'UMAP_inputs_sets_transductive'},
                logger, train_data, valid_data, test_data, train_sets, valid_sets, test_sets, unique_sets, epoch, transductive=True)

    def get_losses(self, scale, smooth, margin):
        sceloss = nn.CrossEntropyLoss(label_smoothing=smooth)
        celoss = nn.CrossEntropyLoss()
        if self.loss == 'mse':
            mseloss = nn.MSELoss()
        elif self.loss == 'l1':
            mseloss = nn.L1Loss()
        if scale == "binarize":
            mseloss = nn.BCELoss()
        triplet_loss = nn.TripletMarginLoss(margin, p=2)

        return sceloss, celoss, mseloss, triplet_loss

    def preprocess_params(self, mz_bin=0.2, rt_bin=20, spd=200, stride=0):
        self.mz_bin = mz_bin
        self.rt_bin = rt_bin
        self.spd = spd
        self.stride = stride


def get_berm(berm):
    # berm: batch effect removal method
    if berm == 'combat':
        berm = comBatR
    if berm == 'harmony':
        berm = harmonyR
    if berm == 'none':
        berm = None
    return berm


# TODO The plates of test samples are not correctly identified. Need a standardisation of file naming
# The easiest solution would be to include the number of the plate in the sample name
# {date}_{acquisitionType}_{samplesPerDay}_{bacteriumName}_{(concentration)+ID}_{plateNumber}  concentration is facultative
# Example: 20220627_dia_200spd_eco_v04_p1, 20220627_blk_12_p3

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../../resources')
    parser.add_argument('--batch_removal_method', type=str, default='none')
    parser.add_argument('--inference_inputs', type=int, default=1)
    # parser.add_argument('--threshold', type=float, default=0.99)
    parser.add_argument('--combat', type=int, default=0)
    parser.add_argument('--random_recs', type=int, default=0)
    parser.add_argument('--predict_tests', type=int, default=0)
    parser.add_argument('--tied_weights', type=int, default=1)
    parser.add_argument('--variational', type=int, default=0)
    parser.add_argument('--zinb', type=int, default=0)
    parser.add_argument('--balanced_rec_loader', type=int, default=1)
    parser.add_argument('--dann_sets', type=int, default=1)
    parser.add_argument('--dann_plates', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=100)
    parser.add_argument('--early_warmup_stop', type=int, default=100)
    # TODO find better names to distinguish scale and scaler (scaler is the scaler used in preprocessing,
    #  scale is the transformation that will be applied in this script)
    parser.add_argument('--preprocess_scaler', type=str, default='none')
    parser.add_argument('--scaler', type=str, default='robust')
    parser.add_argument('--log2', type=str, default='inloop')
    parser.add_argument('--plate', type=str, default='eco,sag,efa,kpn,blk,blk_p,pool')
    parser.add_argument('--alpha_warmup', type=int, default=10000)
    parser.add_argument('--triplet_loss', type=int, default=1)
    parser.add_argument('--mz_bin', type=float, default=0.2)
    parser.add_argument('--rt_bin', type=int, default=20)
    parser.add_argument('--spd', type=int, default=200)
    parser.add_argument('--one_model', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--n_epochs', type=int, default=10000)
    parser.add_argument('--feature_selection', type=str, default='mutual_info_classif')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = args.device

    train = Train(path=args.path, run_name=args.plate, prescaler=args.preprocess_scaler, log2=args.log2, combat=args.combat,
                  n_epochs=args.n_epochs, scaler=args.scaler, warmup=args.warmup, tl=args.triplet_loss,
                  alpha_warmup=args.alpha_warmup, random_recs=args.random_recs, one_model=args.one_model,
                  early_stop=args.early_stop, loss='mse', inference_inputs=args.inference_inputs,
                  balanced_rec_loader=args.balanced_rec_loader, train_after_warmup=0,
                  tied_weights=args.tied_weights, variational=args.variational, zinb=args.zinb,
                  predict_tests=args.predict_tests, features_selection=args.feature_selection,
                  dann_sets=args.dann_sets, dann_plates=args.dann_plates, load_tb=True, berm=args.batch_removal_method)
    train.preprocess_params(mz_bin=args.mz_bin, rt_bin=args.rt_bin, spd=args.spd, stride=0)
    train.get_data(classif='all')
    # train.train()

    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "gamma", "type": "range", "bounds": [1e-3, 1e2], "log_scale": True},
            {"name": "zeta", "type": "range", "bounds": [1e-3, 1e2], "log_scale": True},
            {"name": "beta", "type": "range", "bounds": [1e-3, 1e2], "log_scale": True},
            {"name": "nu", "type": "range", "bounds": [1e-4, 1e2], "log_scale": False},
            {"name": "lr", "type": "range", "bounds": [1e-6, 1e-1], "log_scale": True},
            {"name": "wd", "type": "range", "bounds": [1e-8, 1e-5], "log_scale": True},
            {"name": "smoothing", "type": "range", "bounds": [0., 0.2]},
            {"name": "margin", "type": "range", "bounds": [1., 10.]},

            {"name": "thres", "type": "range", "bounds": [0.0, 0.99]},
            {"name": "dropout", "type": "range", "bounds": [0.0, 0.5]},
            {"name": "ncols", "type": "range", "bounds": [10, 10000]},
            {"name": "scaler", "type": "choice", "values": ['robust']},
            {"name": "layer2", "type": "range", "bounds": [32, 1024]},
            {"name": "layer1", "type": "range", "bounds": [32, 1024]},
        ],
        evaluation_function=train.train,
        objective_name='loss',
        minimize=True,
        total_trials=250,
        random_seed=42,

    )

    fig = plt.figure()
    # render(plot_contour(model=model, param_x="learning_rate", param_y="weight_decay", metric_name='Loss'))
    # fig.savefig('test.jpg')
    print('Best Loss:', values[0]['loss'])
    print('Best Parameters:')
    print(json.dumps(best_parameters, indent=4))
