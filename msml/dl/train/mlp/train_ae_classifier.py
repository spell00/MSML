import matplotlib

matplotlib.use('Agg')
CUDA_VISIBLE_DEVICES = ""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import json
import copy
import torch
from itertools import cycle
from torch import nn
import os
from tensorboardX import SummaryWriter
from ax.service.managed_loop import optimize
from sklearn.metrics import matthews_corrcoef as MCC
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from sklearn.model_selection import StratifiedKFold

from msml.utils.batch_effect_removal import remove_batch_effect, get_berm
from msml.dl.models.pytorch.aedann import AutoEncoder, ReverseLayerF
from msml.dl.models.pytorch.utils.loggings import TensorboardLoggingAE, log_stuff, log_input_ordination
from msml.dl.models.pytorch.utils.dataset import get_loaders
from msml.utils.utils import scale_data, get_unique_labels
from msml.dl.models.pytorch.utils.utils import get_optimizer, to_categorical, get_empty_dicts, get_empty_traces, \
    log_traces, get_best_values_from_tb, get_best_values, add_to_logger

import warnings

warnings.filterwarnings("ignore")

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


# TODO this function can be improved to be more readable and much shorter


class Train:

    def __init__(self, args, log_path, fix_thres=-1, load_tb=False):
        """

        Args:
            args: contains multiple arguments passed in the command line
            log_path: Path where the tensorboard logs are saved
            csvs_path: Path to the data (in .csv format)
            fix_thres: If 1 > fix_thres >= 0 then the threshold is fixed to that value.
                       any other value means the threshold won't be fixed and will be
                       learned as an hyperparameter
            load_tb: If True, loads previous runs already saved
        """
        self.args = args
        self.log_path = log_path
        self.fix_thres = fix_thres
        self.load_tb = load_tb

        # berm stands for batch effect removal method
        self.berm = get_berm(args.batch_removal_method)
        self.verbose = 1

        if args.dann_sets == 1 and args.dann_plates == 1:
            self.args.dann_sets = 0
            print("dann_sets and dann_plates are mutually exclusive. dann_plates will be used")
        self.data = None
        self.unique_labels = None
        self.unique_batches = None

    def train(self, params):
        """

        Args:
            params:

        Returns:

        """
        if not self.args.dann_sets and not self.args.dann_plates:
            # gamma = 0 will ensure DANN is not learned
            params['gamma'] = 0
        if not self.args.variational:
            # beta = 0 because useless outside a variational autoencoder
            params['beta'] = 0
        if not self.args.zinb:
            # zeta = 0 because useless outside a zinb autoencoder
            params['zeta'] = 0
        if 1 > self.fix_thres >= 0:
            # fixes the threshold of 0s tolerated for a feature
            params['thres'] = self.fix_thres
        print(params)

        smooth = params['smoothing']
        layer1 = params['layer1']
        layer2 = params['layer2']
        scale = params['scaler']
        dropout = params['dropout']
        margin = params['margin']
        gamma = params['gamma']
        beta = params['beta']
        zeta = params['zeta']
        thres = params['thres']
        wd = params['wd']
        nu = params['nu']
        lr = params['lr']
        ncols = params['ncols']
        if ncols > self.data['inputs']['all'].shape[1]:
            ncols = self.data['inputs']['all'].shape[1]

        epoch = 0
        best_loss = 1000
        best_closs = 1000
        best_acc = 0
        optimizer_type = 'adam'
        warmup_counter = 0
        warmup = True

        # self.log_path is where tensorboard logs are saved
        self.log_path += f'{scale}/berm{args.batch_removal_method}/{optimizer_type}/' \
                         f'ncols{ncols}/thres{thres}/layers_{layer1}-{layer2}/' \
                         f'd{dropout}/gamma{gamma}/beta{beta}/zeta{zeta}/nu{nu}/' \
                         f'smooth{smooth}/lr{lr}/wd{wd}/'
        print(f'See results using: tensorboard --logdir={self.log_path} --port=6006')

        hparams_filepath = self.log_path + '/hp'
        os.makedirs(hparams_filepath, exist_ok=True)

        tb_logging = TensorboardLoggingAE(hparams_filepath, params, tw=self.args.tied_weights,
                                          tl=self.args.triplet_loss,
                                          variational=self.args.variational, zinb=self.args.zinb,
                                          dann_plates=self.args.dann_plates, dann_sets=self.args.dann_sets,
                                          pseudo=self.args.predict_tests,
                                          train_after_warmup=self.args.train_after_warmup,
                                          berm=self.args.batch_removal_method,
                                          subs=list(self.data['subs']['train'].keys()))

        # event_acc is used to verify if the hparams have already been tested. If they were,
        # the best classification loss is retrieved and we go to the next trial
        event_acc = EventAccumulator(hparams_filepath)
        event_acc.Reload()
        if len(event_acc.Tags()['tensors']) > 2 and self.load_tb:
            best_closs = get_best_values_from_tb(event_acc)
        else:
            # If thres > 0, features that are 0 for a proportion of samples smaller than thres are removed
            data = self.keep_good_features(thres)

            # Transform the data with the chosen scaler
            data = scale_data(scale, ncols, data)

            if not self.args.dann_sets:
                data = remove_batch_effect(self.berm, data['all'], data['train'], data['valid'], data['test'],
                                           self.data['batches']['all'])
            else:
                all_sets = np.array([0 for _ in data['train']] + [1 for _ in data['valid']] + [2 for _ in data['test']])
                data = remove_batch_effect(self.berm, data['all'], data['train'], data['valid'], data['test'], all_sets)

            # Gets all the pytorch dataloaders to train the models
            loaders = get_loaders(self.data, data, self.args.random_recs, None, None)

            # if using dann_sets, the 3 domains are the train, valid and test sets
            if self.args.dann_sets:
                n_batches = 3
            else:
                n_batches = len(set(self.data['batches']['all']))
            ae = AutoEncoder(data['all'].shape[1],
                             n_batches=n_batches,
                             nb_classes=len(set(self.data['labels']['all'])),
                             layer1=layer1, layer2=layer2, dropout=dropout,
                             variational=self.args.variational, conditional=False, zinb=self.args.zinb,
                             add_noise=0, tied_weights=self.args.tied_weights).to(device)

            best_ae = copy.deepcopy(ae)
            logger_cm = SummaryWriter(f'{self.log_path}/cm')
            logger = SummaryWriter(f'{self.log_path}/traces')

            sceloss, celoss, mseloss, triplet_loss = self.get_losses(scale, smooth, margin)

            optimizer_ae = get_optimizer(ae, lr, wd, optimizer_type)

            log_input_ordination(logger, self.data, data, epoch)
            values, best_values, best_lists, best_traces = get_empty_dicts()

            early_stop_counter = 0
            best_vals = values
            for epoch in range(epoch, self.args.n_epochs):
                if early_stop_counter == self.args.early_stop:
                    if self.verbose > 0:
                        print('EARLY STOPPING.', epoch)
                    break
                lists, traces = get_empty_traces(self.subcategories)
                ae.train()

                # with balanced_red_loader, the number of samples to train the autoencoder is balanced
                # between the train set (included the valid data) and the test set
                if self.args.balanced_rec_loader:
                    iterator = enumerate(zip(loaders['train'], cycle(loaders['valid2']), cycle(loaders['test2'])))
                else:
                    # when train/valid/test are not balanced, then only the first of the three iterators
                    # is used. We are using 3 times the same loader so that there is 3 iterators so that
                    # the following code works
                    iterator = enumerate(zip(loaders['all'], loaders['all'], loaders['all']))

                # If option train_after_warmup=1, then this loop is only for preprocessing
                if warmup or self.args.train_after_warmup:
                    for i, (train_batch, valid_batch, test_batch) in iterator:
                        optimizer_ae.zero_grad()
                        inputs, labels, domain, to_rec, not_to_rec, concs = train_batch
                        inputs[torch.isnan(inputs)] = 0
                        inputs = inputs.to(device).float()
                        to_rec = to_rec.to(device).float()
                        enc, rec, zinb_loss, kld = ae(inputs, None, 1, sampling=True)
                        reverse = ReverseLayerF.apply(enc, 1)
                        domain_preds = ae.dann_discriminator(reverse)

                        dloss, domain = self.get_dloss(celoss, domain, domain_preds, 0)
                        # rec_loss = triplet_loss(rec, to_rec, not_to_rec)
                        if self.args.triplet_loss and self.args.balanced_rec_loader and not warmup:
                            not_to_rec = not_to_rec.to(device).float()
                            rec_loss = triplet_loss(rec, to_rec, not_to_rec)
                        else:
                            if scale == 'binarize':
                                rec = torch.sigmoid(rec)
                            rec_loss = mseloss(rec, to_rec)
                        traces['rec_loss'] += [rec_loss.item()]
                        traces['dom_loss'] += [dloss.item()]
                        traces['dom_acc'] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                                    zip(domain_preds.detach().cpu().numpy().argmax(1),
                                                        domain.detach().cpu().numpy())])]

                        (rec_loss + gamma * dloss + beta * kld.mean() + zeta * zinb_loss).backward()
                        optimizer_ae.step()

                        # Only the first iterator is used if train/valid/test sets are nots balanced
                        if not self.args.balanced_rec_loader:
                            break

                        optimizer_ae.zero_grad()
                        inputs, labels, domain, to_rec, not_to_rec, concs = valid_batch
                        inputs[torch.isnan(inputs)] = 0
                        inputs = inputs.to(device).float()
                        to_rec = to_rec.to(device).float()
                        enc, rec, zinb_loss, kld = ae(inputs, None, 1, sampling=True)
                        reverse = ReverseLayerF.apply(enc, 1)

                        domain_preds = ae.dann_discriminator(reverse)
                        dloss, domain = self.get_dloss(celoss, domain, domain_preds, 1)

                        # rec_loss = ((rec - to_rec) ** 2).sum(axis=0).mean()
                        if scale == 'binarize':
                            rec = torch.sigmoid(rec)
                        rec_loss = mseloss(rec, to_rec)

                        traces['rec_loss'][-1] += rec_loss.item()
                        traces['dom_loss'][-1] += dloss.item()
                        traces['dom_acc'][-1] += np.mean([0 if pred != dom else 1 for pred, dom in
                                                       zip(domain_preds.detach().cpu().numpy().argmax(1),
                                                           domain.detach().cpu().numpy())])

                        (rec_loss + gamma * dloss + beta * kld.mean() + zeta * zinb_loss).backward()
                        optimizer_ae.step()

                        optimizer_ae.zero_grad()
                        inputs, labels, domain, to_rec, not_to_rec, concs = test_batch
                        inputs[torch.isnan(inputs)] = 0
                        inputs = inputs.to(device).float()
                        to_rec = to_rec.to(device).float()
                        enc, rec, zinb_loss, kld = ae(inputs, None, 1, sampling=True)
                        reverse = ReverseLayerF.apply(enc, 1)

                        domain_preds = ae.dann_discriminator(reverse)

                        dloss, domain = self.get_dloss(celoss, domain, domain_preds, 2)
                        rec_loss = mseloss(rec, to_rec)

                        traces['rec_loss'][-1] += rec_loss.item()
                        traces['dom_loss'][-1] += dloss.item()
                        traces['dom_acc'][-1] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                                        zip(domain_preds.detach().cpu().numpy().argmax(1),
                                                            domain.detach().cpu().numpy())])]

                        (rec_loss + gamma * dloss + beta * kld.mean() + zeta * zinb_loss).backward()
                        optimizer_ae.step()

                        # Divided by 3 because the results of train/valid/test have been summed
                        traces['rec_loss'][-1] /= 3
                        traces['dom_loss'][-1] /= 3
                        traces['dom_acc'][-1] /= 3

                if np.mean(traces['rec_loss']) < best_loss:
                    "Every counters go to 0 when a better reconstruction loss is reached"
                    print(
                        f"Best Loss Epoch {epoch}, Losses: {np.mean(traces['rec_loss'])}, "
                        f"Domain Losses: {np.mean(traces['dom_loss'])}, "
                        f"Domain Accuracy: {np.mean(traces['dom_acc'])}")
                    warmup_counter = 0
                    early_stop_counter = 0
                    best_loss = np.mean(traces['rec_loss'])

                elif warmup_counter == self.args.early_warmup_stop and warmup:  # or warmup_counter == 100:
                    # When the warnup counter gets to
                    print(f"\n\nWARMUP FINISHED. {epoch}\n\n")
                    warmup = False
                if epoch < self.args.warmup and warmup:  # and np.mean(traces['rec_loss']) >= best_loss:
                    warmup_counter += 1
                    get_best_values(traces, ae_only=True, subs=self.subcategories)
                    continue
                ae.train()

                # If training of the autoencoder is retricted to the warmup, (train_after_warmup=0),
                # all layers except the classification layers are frozen
                ae = self.freeze_layers(ae)

                closs, lists, traces = self.loop('train', optimizer_ae, ae, sceloss, triplet_loss,
                                                 loaders['train'], lists, traces, nu=nu)

                if torch.isnan(closs):
                    break
                ae.eval()
                closs, lists, traces = self.loop('valid', None, ae, celoss, triplet_loss,
                                                 loaders['valid'], lists, traces, nu=0)
                closs, lists, traces = self.loop('test', None, ae, celoss, triplet_loss,
                                                 loaders['test'], lists, traces, nu=0)

                traces = self.get_mccs(lists, traces, list(self.data['subs']['all'].keys()))
                values = log_traces(traces, values, subsets=self.subcategories)
                try:
                    add_to_logger(values, logger, epoch)
                except:
                    pass

                if values['valid']['acc'][-1] > best_acc:
                    subs_accs = ''
                    for k in list(self.data['subs']['all'].keys()):
                        subs_accs += f"{k}: {values['test'][f'acc_{k}'][-1]}, "
                    print(f"Best Classification Acc Epoch {epoch}, "
                          f"Acc: {values['test']['acc'][-1]}, {subs_accs}"
                          f"Classification train loss: {values['train']['closs'][-1]},"
                          f" valid loss: {values['valid']['closs'][-1]},"
                          f" test loss: {values['test']['closs'][-1]}")

                    best_acc = values['valid']['acc'][-1]
                    early_stop_counter = 0

                if values['valid']['closs'][-1] < best_closs:
                    # if self.logging:
                    #     log_stuff(logger_cm, logger, lists, values, traces, ae,
                    #               self.unique_labels, self.data['batches'], epoch, device=device)
                    subs_accs = ''
                    for k in list(self.data['subs']['train'].keys()):
                        subs_accs += f"{k}: {values['test'][f'acc_{k}'][-1]}, "
                    print(f"Best Classification Loss Epoch {epoch}, "
                          f"Acc: {values['test']['acc'][-1]}, {subs_accs}"
                          f"Classification train loss: {values['train']['closs'][-1]}, "
                          f"valid loss: {values['valid']['closs'][-1]}, "
                          f"test loss: {values['test']['closs'][-1]}")
                    best_closs = values['valid']['closs'][-1]
                    best_values = get_best_values(values, ae_only=False, subs=self.subcategories)
                    # best_acc = values['valid']['acc'][-1]
                    best_vals = values
                    best_ae = copy.deepcopy(ae)
                    best_lists = lists
                    best_traces = traces

                    early_stop_counter = 0
                else:
                    # if epoch > self.warmup:
                    early_stop_counter += 1

                if self.args.predict_tests:
                    loaders = get_loaders(self.data, data, self.args.random_recs, ae, ae.classifier)
            try:
                log_stuff(logger_cm, logger, best_lists, best_vals, best_traces, best_ae,
                          self.unique_labels, self.data['batches'], epoch, device=device)
            except:
                pass
            tb_logging.logging(best_values)

        return best_closs

    def loop(self, group, optimizer_ae, ae, celoss, triplet_loss, loader, lists, traces, nu=1):
        """

        Args:
            group:
            optimizer_ae:
            ae:
            celoss:
            triplet_loss:
            loader:
            lists:
            traces:
            nu:

        Returns:

        """
        n_cats = len(np.unique(self.data['labels']['all']))
        if group == 'train':
            sampling = True
        else:
            sampling = False
        classif_loss = None
        for i, batch in enumerate(loader):
            if group == 'train':
                optimizer_ae.zero_grad()
            data, labels, domain, to_rec, not_to_rec, concs = batch
            data[torch.isnan(data)] = 0
            data = data.to(device).float()
            to_rec = to_rec.to(device).float()
            not_to_rec = not_to_rec.to(device).float()
            enc, rec, _, kld = ae(data, None, 1, sampling=sampling)
            preds = ae.classifier(enc)
            domain_preds = ae.dann_discriminator(enc)

            classif_loss = celoss(preds, to_categorical(labels.long(), n_cats).to(device).float())
            if self.args.triplet_loss and not self.args.predict_tests:
                rec_loss = triplet_loss(rec, to_rec, not_to_rec)
            else:
                rec_loss = torch.zeros(1).to(device)[0]

            lists[group]['set'] += [np.array([group for _ in range(len(domain))])]
            lists[group]['domains'] += [domain.detach().cpu().numpy()]
            lists[group]['domain_preds'] += [domain_preds.detach().cpu().numpy()]
            lists[group]['preds'] += [preds.detach().cpu().numpy()]
            lists[group]['classes'] += [labels.detach().cpu().numpy()]
            lists[group]['encoded_values'] += [enc.view(enc.shape[0], -1).detach().cpu().numpy()]
            lists[group]['inputs'] += [data.view(rec.shape[0], -1).detach().cpu().numpy()]
            lists[group]['rec_values'] += [rec.view(rec.shape[0], -1).detach().cpu().numpy()]
            lists[group]['labels'] += [np.array(
                [self.unique_labels[x] for x in labels.detach().cpu().numpy()])]

            traces[group]['acc'] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                              zip(preds.detach().cpu().numpy().argmax(1),
                                                  labels.detach().cpu().numpy())])]

            for k in list(concs.keys()):
                lists[group]['concs'][k] += [concs[k]]
                if sum(concs[k]) > -1:
                    traces[group][f'acc_{k}'] += [np.mean([0 if pred != dom else 1 for pred, dom, low in
                                                           zip(preds.detach().cpu().numpy().argmax(1),
                                                               labels.detach().cpu().numpy(),
                                                               concs[k]) if low > -1])]
            traces[group]['closs'] += [classif_loss.item()]

            if group == 'train':
                total_loss = nu * classif_loss
                if self.args.train_after_warmup:
                    total_loss += rec_loss
                total_loss.backward()
                optimizer_ae.step()

        return classif_loss, lists, traces

    def get_dloss(self, celoss, domain, domain_preds, set_num=None):
        """
        This function is used to get the domain loss
        Args:
            celoss: PyTorch CrossEntropyLoss instance object
            domain: If not dann_sets, one-hot encoded domain classes []
            domain_preds: Matrix containing the predicted domains []
            set_num: If dann_sets, defines what the domain is.
                     train=0, valid=1, test=2

        Returns:

        """
        if self.args.dann_sets:
            domain = torch.zeros(domain_preds.shape[0], 3).float().to(device)
            domain[:, set_num] = 1
            dloss = celoss(domain_preds, domain)
            domain = domain.argmax(1)
        elif self.args.dann_plates:
            domain = domain.to(device).long().to(device)
            dloss = celoss(domain_preds, domain)
        else:
            dloss = torch.zeros(1)[0].float().to(device)
        return dloss, domain

    def get_data(self, csvs_path):
        """


        Returns: Nothing

        """
        unique_labels = []
        data = {}
        for info in ['subs', 'inputs', 'names', 'labels', 'cats', 'batches']:
            data[info] = {}
            for group in ['all', 'all_pool', 'train', 'train_pool', 'valid', 'valid_pool', 'test', 'test_pool']:
                data[info][group] = np.array([])
        for group in ['train', 'test', 'valid']:
            if group == 'valid' and not self.args.use_valid:
                skf = StratifiedKFold(n_splits=5)
                train_nums = np.arange(0, len(data['labels']['train']))
                train_inds, valid_inds = skf.split(train_nums, data['labels']['train']).__next__()
                data['inputs']['train'], data['inputs']['valid'] = data['inputs']['train'].iloc[train_inds], \
                                                                   data['inputs']['train'].iloc[valid_inds]
                data['labels']['train'], data['labels']['valid'] = data['labels']['train'][train_inds], \
                                                                   data['labels']['train'][valid_inds]
                data['names']['train'], data['names']['valid'] = data['names']['train'].iloc[train_inds], \
                                                                 data['names']['train'].iloc[valid_inds]
                data['batches']['train'], data['batches']['valid'] = data['batches']['train'][train_inds], \
                                                                     data['batches']['train'][valid_inds]
                data['cats']['train'], data['cats']['valid'] = data['cats']['train'][train_inds], data['cats']['train'][
                    valid_inds]
                subcategories = np.unique(
                    ['v' for x in data['names'][group]])
                subcategories = np.array([x for x in subcategories if x != ''])
                data['subs'][group] = {x: np.array([]) for x in subcategories}
                for sub in list(data['subs'][group]):
                    data['subs']['train'][sub], data['subs']['valid'][sub] = data['subs']['train'][sub][train_inds], \
                                                                             data['subs']['train'][sub][valid_inds]

            if group == 'test' and not self.args.use_test:
                skf = StratifiedKFold(n_splits=5)
                train_nums = np.arange(0, len(data['labels']['train']))
                train_inds, test_inds = skf.split(train_nums, data['labels']['train']).__next__()
                data['inputs']['train'], data['inputs']['test'] = data['inputs']['train'].iloc[train_inds], \
                                                                  data['inputs']['train'].iloc[test_inds]
                data['names']['train'], data['names']['test'] = data['names']['train'].iloc[train_inds], \
                                                                data['names']['train'].iloc[test_inds]
                data['labels']['train'], data['labels']['test'] = data['labels']['train'][train_inds], \
                                                                  data['labels']['train'][test_inds]
                data['batches']['train'], data['batches']['test'] = data['batches']['train'][train_inds], \
                                                                    data['batches']['train'][test_inds]
                data['cats']['train'], data['cats']['test'] = \
                    data['cats']['train'][train_inds], data['cats']['train'][test_inds]
                subcategories = np.unique(
                    ['v' for x in data['names'][group]])
                subcategories = np.array([x for x in subcategories if x != ''])
                data['subs'][group] = {x: np.array([]) for x in subcategories}
                for sub in list(data['subs'][group]):
                    data['subs']['train'][sub], data['subs']['test'][sub] = data['subs']['train'][sub][train_inds], \
                                                                            data['subs']['train'][sub][test_inds]

            else:
                data['inputs'][group] = pd.read_csv(
                    f"{csvs_path}/{group}_inputs.csv"
                )
                data['names'][group] = data['inputs'][group]['ID']
                data['labels'][group] = np.array([d.split('_')[1] for d in data['names'][group]])
                unique_labels = get_unique_labels(data['labels'][group])
                # data['batches'][group] = np.array([int(d.split('_')[0]) for d in data['names'][group]])
                try:
                    data['batches'][group] = np.array([int(d.split('_')[2].split('p')[1]) for d in data['names'][group]])
                except:
                    data['batches'][group] = np.array([int(d.split('_')[0]) for d in data['names'][group]])

                # Drops the ID column
                data['inputs'][group] = data['inputs'][group].iloc[:, 1:]
                data['cats'][group] = np.array(
                    [np.where(x == unique_labels)[0][0] for i, x in enumerate(data['labels'][group])])
                import re
                subcategories = np.unique(
                    [re.split('\d+', x.split('_')[3])[0] for x in data['names'][group]])
                subcategories = np.array([x for x in subcategories if x != ''])
                data['subs'][group] = {x: np.array([]) for x in subcategories}
                for sub in subcategories:
                    data['subs'][group][sub] = np.array(
                        [i for i, x in
                         enumerate(data['names'][group])]
                    )

                # TODO this should not be necessary
                if group != 'train':
                    data['inputs'][group] = data['inputs'][group].loc[:, data['inputs']['train'].columns]

        subcategories = np.unique(
            np.concatenate(
                [np.unique(
                    [re.split('\d+', x.split('_')[3])[0] for x in data['names'][group]]) for group in
                    list(data['names'].keys())
                ]
            )
        )
        subcategories = np.array([x for x in subcategories if x != ''])
        self.subcategories = subcategories

        for key in list(data.keys()):
            if key == 'inputs':
                data[key]['all'] = pd.concat((data[key]['train'], data[key]['valid'], data[key]['test']), 0)
            elif key != 'subs':
                data[key]['all'] = np.concatenate((data[key]['train'], data[key]['valid'], data[key]['test']), 0)

        # Add values for sets without a subset
        for s in subcategories:
            for group in ['train', 'valid', 'test']:
                if s not in list(data['subs'][group].keys()):
                    data['subs'][group][s] = np.array([-1 for _ in range(len(data['cats'][group]))])

        data['subs']['all'] = {sub: np.concatenate(
            (data['subs']['train'][sub], data['subs']['valid'][sub], data['subs']['test'][sub]), 0) for sub in
            subcategories}
        unique_batches = np.unique(data['batches']['all'])
        for group in ['train', 'valid', 'test', 'all']:
            data['batches'][group] = np.array([np.argwhere(unique_batches == x)[0][0] for x in data['batches'][group]])

        self.data = data
        self.unique_labels = unique_labels
        self.unique_batches = unique_batches

    def keep_good_features(self, thres):
        """
        All dataframes have a shape of N samples (rows) x M features (columns)
        Args:
            thres: Ratio of 0s tolerated. Features with ratios > thres are removed

        Returns:
        A dictionary of pandas datasets with keys:
            'all': Pandas dataframe containing all data (train, valid and test data),
            'train': Pandas dataframe containing the training data,
            'valid': Pandas dataframe containing the validation data,
            'test: Pandas dataframe containing the test data'
        """
        if thres > 0:
            good_features = np.array(
                [i for i in range(self.data['inputs']['all'].shape[1]) if
                 sum(self.data['inputs']['all'].iloc[:, i] != 0) > int(self.data['inputs']['all'].shape[0] * thres)])
            print(f"{self.data['inputs']['all'].shape[1] - len(good_features)} features were < {thres} 0s")

            data = self.data['inputs']['all'].iloc[:, good_features]
            train_data = self.data['inputs']['train'].iloc[:, good_features]
            valid_data = self.data['inputs']['valid'].iloc[:, good_features]
            test_data = self.data['inputs']['test'].iloc[:, good_features]
        elif thres < 0 or thres >= 1:
            exit('thres value should be: 0 <= thres < 1 ')
        else:
            data = self.data['inputs']['all'].iloc[:]
            train_data = self.data['inputs']['train'].iloc[:]
            valid_data = self.data['inputs']['valid'].iloc[:]
            test_data = self.data['inputs']['test'].iloc[:]

        return {'all': data, 'train': train_data, 'valid': valid_data, 'test': test_data}

    def get_losses(self, scale, smooth, margin):
        """
        Getter for the losses.
        Args:
            scale: Scaler that was used, e.g. normalizer or binarize
            smooth: Parameter for label_smoothing
            margin: Parameter for the TripletMarginLoss

        Returns:
            sceloss: CrossEntropyLoss (with label smoothing)
            celoss: CrossEntropyLoss object (without label smoothing)
            mseloss: MSELoss object
            triplet_loss: TripletMarginLoss object
        """
        sceloss = nn.CrossEntropyLoss(label_smoothing=smooth)
        celoss = nn.CrossEntropyLoss()
        if self.args.rec_loss == 'mse':
            mseloss = nn.MSELoss()
        elif self.args.rec_loss == 'l1':
            mseloss = nn.L1Loss()
        if scale == "binarize":
            mseloss = nn.BCELoss()
        triplet_loss = nn.TripletMarginLoss(margin, p=2)

        return sceloss, celoss, mseloss, triplet_loss

    def freeze_layers(self, ae):
        """
        Freeze all layers except the classifier
        Args:
            ae:

        Returns:

        """
        if not self.args.train_after_warmup:
            for param in ae.dec.parameters():
                param.requires_grad = False
            for param in ae.enc.parameters():
                param.requires_grad = False
            for param in ae.classifier.parameters():
                param.requires_grad = True
        return ae

    @staticmethod
    def get_mccs(lists, traces, subs):
        for group in lists.keys():
            preds, classes = np.concatenate(lists[group]['preds']).argmax(1), np.concatenate(
                lists[group]['classes'])
            traces[group]['mcc'] = MCC(preds, classes)
            for k in subs:
                inds = torch.concat(lists[group]['concs'][k]).detach().cpu().numpy()
                inds = np.array([i for i, x in enumerate(inds) if x > -1])
                if len(inds) > 0:
                    traces[group][f'mcc_{k}'] = MCC(preds[inds], classes[inds])
                else:
                    traces[group][f'mcc_{k}'] = -1

        return traces


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_removal_method', type=str, default='none')
    parser.add_argument('--random_recs', type=int, default=0)
    parser.add_argument('--predict_tests', type=int, default=0)
    parser.add_argument('--balanced_rec_loader', type=int, default=0)
    parser.add_argument('--dann_sets', type=int, default=0)
    parser.add_argument('--dann_plates', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=100)
    parser.add_argument('--early_warmup_stop', type=int, default=100)
    parser.add_argument('--triplet_loss', type=int, default=1)
    parser.add_argument('--train_after_warmup', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--n_epochs', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rec_loss', type=str, default='mse')
    parser.add_argument('--tied_weights', type=int, default=1)
    parser.add_argument('--variational', type=int, default=0)
    parser.add_argument('--zinb', type=int, default=0)
    parser.add_argument('--use_valid', type=int, default=0, help='Use if valid data is in a seperate file')
    parser.add_argument('--use_test', type=int, default=0, help='Use if valid data is in a seperate file')

    parser.add_argument('--path', type=str, default='./example_resources/')
    parser.add_argument('--experiment', type=str, default='Data_FS')

    args = parser.parse_args()

    device = args.device
    log_path = f'logs/ae_classifier/{args.experiment}/valid{args.use_valid}/test{args.use_test}/' \
               f'tw{args.tied_weights}/taw{args.train_after_warmup}/' \
               f'tl{args.triplet_loss}/pseudo{args.predict_tests}/vae{args.variational}/' \
               f'zinb{args.zinb}/balanced{args.balanced_rec_loader}/' \
               f'dannset{args.dann_sets}/loss{args.rec_loss}/'

    csvs_path = f"{args.path}/{args.experiment}/"

    train = Train(args, log_path, fix_thres=-1, load_tb=True)

    train.get_data(csvs_path)
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
        random_seed=4,

    )

    fig = plt.figure()
    # render(plot_contour(model=model, param_x="learning_rate", param_y="weight_decay", metric_name='Loss'))
    # fig.savefig('test.jpg')
    print('Best Loss:', values[0]['loss'])
    print('Best Parameters:')
    print(json.dumps(best_parameters, indent=4))
