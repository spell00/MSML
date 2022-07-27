#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

import numpy as np
from msml.scikit_learn.utils import load_data
from msml.utils.utils import get_unique_labels


def lows_train_to_test(variables):
    """
    All data is in highs, lows index in the indices to separate them

    :param variables:
    :return:
    """
    test_lows = variables['tests']['highs']['lows']
    train_lows = variables['train']['highs']['lows']
    test_low_inds = np.arange(len(test_lows))[test_lows == 1]
    train_low_inds = np.arange(len(train_lows))[train_lows == 1]

    low_train_data = variables['train']['highs']['data'][train_low_inds]
    low_train_labels = variables['train']['highs']['labels'][train_low_inds]
    low_train_batches = variables['train']['highs']['batches'][train_low_inds]

    low_test_data = variables['tests']['highs']['data'][test_low_inds]
    low_test_labels = variables['tests']['highs']['labels'][test_low_inds]
    low_test_batches = variables['tests']['highs']['batches'][test_low_inds]

    for name in list(variables['tests']['highs'].keys()):
        variables['tests']['highs'][name] = \
            np.delete(variables['tests']['highs'][name], test_low_inds, axis=0)

    low_test_data = np.concatenate((low_train_data, low_test_data))
    low_test_labels = np.concatenate((low_test_labels, low_train_labels))
    low_test_batches = np.concatenate((low_test_batches, low_train_batches))
    low_test_classes = np.array(
        [np.argwhere(label == get_unique_labels(low_test_labels))[0][0]
         for label in low_test_labels]
    )
    variables['tests']['lows'] = {
        'data': low_test_data,
        'labels': low_test_labels,
        'batches': low_test_batches,
        'classes': low_test_classes,
    }
    return variables


def get_variables(args):
    """
    Gets variables for training and testing

    :param args:
    :param binary:
    :return:
    """
    variables = {
        'train': {
            'highs': {},
            'lows': {},
        },
        'tests': {
            'highs': {},
            'lows': {},
        },
        'all': {
            'highs': {},
            'lows': {},
        },
    }
    if args.drop_lows == "train":
        variables['train']['highs'] = load_data(args.train_input, drop_l=False,
                                                drop_b=False, binary=args.binary)
        variables['tests']['highs'] = load_data(args.test_input, drop_l=False,
                                                drop_b=False, binary=args.binary)
        variables = lows_train_to_test(variables)
    # elif args.drop_lows == "tests":
    #     variables['multiclass']['train'] = load_data(args.train_input, drop_l=False,
    #     drop_b=False, binary=False)
    #     variables['multiclass']['tests'] = load_data(args.test_input, drop_l=True,
    #     drop_b=False, binary=False)
    #     if binary:
    #         variables['binary']['train'] = load_data(args.train_input, drop_l=False,
    #         drop_b=False, binary=True)
    #         variables['binary']['tests'] = load_data(args.test_input, drop_l=True,
    #         drop_b=False, binary=True)
    # elif args.drop_lows == "all":
    #     variables['multiclass']['train'] = load_data(args.train_input, drop_l=True,
    #     drop_b=False, binary=False)
    #     variables['multiclass']['tests'] = load_data(args.test_input, drop_l=True,
    #     drop_b=False, binary=False)
    #     if binary:
    #         variables['binary']['train'] = load_data(args.train_input, drop_l=True,
    #         drop_b=False, binary=True)
    #         variables['binary']['tests'] = load_data(args.test_input, drop_l=True,
    #         drop_b=False, binary=True)
    # elif args.drop_lows == "no":
    #     variables['multiclass']['train'] = load_data(args.train_input, drop_l=False,
    #     drop_b=False, binary=False)
    #     variables['multiclass']['tests'] = load_data(args.test_input, drop_l=False,
    #     drop_b=False, binary=False)
    #     if binary:
    #         variables['binary']['train'] = load_data(args.train_input, drop_l=False,
    #         drop_b=False, binary=True)
    #         variables['binary']['tests'] = load_data(args.test_input, drop_l=False,
    #         drop_b=False, binary=True)

    # elif args.drop_lows == "tests":
    #     low_inds = np.arange(len(test_lows))[test_lows == 1]
    #     train_data = np.concatenate((train_data, test_data[low_inds]))
    #     train_labels = np.concatenate((train_labels, test_labels[low_inds]))
    #     train_batches = np.concatenate((train_batches, test_batches[low_inds]))

    #     test_data = np.delete(test_data, low_inds, axis=0)
    #     test_labels = np.delete(test_labels, low_inds, axis=0)
    #     test_batches = np.delete(test_batches, low_inds, axis=0)

    # elif args.drop_lows == "all":
    #     variables['train']['highs'] = load_data(args.train_input, drop_l=True,
    #                                             drop_b=args.drop_blks, binary=args.binary)
    #     variables['tests']['highs'] = load_data(args.test_input, drop_l=True,
    #                                            drop_b=args.drop_blks, binary=args.binary)
    # else:
    #      = load_data(args.train_input, drop_l=False, drop_b=args.drop_blks, binary=args.binary)
    #      = load_data(args.test_input, drop_l=False, drop_b=args.drop_blks, binary=args.binary)
    #     test_low_inds = np.arange(len(test_lows))[test_lows == 1]
    #     train_low_inds = np.arange(len(train_lows))[train_lows == 1]

    #     low_test_data = test_data[test_low_inds]
    #     low_test_labels = test_labels[test_low_inds]
    #     low_test_batches = test_batches[test_low_inds]

    #     test_data = np.delete(test_data, test_low_inds, axis=0)
    #     test_labels = np.delete(test_labels, test_low_inds, axis=0)
    #     test_batches = np.delete(test_batches, test_low_inds, axis=0)

    return variables
