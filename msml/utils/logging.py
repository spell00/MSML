import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
CUDA_VISIBLE_DEVICES = ""

from tensorboard.plugins.hparams import api as hp
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('CPU')
tf.config.set_visible_devices(physical_devices)


# It is useless to run tensorflow on GPU and it takes a lot of GPU RAM for nothing
class TensorboardLogging:
    def __init__(self, hparams_filepath, params):
        self.params = params
        self.hparams_filepath = hparams_filepath
        HPARAMS = [
            hp.HParam('dropout', hp.RealInterval(0.0, 1.0)),
            hp.HParam('translate_h', hp.RealInterval(0.0, 0.3)),
            hp.HParam('translate_v', hp.RealInterval(0.0, 0.3)),
            hp.HParam('shear', hp.RealInterval(0.0, 0.2)),
            hp.HParam('scale_ratio_max', hp.RealInterval(0.2, 3.3)),
            hp.HParam('scale_erasing_max', hp.RealInterval(0.2, 0.33)),
            hp.HParam('crop_ratio', hp.RealInterval(0.0, 1.0)),
            hp.HParam('crop_scale', hp.RealInterval(0.0, 1.0)),
            hp.HParam('blur_max', hp.RealInterval(0.0, 0.1)),
            hp.HParam('n_res', hp.IntInterval(1, 10)),
            hp.HParam('beta1', hp.RealInterval(0.9, 1.0)),
            hp.HParam('beta2', hp.RealInterval(0.9, 1.0)),
            hp.HParam('min_lr', hp.RealInterval(1e-15, 1e-1)),
            hp.HParam('l1', hp.RealInterval(1e-15, 1e-1)),
            hp.HParam('wd', hp.RealInterval(1e-15, 1e-1)),
            hp.HParam('lr', hp.RealInterval(1e-8, 1e-5))
        ]

        with tf.summary.create_file_writer(hparams_filepath).as_default():
            hp.hparams_config(
                hparams=HPARAMS,
                metrics=[
                    hp.Metric('train_accuracy', display_name='Train Accuracy'),
                    hp.Metric('valid_accuracy', display_name='Valid Accuracy'),
                    hp.Metric('train_auc', display_name='Train AUC'),
                    hp.Metric('valid_auc', display_name='Valid AUC'),
                    # hp.Metric('test_accuracy', display_name='Test Accuracy'),
                    hp.Metric('train_loss', display_name='Train Loss'),
                    hp.Metric('valid_loss', display_name='Valid Loss'),
                    # hp.Metric('test_loss', display_name='Test Loss'),
                    # hp.Metric('train_mcc', display_name='Train MCC'),
                    # hp.Metric('valid_mcc', display_name='Valid MCC'),
                    # hp.Metric('test_mcc', display_name='Test MCC')
                ],
            )

    def logging(self, traces):
        with tf.summary.create_file_writer(self.hparams_filepath).as_default():
            hp.hparams({
                'dropout': self.params['dropout'],
                'translate_h': self.params['translate_h'],
                'translate_v': self.params['translate_v'],
                'shear': self.params['shear'],
                'lr': self.params['learning_rate'],
                'wd': self.params['weight_decay'],
                'l1': self.params['l1'],
                # 'min_lr': self.params['min_lr'],
                # 'beta2': self.params['beta2'],
                # 'beta1': self.params['beta1'],
                # 'n_res': self.params['n_res'],
                'blur_max': self.params['blur_max'],
                'crop_scale': self.params['crop_scale'],
                'crop_ratio': self.params['crop_ratio'],
                'scale_erasing_max': self.params['scale_erasing_max'],
                'scale_ratio_max': self.params['scale_ratio_max'],
            })  # record the values used in this trial
            tf.summary.scalar('train_accuracy', traces['train_accuracy'], step=1)
            tf.summary.scalar('valid_accuracy', traces['valid_accuracy'], step=1)
            tf.summary.scalar('train_auc', traces['train_auc'], step=1)
            tf.summary.scalar('valid_auc', traces['valid_auc'], step=1)
            tf.summary.scalar('valid_accuracy_highs', traces['valid_accuracy_highs'], step=1)
            tf.summary.scalar('valid_accuracy_lows', traces['valid_accuracy_lows'], step=1)
            tf.summary.scalar('train_loss', traces['train_loss'], step=1)
            tf.summary.scalar('valid_loss', traces['valid_loss'], step=1)

    def logging_sdae(self, traces):
        with tf.summary.create_file_writer(self.hparams_filepath).as_default():
            hp.hparams({
                'dropout': self.params['dropout'],
                'translate_h': self.params['translate_h'],
                'translate_v': self.params['translate_v'],
                'shear': self.params['shear'],
                'lr': self.params['learning_rate'],
                'wd': self.params['weight_decay'],
                'l1': self.params['l1'],
                'min_lr': self.params['min_lr'],
                'beta2': self.params['beta2'],
                'beta1': self.params['beta1'],
                'n_res': self.params['n_res'],
                'blur_max': self.params['blur_max'],
                'crop_scale': self.params['crop_scale'],
                'crop_ratio': self.params['crop_ratio'],
                'scale_erasing_max': self.params['scale_erasing_max'],
                'scale_ratio_max': self.params['scale_ratio_max'],
            })  # record the values used in this trial
            tf.summary.scalar('train_accuracy', traces['train_accuracy'], step=1)
            tf.summary.scalar('valid_accuracy', traces['valid_accuracy'], step=1)
            tf.summary.scalar('valid_accuracy_highs', traces['valid_accuracy_highs'], step=1)
            tf.summary.scalar('valid_accuracy_lows', traces['valid_accuracy_lows'], step=1)
            tf.summary.scalar('train_loss', traces['train_loss'], step=1)
            tf.summary.scalar('valid_loss', traces['valid_loss'], step=1)


class TensorboardLoggingNoAug:
    def __init__(self, hparams_filepath, params):
        self.params = params
        self.hparams_filepath = hparams_filepath
        HPARAMS = [
            hp.HParam('dropout', hp.RealInterval(0.0, 1.0)),
            hp.HParam('l1', hp.RealInterval(1e-15, 1e-1)),
            hp.HParam('wd', hp.RealInterval(1e-15, 1e-1)),
            hp.HParam('wdd', hp.RealInterval(1e-15, 1e-1)),
            hp.HParam('wdc', hp.RealInterval(1e-15, 1e-1)),
            hp.HParam('lr', hp.RealInterval(1e-8, 1e-1)),
            hp.HParam('lrd', hp.RealInterval(1e-8, 1e-1)),
            hp.HParam('lrc', hp.RealInterval(1e-8, 1e-1)),
        ]

        with tf.summary.create_file_writer(hparams_filepath).as_default():
            hp.hparams_config(
                hparams=HPARAMS,
                metrics=[
                    hp.Metric('train_accuracy', display_name='Train Accuracy'),
                    hp.Metric('valid_accuracy', display_name='Valid Accuracy'),
                    hp.Metric('train_auc', display_name='Train AUC'),
                    hp.Metric('valid_auc', display_name='Valid AUC'),
                    # hp.Metric('test_accuracy', display_name='Test Accuracy'),
                    hp.Metric('train_loss', display_name='Train Loss'),
                    hp.Metric('valid_loss', display_name='Valid Loss'),
                    # hp.Metric('test_loss', display_name='Test Loss'),
                    # hp.Metric('train_mcc', display_name='Train MCC'),
                    # hp.Metric('valid_mcc', display_name='Valid MCC'),
                    # hp.Metric('test_mcc', display_name='Test MCC')
                ],
            )

    def logging(self, traces):
        with tf.summary.create_file_writer(self.hparams_filepath).as_default():
            hp.hparams({
                'dropout': self.params['dropout'],
                'lr': self.params['lr'],
                'lrc': self.params['lrc'],
                'lrd': self.params['lrd'],
                'wd': self.params['wd'],
                'wdc': self.params['wdc'],
                'wdd': self.params['wdd'],
                'l1': self.params['l1'],
            })  # record the values used in this trial
            tf.summary.scalar('train_accuracy', traces['train_accuracy'], step=1)
            tf.summary.scalar('valid_accuracy', traces['valid_accuracy'], step=1)
            tf.summary.scalar('train_auc', traces['train_auc'], step=1)
            tf.summary.scalar('valid_auc', traces['valid_auc'], step=1)
            tf.summary.scalar('valid_accuracy_highs', traces['valid_accuracy_highs'], step=1)
            tf.summary.scalar('valid_accuracy_lows', traces['valid_accuracy_lows'], step=1)
            tf.summary.scalar('train_loss', traces['train_loss'], step=1)
            tf.summary.scalar('valid_loss', traces['valid_loss'], step=1)

    def logging_sdae(self, traces):
        with tf.summary.create_file_writer(self.hparams_filepath).as_default():
            hp.hparams({
                'dropout': self.params['dropout'],
                'translate_h': self.params['translate_h'],
                'translate_v': self.params['translate_v'],
                'shear': self.params['shear'],
                'lr': self.params['learning_rate'],
                'wd': self.params['weight_decay'],
                'l1': self.params['l1'],
                'min_lr': self.params['min_lr'],
                'beta2': self.params['beta2'],
                'beta1': self.params['beta1'],
                'n_res': self.params['n_res'],
                'blur_max': self.params['blur_max'],
                'crop_scale': self.params['crop_scale'],
                'crop_ratio': self.params['crop_ratio'],
                'scale_erasing_max': self.params['scale_erasing_max'],
                'scale_ratio_max': self.params['scale_ratio_max'],
            })  # record the values used in this trial
            tf.summary.scalar('train_accuracy', traces['train_accuracy'], step=1)
            tf.summary.scalar('valid_accuracy', traces['valid_accuracy'], step=1)
            tf.summary.scalar('valid_accuracy_highs', traces['valid_accuracy_highs'], step=1)
            tf.summary.scalar('valid_accuracy_lows', traces['valid_accuracy_lows'], step=1)
            tf.summary.scalar('train_loss', traces['train_loss'], step=1)
            tf.summary.scalar('valid_loss', traces['valid_loss'], step=1)


class TensorboardLoggingSDAE:

    def __init__(self, hparams_filepath, params):
        self.params = params
        self.hparams_filepath = hparams_filepath
        HPARAMS = [
            # hp.HParam('dropout', hp.RealInterval(0.0, 1.0)),
            hp.HParam('beta1', hp.RealInterval(0.9, 1.0)),
            hp.HParam('beta2', hp.RealInterval(0.9, 1.0)),
            hp.HParam('min_lr', hp.RealInterval(1e-15, 1e-1)),
            hp.HParam('l1', hp.RealInterval(1e-15, 1e-1)),
            hp.HParam('wd', hp.RealInterval(1e-15, 1e-1)),
            hp.HParam('lr', hp.RealInterval(1e-8, 1e-5))
        ]

        with tf.summary.create_file_writer(hparams_filepath).as_default():
            hp.hparams_config(
                hparams=HPARAMS,
                metrics=[
                    hp.Metric('train_linsvc_acc', display_name='Train Accuracy (linsvc)'),
                    hp.Metric('valid_linsvc_acc', display_name='Valid Accuracy (linsvc)'),
                    hp.Metric('test_linsvc_acc', display_name='Test Accuracy (linsvc)'),
                    hp.Metric('train_lda_acc', display_name='Train Accuracy (lda)'),
                    hp.Metric('valid_lda_acc', display_name='Valid Accuracy (lda)'),
                    hp.Metric('test_lda_acc', display_name='Test Accuracy (lda)'),
                    hp.Metric('train_loss', display_name='Train Loss'),
                ],
            )

    def logging(self, traces):
        with tf.summary.create_file_writer(self.hparams_filepath).as_default():
            hp.hparams({
                'dropout': self.params['dropout'],
                'lr': self.params['learning_rate'],
                'wd': self.params['weight_decay'],
                'l1': self.params['l1'],
                'min_lr': self.params['min_lr'],
                'beta2': self.params['beta2'],
                'beta1': self.params['beta1'],
            })  # record the values used in this trial
            tf.summary.scalar('train_linsvc_acc', traces['train_linsvc_acc'], step=1)
            tf.summary.scalar('valid_linsvc_acc', traces['valid_linsvc_acc'], step=1)
            tf.summary.scalar('test_linsvc_acc', traces['test_linsvc_acc'], step=1)
            tf.summary.scalar('train_lda_acc', traces['train_lda_acc'], step=1)
            tf.summary.scalar('valid_lda_acc', traces['valid_lda_acc'], step=1)
            tf.summary.scalar('test_lda_acc', traces['test_lda_acc'], step=1)
            tf.summary.scalar('train_loss', traces['train_loss'], step=1)
        del traces
