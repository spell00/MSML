import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.rinterface import NULL

base = importr('base')


def comBatR(data, batches, classes=NULL, par_prior=True, ref_batch=NULL):
    df = pd.DataFrame(data.values.T)
    data_r = robjects.r.matrix(robjects.FloatVector(df.values.reshape(-1)), nrow=df.shape[0])
    batches_r = robjects.IntVector(batches.reshape(-1))
    sva = importr('sva')
    stats = importr('stats')

    if classes != NULL:
        mod = stats.model_matrix(~base.as_factor(classes))
    newdata = sva.ComBat(data_r, batches_r)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        newdata = np.array(robjects.conversion.rpy2py(newdata))
    return newdata.T


def harmonyR(data, batches, classes=NULL, par_prior=True, ref_batch=NULL):
    df = pd.DataFrame(data.values)
    data_r = robjects.r.matrix(robjects.FloatVector(df.values.reshape(-1)), nrow=df.shape[0])
    batches_r = robjects.IntVector(batches.reshape(-1))
    harmony = importr('harmony')
    newdata = harmony.HarmonyMatrix(data_r, batches_r)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        newdata = np.array(robjects.conversion.rpy2py(newdata))
    return newdata


def zinbWaveR(data, batches, classes=NULL, par_prior=True, ref_batch=NULL):
    df = pd.DataFrame(data.values)
    data_r = robjects.r.matrix(robjects.FloatVector(df.values.reshape(-1)), nrow=df.shape[0])
    batches_r = robjects.IntVector(batches.reshape(-1))
    zinbwave = importr('zinbwave')
    newdata = zinbwave.zinbwave(data_r, batches_r)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        newdata = np.array(robjects.conversion.rpy2py(newdata))
    return newdata


def ligerR(data, batches, classes=NULL, par_prior=True, ref_batch=NULL):
    df = pd.DataFrame(data.values)
    data_r = robjects.r.matrix(robjects.FloatVector(df.values.reshape(-1)), nrow=df.shape[0])
    batches_r = robjects.IntVector(batches.reshape(-1))
    rliger = importr('rliger')
    newdata = rliger.normalize(data_r, batches_r)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        newdata = np.array(robjects.conversion.rpy2py(newdata))
    return newdata


def scMergeR(data, batches, classes=NULL, par_prior=True, ref_batch=NULL):
    df = pd.DataFrame(data.values)
    data_r = robjects.r.matrix(robjects.FloatVector(df.values.reshape(-1)), nrow=df.shape[0])
    batches_r = robjects.IntVector(batches.reshape(-1))
    scMerge = importr('scMerge')
    newdata = scMerge.scMerge(data_r, batches_r)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        newdata = np.array(robjects.conversion.rpy2py(newdata))
    return newdata

