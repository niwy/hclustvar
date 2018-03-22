# -*- coding:utf-8 -*-
import hclustvar.recode as recode
import numpy as np
from pandas.api.types import is_numeric_dtype
import sys


# Returns the similarity between two quantitative variables: Pearson correlation
def quant_var_similarity(x1, x2):
    n = len(x1.index)
    z1 = recode.recode_quant(x1).iloc[:, 0].values
    z2 = recode.recode_quant(x2).iloc[:, 0].values
    sim = (np.dot(z1, z2) / n) ** 2
    return sim


# Returns the similarity between a quantitative variable and a qualitative variable.
# The similarity between two variables is defined as the correlation ratio.
def mixed_var_similarity(x_quant, x_quali):
    n = len(x_quant.index)
    z1 = recode.recode_quant(x_quant).iloc[:, 0].values
    g2 = recode.recode_quali(x_quali).values
    ns = np.sum(g2, axis=0)
    a = np.divide(g2.T @ z1, ns)
    sim = np.sum(np.dot(a ** 2, ns) / n)
    return sim


# Square of the canonical correlation between two sets of dummy variables
def quali_var_similarity(x1, x2):
    n = len(x1.index)
    g1 = recode.recode_quali(x1)
    ns = g1.sum()
    ps = ns / n
    z1 = g1.divide(np.sqrt(ps)).values

    g2 = recode.recode_quali(x2)
    ns = g2.sum()
    ps = ns / n
    z2 = g2.divide(np.sqrt(ps)).values

    list_t = [n, z1.shape[1], z2.shape[1]]
    m = list_t.index(min(list_t)) + 1

    if m == 1:
        a1 = z1 @ z1.T / n
        a2 = z2 @ z2.T / n
        a = a1 @ a2
        values, _ = np.linalg.eig(a)
        idx = values.argsort()[::-1]
        values = values[idx]
        sim = values[1].real
    else:
        v12 = z1.T @ z2 / n
        v21 = z2.T @ z1 / n
        if m == 2:
            v = v12 @ v21
        if m == 3:
            v = v21 @ v12
        values, _ = np.linalg.eig(v)
        idx = values.argsort()[::-1]
        values = values[idx]
        sim = values[1].real
    return sim


# Returns the similarity between two quantitative variables, two qualitative
# variables or a quantitative variable and a qualitative variable.
# The similarity between two variables is defined as a square cosine:
# the square of the Pearson correlation when the two variables are quantitative;
# the correlation ratio when one variable is quantitative and the other one is qualitative;
# the square of the canonical correlation between two sets of dummy variables, when the two variables are qualitative.
def mixed_var_sim(x1, x2):
    # x1, x2 are pandas dataframes
    if len(x1.index) != len(x2.index):
        sys.exit('Error: two sets of variables must have the same length!')
    # case quanti-quanti
    if is_numeric_dtype(x1.iloc[:, 0]) and is_numeric_dtype(x2.iloc[:, 0]):
        sim = quant_var_similarity(x1, x2)
    # case quanti-quali
    if is_numeric_dtype(x1.iloc[:, 0]) and not is_numeric_dtype(x2.iloc[:, 0]):
        sim = mixed_var_similarity(x1, x2)
    # case quali-quant
    if not is_numeric_dtype(x1.iloc[:, 0]) and is_numeric_dtype(x2.iloc[:, 0]):
        sim = mixed_var_similarity(x2, x1)
    # case quali-quali
    if not is_numeric_dtype(x1.iloc[:, 0]) and not is_numeric_dtype(x2.iloc[:, 0]):
        sim = quali_var_similarity(x1, x2)
    return sim