# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import sys

# recode categorical variables
def recode_quali(df_quali):
    # one categorical var will be converted to n columns, n equals to number of distinct categories
    df_logic = pd.get_dummies(df_quali)
    col_n_miss = df_logic.isna().sum()
    # replace missing values with 0
    df_logic.fillna(0, inplace=True)
    col_sum = df_logic.sum()
    n_row = len(df_logic.index)
    # Columns like ['a', NA, NA], ['a', 'a', 'a'] are taken as identical
    if ((n_row - col_n_miss) == col_sum).sum() != 0:
        sys.exit('Error: There are columns in df_quali where all the categories are identical!')
    return df_logic


# recode numeric variables
def recode_quant(df_quant):
    for col in df_quant:
        if df_quant[col].value_counts().size < 2:
            sys.exit('Error: Found variable in df_quali with identical value: ' + col)
    # replace missing values with mean values
    df_cod = df_quant.fillna(df_quant.mean())
    df_cod_scale = pd.DataFrame(scale(df_cod), columns=df_quant.columns)
    return df_cod_scale


def recode_var(df_quant, df_quali):
    # check data size
    if df_quant is not None and df_quali is not None:
        if len(df_quant.index) != len(df_quali.index):
            sys.exit('Different row numbers in df_quant and df_quali')

    # process numeric vars
    if df_quant is not None:
        for col in df_quant:
            if not np.issubdtype(df_quant[col].dtype, np.number):
                sys.exit('Error: Found non-numeric variable in df_quant: ' + col)
        df_quant_recode = recode_quant(df_quant)

    if df_quali is not None:
        for col in df_quali:
            if np.issubdtype(df_quali[col].dtype, np.number):
                sys.exit('Error: Found numeric variable in df_quanli: ' + col)
        df_quali_logic = recode_quali(df_quali)
        # df_quali_recode, index_quali = recode_quali(df_quali)

        col_mean = df_quali_logic.mean()
        col_mean_sqrt = np.sqrt(col_mean)

        df_cod = df_quali_logic.divide(col_mean_sqrt)
        col_moy = df_cod.mean()
        df_quali_recode = df_cod.subtract(col_moy)

        num_unique_val = []
        for col in df_quali:
            num_unique_val.append(df_quali[col].value_counts().size)
        # describe the new cols' original var indices
        index_quali = []
        for i in range(0, len(df_quali.columns)):
            index_quali.extend([i] * num_unique_val[i])

    # Merge numeric variables and categorical variables
    if df_quant is not None and df_quali is not None:
        df_out = pd.concat([df_quant_recode, df_quali_recode], axis=1)
        n_var_quant = len(df_quant.columns)
        index_var = list(range(n_var_quant))
        new_index_quali = [idx + n_var_quant for idx in index_quali]
        index_var.extend(new_index_quali)

    if df_quant is not None and df_quali is None:
        df_out = df_quant_recode
        index_var = range(0, len(df_quali.columns))

    if df_quant is None and df_quali is not None:
        df_out = df_quali_recode
        index_var = index_quali

    return df_out, index_var




