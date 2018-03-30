# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from hclustvar.recode import recode_var
import scipy.linalg as la
from scipy.cluster.hierarchy import cut_tree
from hclustvar.mixedvarsim import mixed_var_sim
import sys


# Store hcluster_var results
class HClusterResult:
    X = None          # Dataframe of original variables
    Rec = None        # Recoded vectors
    indexj = None     # Indicator for recoded vectors' variable IDs
    Z = None          # HClustering result matrix, compatible with scipy.cluster.hierarchy


# Return the largest eigenvalue as matrix score
def cluster_score(mat_in):
    nrow, ncol = mat_in.shape
    if ncol == 1:
        e_val = 1
        f = mat_in
    else:
        mat_t = mat_in / np.sqrt(nrow)
        mat_t -= np.mean(mat_t, axis=0)
        # calculate the covariance matrix
        C = np.corrcoef(mat_t, rowvar=0)
        e_vals, e_vecs = la.eig(C)
        e_vals = list(e_vals.real)
        idx = e_vals.index(max(e_vals))
        e_val = e_vals[idx]
        f = mat_in @ e_vecs[:, idx]
    return e_val, f


# calculate dissimilarity between two cluster of variables
# based on the idea of PCA, a smaller output value indicates two clusters are more correlated
def cluster_diss(mat_a, mat_b):
    mat_ab = np.concatenate((mat_a, mat_b), axis=1)
    valproA, _ = cluster_score(mat_a)
    valproB, _ = cluster_score(mat_b)
    valproAUB, _ = cluster_score(mat_ab)
    crit = valproA + valproB - valproAUB
    return (crit)


# get nearest neighbor for each cluster
def get_nn_var(diss_mat, flag):
    n_row = diss_mat.shape[0]
    n_col = diss_mat.shape[1]
    nn = [0] * n_row          # store the nearest neighbor fro each variable
    nn_diss = [0] * n_row     # store minimal dissimilarity values for each variable
    MAX_VAL = 1e+12
    if n_row != n_col:
        sys.exit('Error: diss_mat must be a square matrix!')
    if n_row != len(flag):
        sys.exit('Error: flag size not match!')
    for i in range(0, n_row):
        if flag[i] == 1:
            min_obs = -1
            min_dis = MAX_VAL
            for j in range(0, n_col):
                if diss_mat[i, j] < min_dis and i != j:
                    min_dis = diss_mat[i, j]
                    min_obs = j

            nn[i] = min_obs
            nn_diss[i] = min_dis
    return nn, nn_diss


# A (n_var−1) by 4 matrix z is returned.
# At the i-th iteration, clusters with indices z[i, 0] and z[i, 1] are combined to form cluster n_var+i.
# A cluster with an index less than n corresponds to one of the n original observations.
# The distance between clusters z[i, 0] and z[i, 1] is given by z[i, 2].
# The fourth value z[i, 3] represents the number of original observations in the newly formed cluster.
def hcluster_var(df_quant=None, df_quali=None):
    n_var = 0
    if df_quant is not None:
        n_var += len(df_quant.columns)
    if df_quali is not None:
        n_var += len(df_quali.columns)
    if n_var <= 2:
        sys.exit('Error: The number of variables must be greater than 2!')

    df_rec, index_rec = recode_var(df_quant, df_quali)
    MAX_VAL = 1e+12
    init_clust = list(range(n_var))
    flag = [1] * n_var                    # active/inactive indicator
    left_clust = [0] * (n_var - 1)         # left sub-cluster to combine
    right_clust = [0] * (n_var - 1)        # right sub-cluster to combine
    diss = np.zeros((n_var, n_var))       # dissimilarity matrix between variables
    for i in range(0, n_var):
        for j in range(i+1, n_var):
            index_i = pd.Series(index_rec).isin([i]).values
            mat_i = df_rec.iloc[:, index_i].values

            index_j = pd.Series(index_rec).isin([j]).values
            mat_j = df_rec.iloc[:, index_j].values

            diss[i, j] = cluster_diss(mat_i, mat_j)
            diss[j, i] = diss[i, j]

    # find the initial nearest neighbor for each cluster (variable)
    nn, nn_diss = get_nn_var(diss, flag)

    # clust_mat: each column indicates cluster IDs for variables at the specific level
    # a later column related to a lower level, i.e., larger number of clusters
    clust_mat = np.zeros((n_var, n_var))
    # the last column contains the initial cluster IDs, i.e., variable IDs as default
    for i in range(0, n_var):
        clust_mat[i, n_var-1] = i

    clust_mat_py = np.zeros((n_var, n_var))
    for i in range(0, n_var):
        clust_mat_py[i, n_var-1] = i

    # linkage result, comptible with scipy hierarchical clustering
    z = np.zeros((n_var - 1, 4))
    # id for the new combined cluster (compatible with scipy.cluster)
    new_cluster_id = n_var

    # find two clusters with smallest distance to combine
    n_iter = 0
    for n_col in range(n_var-2, -1, -1):
        min_obs = -1
        min_dis = MAX_VAL
        for i in range(0, n_var):
            if flag[i] == 1:
                if nn_diss[i] < min_dis:
                    min_dis = nn_diss[i]
                    min_obs = i
        # clus2 (right) is combined to clus1 (left)
        if min_obs < nn[min_obs]:
            clus1 = min_obs
            clus2 = nn[min_obs]
        if min_obs > nn[min_obs]:
            clus1 = nn[min_obs]
            clus2 = min_obs

        # find variables in clus1 at the previous lower level
        index_col1 = [idx for idx, val in enumerate(clust_mat[:, n_col + 1]) if val == clus1]
        xclus1 = df_rec.iloc[:, pd.Series(index_rec).isin(index_col1).values]
        index_col2 = [idx for idx, val in enumerate(clust_mat[:, n_col + 1]) if val == clus2]
        xclus2 = df_rec.iloc[:, pd.Series(index_rec).isin(index_col2).values]

        # size of the new cluster
        new_cluster_size = len(index_col1) + len(index_col2)
        z[n_iter, 0] = clust_mat_py[index_col1[0], n_col + 1]
        z[n_iter, 1] = clust_mat_py[index_col2[0], n_col + 1]
        z[n_iter, 2] = min_dis
        z[n_iter, 3] = new_cluster_size

        xclus_new = pd.concat([xclus1, xclus2], axis=1)
        left_clust[n_col] = clus1
        right_clust[n_col] = clus2

        # update cluster indicators
        clust_mat[:, n_col] = clust_mat[:, n_col + 1]
        ind1 = np.where(clust_mat[:, n_col] == clus1)[0]
        ind2 = np.where(clust_mat[:, n_col] == clus2)[0]
        clust_mat[ind2, n_col] = clus1

        clust_mat_py[:, n_col] = clust_mat_py[:, n_col + 1]
        clus1_py = clust_mat_py[ind1, n_col][0]
        clus2_py = clust_mat_py[ind2, n_col][0]
        ind_py = np.isin(clust_mat_py[:, n_col], [clus1_py, clus2_py])
        clust_mat_py[ind_py, n_col] = new_cluster_id

        # update dissimilarity between the combined cluster and the rest
        for i in range(0, n_var):
            if i != clus1 and i != clus2 and flag[i] == 1:
                indicescol = [idx for idx, val in enumerate(clust_mat[:, n_col + 1]) if val == i]
                mati = df_rec.iloc[:, pd.Series(index_rec).isin(indicescol).values]
                diss[clus1, i] = cluster_diss(xclus_new, mati)
                diss[i, clus1] = diss[clus1, i]
        for i in range(0, n_var):
            diss[clus2, i] = MAX_VAL
            diss[i, clus2] = diss[clus2, i]
        # turn the combined cluster id to "inactive"
        flag[clus2] = 0
        # update nearest neighbors for the current clustering level
        nn, nn_diss = get_nn_var(diss, flag)
        n_iter += 1
        new_cluster_id += 1

    result = HClusterResult()
    result.Z = z
    result.X = pd.concat([df_quant, df_quali], axis=1)
    result.indexj = index_rec
    result.Rec = df_rec
    return result


# cut tree into n_clusters, and return inter-cluster similarity
def cutree(hclust_result, n_cluster, cal_sim_mat=False):
    z = hclust_result.Z
    indexj = hclust_result.indexj
    df_rec = hclust_result.Rec
    df_x = hclust_result.X
    clust_result = cut_tree(z, n_clusters=n_cluster)
    indexk = indexj.copy()
    for i in range(0, len(indexj)):
        indexk[i] = clust_result[indexj[i], 0]
    var_set = {}
    sim_set = {}
    for g in range(0, n_cluster):
        zclass = df_rec.iloc[:, pd.Series(indexk).isin([g]).values]
        _, latent_f = cluster_score(zclass)
        indices_clus = [idx for idx, val in enumerate(clust_result) if val == g]
        col_names = list(df_x.iloc[:, indices_clus])

        df_squared_loading = pd.DataFrame(columns=['Squared-Loading'], index=col_names)
        for i in range(0, len(indices_clus)):
            col = col_names[i]
            df_squared_loading.loc[col, 'Squared-Loading'] = mixed_var_sim(pd.DataFrame(data=latent_f), df_x[[col]])
        var_set[g] = df_squared_loading

        df_sim = pd.DataFrame(np.ones((len(col_names), len(col_names))), columns=col_names, index=col_names)
        if cal_sim_mat:
            for i in range(0, len(indices_clus) - 1):
                col1 = col_names[i]
                for j in range(i + 1, len(indices_clus)):
                    col2 = col_names[j]
                    df_sim.loc[col1, col2] = mixed_var_sim(df_x[[col1]], df_x[[col2]])
                    df_sim.loc[col2, col1] = df_sim.loc[col1, col2]
            sim_set[g] = df_sim

    return clust_result, var_set, sim_set











