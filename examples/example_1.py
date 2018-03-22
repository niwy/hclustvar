import pandas as pd
import hclustvar.hclustvar as hh
from scipy.cluster.hierarchy import dendrogram


if __name__ == '__main__':
    df_n = pd.DataFrame({'Var1': [1, 3, 4, 2, 4, 6],
                         'Var2': [2, 6, 6, 3, 1, 7],
                         'Var3': [3, 1, 1, 4, 2, 9]})
    df_c = pd.DataFrame({'A': ['a', 'b', 'a', 'a', 'c', 'b'],
                         'B': ['b', 'a', 'c', 'b', 'c', 'a']})

    result = hh.hcluster_var(df_quant=df_n, df_quali=df_c)
    dendrogram(result.Z, labels=list(df_n) + list(df_c))
    groups, var, sim = hh.cutree(result, n_cluster=2, cal_sim_mat=True)