import networkx as nx
import numpy as np
from threading import Thread

def format(filename, values):
    f = open(filename, 'w+')
    for e in values:
        f.write('{}\n'.format(e))

from scipy.sparse import csc_matrix

DIM = 20000
PERC_SPARSE = 0.001
g = nx.fast_gnp_random_graph(DIM, PERC_SPARSE)

mat = nx.to_numpy_array(g)

for el in mat:
    non_zero = (sum(x != 0 for x in el))
    if non_zero != 0:
        for idx, e in enumerate(el):
            if e != 0:
                el[idx] = 1 / non_zero

csc = csc_matrix(mat)

data = csc.data
indices_col = csc.indices
indptr = csc.indptr


data_t = Thread(target=format, args=('generated_csc/val-{}.txt'.format(PERC_SPARSE), data))
col_idx_t = Thread(target=format, args=('generated_csc/col_idx-{}.txt'.format(PERC_SPARSE), indices_col))
non_zero_t = Thread(target=format, args=('generated_csc/non_zero-{}.txt'.format(PERC_SPARSE), indptr))

data_t.start()
col_idx_t.start()
non_zero_t.start()

data_t.join()
col_idx_t.join()
non_zero_t.join()

