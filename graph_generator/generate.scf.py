import networkx as nx
import matplotlib.pyplot as plt

from threading import Thread
import numpy as np
from scipy.sparse import csc_matrix
from time import time
from tqdm import tqdm

m = [[1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1]]


def to_digraph(G):
    H = nx.Graph()
    for u, v, d in G.edges(data=True):
        H.add_edge(u, v)

    return H

def write_log(message, endl='\n'):
    print(message)
    f = open("log/graph_generator-cur.log".format(str(time())), "a+")
    if endl == '\n':
        f.write("{}{}".format(message, endl))
    else:
        f.write("{}: {}{} ".format(str(time()), message, endl))

    f.close()


def format_file(filename, values):

    write_log("Formatting {}".format(filename))

    f = open(filename, 'w+')
    for e in values:
        f.write('{}\n'.format(e))


DIM = 100

#PERC_SPARSE = 0.000000001

#write_log("Generating graph...")
print("Generating graph")
g = nx.scale_free_graph(DIM, seed=1)

print("Formatting to stochastic matrix")
g = to_digraph(g)
g = g.to_directed()

pr = nx.pagerank(g)

write_log("Sorting and dumping pr values...", endl='')
start = time()

pr_sorted = sorted(pr.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

f = open('generated_csc/scf/results.txt', 'w+')

for el in pr_sorted:
    f.write(str(el[0]) + '\n')

print("Formatting to csc")
g = nx.stochastic_graph(g, copy=True)

csc = nx.to_scipy_sparse_matrix(g, format='csc')

write_log("DONE [{}s]".format(time() - start))

data = csc.data
indices_col = csc.indices
indptr = csc.indptr

write_log("Dumping matrices: ")
start = time()

data_t = Thread(target=format_file, args=('generated_csc/scf/val.txt', data))
col_idx_t = Thread(target=format_file, args=('generated_csc/scf/col_idx.txt', indices_col))
non_zero_t = Thread(target=format_file, args=('generated_csc/scf/non_zero.txt', indptr))


data_t.start()
col_idx_t.start()
non_zero_t.start()

data_t.join()
col_idx_t.join()
non_zero_t.join()

write_log("DONE [{}s]".format(time() - start))



