import networkx as nx

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
    H = nx.DiGraph()
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

def remove_self_loops(g):
    for u, v in list(g.edges()):
        if u == v:
            g.remove_edge(u, v)

def format_file(filename, values):

    write_log("Formatting {}".format(filename))

    f = open(filename, 'w+')
    for e in values:
        f.write('{}\n'.format(e))


DIM = int(1e4)

write_log("Generating graph...", endl='')
start = time()
g = nx.watts_strogatz_graph(DIM, 10, 0.2)
sm = nx.stochastic_graph(g.to_directed())

write_log("DONE [{:.10f}]s".format(tim e() - start))

write_log("Formatting to 0 1 matrix...", endl='')
start = time()
cp = nx.to_numpy_array(g)


# Format to 0 / 1 matrix
for idx, el in tqdm(enumerate(cp)):
    for idy, cur in enumerate(el):
        if cp[idx][idy] > 0:
            cp[idx][idy] = 1.0

g = nx.from_numpy_matrix(cp)
write_log("DONE [{:.10f}]s".format(time() - start))

write_log("Formatting to stochastic_graph", endl='')
start = time()
g = g.to_directed()

remove_self_loops(g)
g = nx.stochastic_graph(g)

write_log("DONE [{:.10f}]s".format(time() - start))

####################COMPUTING PAGERANK#########################
write_log("Computing pagerank", endl='')
start = time()
pr = nx.pagerank(g, alpha=0.85, max_iter=200, tol=10**-12)

write_log("DONE [{:.10f}]s".format(time() - start))

########################END COMPUTING PAGERANK#################

write_log("Sorting and dumping pr values...", endl='')
start = time()

pr_sorted = sorted(pr.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

f = open('generated_csc/smw/results.txt', 'w+')

for el in pr_sorted:
    f.write(str(el[0]) + '\n')

write_log("DONE [{:.10f}]s".format(time() - start))


write_log("Formatting matrix in stochastic form...", endl='')
start = time()

g = nx.stochastic_graph(g, copy=True)

write_log("DONE [{:.10f}]s".format(time() - start))

write_log("Converting matrix in csc", endl='')
start = time()

csc = nx.to_scipy_sparse_matrix(g, format='csc')


write_log("DONE [{:.10f}]s".format(time() - start))

data = csc.data
indices_col = csc.indices
indptr = csc.indptr

write_log("Dumping matrices: ")
start = time()

data_t = Thread(target=format_file, args=('generated_csc/smw/val.txt', data))
col_idx_t = Thread(target=format_file, args=('generated_csc/smw/col_idx.txt', indices_col))
non_zero_t = Thread(target=format_file, args=('generated_csc/smw/non_zero.txt', indptr))


data_t.start()
col_idx_t.start()
non_zero_t.start()

data_t.join()
col_idx_t.join()
non_zero_t.join()

write_log("DONE [{:.10f}]s".format(time() - start))

