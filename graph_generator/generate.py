import networkx as nx
from threading import Thread
from scipy.sparse import csc_matrix
from time import time
from tqdm import tqdm


def write_log(message, endl='\n'):
    print(message)
    f = open("log/graph_generator.log".format(str(time())), "a+")
    if endl == '\n':
        f.write("{}{}".format(message, endl))
    else:
        f.write("{}: {}{} ".format(str(time()), message, endl))

    f.close()

def format(filename, values):

    write_log("Formatting {}".format(filename))

    f = open(filename, 'w+')
    for e in values:
        f.write('{}\n'.format(e))


DIM = 30000
write_log("DIM = {}".format(DIM))

PERC_SPARSE = 0.001
write_log("PERC_SPARSE = {}".format(str((1 - PERC_SPARSE)*100) + '%'))

write_log("Generating graph...")
g = nx.fast_gnp_random_graph(DIM, PERC_SPARSE, seed=1, directed=True)


write_log("Computing pagerank... alpha=0.85, max_iter=200, tol=10**-6", endl='')
start = time()

pr = nx.pagerank(g, alpha=0.85, max_iter=200, tol=10**-6)

write_log("DONE [{}s]".format(time() - start))


write_log("Sorting and dumping pr values...", endl='')
start = time()

pr_sorted = sorted(pr.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

f = open('generated_csc/cur/results.txt', 'w+')

for el in pr_sorted:
    f.write(str(el[0]) + '\n')

f.close()
write_log("DONE [{}s]".format(time() - start))


write_log("Converting matrix to numpy array...", endl='')
start = time()

mat = nx.to_numpy_array(g)

write_log("DONE [{}s]".format(time() - start))


write_log("Formatting matrix in stochastic form...", endl='')
start = time()

for el in tqdm(mat):
    non_zero = (sum(x != 0 for x in el))
    if non_zero != 0:
        for idx, e in enumerate(el):
            if e != 0:
                el[idx] = 1 / non_zero

write_log("DONE [{}s]".format(time() - start))

write_log("Converting matrix in csc", endl='')
start = time()

csc = csc_matrix(mat)

write_log("DONE [{}s]".format(time() - start))

data = csc.data
indices_col = csc.indices
indptr = csc.indptr


write_log("Dumping matrices: ")
start = time()

data_t = Thread(target=format, args=('generated_csc/cur/val.txt'.format(PERC_SPARSE), data))
col_idx_t = Thread(target=format, args=('generated_csc/cur/col_idx.txt'.format(PERC_SPARSE), indices_col))
non_zero_t = Thread(target=format, args=('generated_csc/cur/non_zero.txt'.format(PERC_SPARSE), indptr))


data_t.start()
col_idx_t.start()
non_zero_t.start()

data_t.join()
col_idx_t.join()
non_zero_t.join()

write_log("DONE [{}s]".format(time() - start))

