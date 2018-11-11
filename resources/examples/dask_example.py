import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38\\bin'

import dask.array as da
x = da.ones((15, 15), chunks=(5, 5))

y = x + x.T

y.compute()
y.visualize(filename='dask_graph_example.svg')