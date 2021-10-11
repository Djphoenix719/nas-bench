import os
from nasbench import api

from trials.FastList import FastList

INPUT = "input"
OUTPUT = "output"
CONV3X3 = "conv3x3-bn-relu"
CONV1X1 = "conv1x1-bn-relu"
MAXPOOL3X3 = "maxpool3x3"

NUM_VERTICES = 7
MAX_EDGES = 9

EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2  # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2  # Input/output vertices are fixed

ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
ALLOWED_EDGES = [0, 1]  # Binary adjacency matrix

RNG_SEED = 42

OUTPUT_FOLDER = os.path.join(os.getcwd(), 'graphs')

# nasbench = api.NASBench("nasbench_full.tfrecord")
nasbench = api.NASBench("nasbench_only108.tfrecord")

ALL_HASH = FastList(nasbench.hash_iterator())
