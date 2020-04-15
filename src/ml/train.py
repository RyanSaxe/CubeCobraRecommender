# enable sibling imports
if __name__ == "__main__":
    from sys import path
    from os.path import dirname as dir
    path.append(dir(path[0]))

from model import CC_Recommender
import tensorflow as tf
from non_ml import utils
from generator import DataGenerator
import numpy as np
import json
import os
import os.path
import random
import sys
import pdb


def reset_random_seeds(seed):
    # currently not used
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


args = sys.argv[1:]

epochs = int(args[0])
batch_size = int(args[1])
name = args[2]
reg = float(args[3])
noise = float(args[4])

if len(args) == 6:
    seed = int(args[5])
    reset_random_seeds(seed)

map_file = '././data/maps/nameToId.json'
folder = "././data/cube/"

print('Loading Cube Data . . .\n')

num_cards, name_lookup, card_to_int, int_to_card = \
    utils.get_card_maps(map_file)

num_cubes = utils.get_num_cubes(folder)

cubes = utils.build_cubes(folder, num_cubes, num_cards, name_lookup,
                          card_to_int)

print('Loading Adjacency Matrix . . .\n')

adj_mtx = np.load('././output/full_adj_mtx.npy')

# print('Converting Graph Weights to Probabilities . . . \n')
print('Creating Graph for Regularization . . . \n')

# make easier to learn by dropping super low conditional probabilities
# too_small = np.where(adj_mtx < thresh)
# y_mtx = adj_mtx.copy()
# y_mtx[too_small] = 0
# np.fill_diagonal(y_mtx,1)
# y_mtx = (adj_mtx/adj_mtx.sum(1)[:,None])
# y_mtx = np.nan_to_num(y_mtx,0)
# y_mtx[np.where(y_mtx.sum(1) == 0),np.where(y_mtx.sum(1) == 0)] = 1

y_mtx = adj_mtx.copy()
np.fill_diagonal(y_mtx,1)
y_mtx = (y_mtx/y_mtx.sum(1)[:,None])

print('Setting Up Data for Training . . .\n')

# x_train = np.concatenate([cubes,cubes,cubes[:494]])

# x_items = np.zeros(adj_mtx.shape)
# np.fill_diagonal(x_items,1)

print('Setting Up Model . . . \n')

autoencoder = CC_Recommender(num_cards)
autoencoder.compile(
    optimizer='adam',
    loss=['binary_crossentropy','kullback_leibler_divergence'],
    loss_weights=[1.0,reg],
    metrics=['accuracy'],
)

generator = DataGenerator(
    y_mtx,
    cubes,
    batch_size=batch_size,
    noise=noise,
)

# pdb.set_trace()

autoencoder.fit(
    generator,
    epochs=epochs,
)

# autoencoder.fit(
#     [x_train, x_items],
#     [x_train, adj_mtx],
#     epochs=epochs,
#     batch_size=batch_size,
#     shuffle=True,
# )

dest = f'././ml_files/{name}'
if not os.path.isdir(dest):
    os.makedirs(dest)
autoencoder.save(dest, save_format='tf')
