from tensorflow.keras.utils import Sequence
import numpy as np

class DataGenerator(Sequence):

    def __init__(
        self,
        adj_mtx,
        cubes,
        batch_size=64,
        shuffle=True,
        to_fit=True,
        noise=0.2,
        noise_std=0.1,
    ):
        self.noise_std = noise_std
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.noise = noise
        #initialize inputs and outputs
        self.y_reg = adj_mtx
        self.x_reg = np.zeros_like(adj_mtx)
        np.fill_diagonal(self.x_reg,1)
        self.x_main = cubes
        #initialize other needed inputs
        self.N_cubes = self.x_main.shape[0]
        self.N_cards = self.x_main.shape[1]
        self.reset_indices()
        self.neg_sampler = adj_mtx.sum(0)/adj_mtx.sum()

    def __len__(self):
        """
        return: number of batches per epoch
        """
        return self.N_cubes // self.batch_size

    def __getitem__(self, batch_number):
        """
        Generates a data mini-batch
        param batch_number: which batch to generate  
        return: X and y when fitting. X only when predicting
        """
        main_indices = self.indices[
            batch_number * self.batch_size:(batch_number + 1) * self.batch_size
        ]
        reg_indices = np.random.choice(
            np.arange(self.N_cards),
            len(main_indices),
            p=self.neg_sampler,
        )

        X,y = self.generate_data(
            main_indices,
            reg_indices,
        )

        if self.to_fit:
            return [X[0],X[1]], [y[0],y[1]]
        else:
            return [X[0],X[1]]

    def reset_indices(self):
        self.indices = np.arange(self.N_cubes)
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def on_epoch_end(self):
        """
        Update indices after each epoch
        """
        self.reset_indices()

    def generate_data(self,main_indices,reg_indices):
        cubes = self.x_main[main_indices]
        x_regularization = self.x_reg[reg_indices]
        y_regularization = self.y_reg[reg_indices]

        cut_mask = np.zeros((self.batch_size,self.N_cards))
        add_mask = np.zeros((self.batch_size,self.N_cards))
        y_cut_mask = np.zeros((self.batch_size,self.N_cards))
        for i,cube in enumerate(cubes):
            includes = np.where(cube == 1)[0]
            excludes = np.where(cube == 0)[0]
            size = len(includes)
            noise = np.clip(
                np.random.normal(self.noise,self.noise_std),
                a_min = 0.05,
                a_max = 0.8,
            )
            flip_amount = int(size * noise)
            flip_include = np.random.choice(includes, flip_amount)
            neg_sampler = self.neg_sampler[excludes]/self.neg_sampler[excludes].sum()
            flip_exclude = np.random.choice(excludes, flip_amount, p=neg_sampler)
            y_flip_include = np.random.choice(flip_include, flip_amount//4)
            cut_mask[i,flip_include] = -1
            y_cut_mask[i,y_flip_include] = -1
            add_mask[i,flip_exclude] = 1

        x_cubes = cubes + cut_mask + add_mask
        y_cubes = cubes + y_cut_mask

        return [(x_cubes,x_regularization),(y_cubes,y_regularization)]
