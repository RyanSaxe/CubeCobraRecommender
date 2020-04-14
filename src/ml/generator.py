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
        pos_noise=0.2,
        neg_noise=0.2,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.pos_noise = pos_noise
        self.neg_noise = neg_noise
        #initialize inputs and outputs
        self.y_reg = adj_mtx
        self.x_reg = np.zeros_like(adj_mtx)
        np.fill_diagonal(self.x_reg,1)
        self.x_main = cubes
        #initialize other needed inputs
        self.N_cubes = self.x_main.shape[0]
        self.N_cards = self.x_main.shape[1]
        self.reset_indices()

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
        )

        X = self._generate_X(
            main_indices,
            reg_indices,
        )

        if self.to_fit:
            y = self._generate_y(
                main_indices,
                reg_indices,
            )
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

    def _generate_X(self,main_indices,reg_indices):
        cubes = self.x_main[main_indices]
        cut_mask,add_mask = self.add_noise(cubes)
        cubes = cubes + cut_mask + add_mask
        regularization = self.x_reg[reg_indices]
        return cubes,regularization
    
    def _generate_y(self,main_indices,reg_indices):
        cubes = self.x_main[main_indices]
        regularization = self.y_reg[reg_indices]
        return cubes,regularization

    def add_noise(self,cubes):
        cut_mask = np.zeros((self.batch_size,self.N_cards))
        add_mask = np.zeros((self.batch_size,self.N_cards))
        for i,cube in enumerate(cubes):
            includes = np.where(cube == 1)[0]
            excludes = np.where(cube == 0)[0]
            size = len(includes)
            flip_include_amount = int(size * self.pos_noise)
            flip_exclude_amount = int(size * self.neg_noise)
            flip_include = np.random.choice(includes, flip_include_amount)
            flip_exclude = np.random.choice(excludes, flip_exclude_amount)
            cut_mask[i,flip_include] = -1
            add_mask[i,flip_exclude] = 1
        return cut_mask, add_mask
