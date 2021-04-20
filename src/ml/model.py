import tensorflow as tf
from tensorflow.keras.layers import Dense,Input,Dropout
from tensorflow.keras.models import Model

"""
Below is what was used for training the most recent version of the model:

Optimizer: Adagrad, with default hyperparameters

Loss: Binary Crossentropy + MSE(adj_mtx,decoded_for_reg)
    - adj_mtx is the adjacency matrix created by create_mtx.py
    and then updated such that each row sums to 1.
    - decoded_for_reg is an output of the model

Epochs: 100

Batch Size: 64
"""

class Encoder(Model):
    """
    Encoder part of the model -> compress dimensionality
    """
    def __init__(self,name, embedding_activation='linear'):
        super().__init__()
        #self.input_drop = Dropout(0.2)
        self.encoded_1 = Dense(512, activation='relu', name=name + "_e1")
        #self.e1_drop = Dropout(0.5)
        self.encoded_2 = Dense(256, activation='relu', name=name + "_e2")
        #self.e2_drop = Dropout(0.5)
        self.encoded_3 = Dense(128, activation='relu', name=name + "_e3")
        #self.e3_drop = Dropout(0.2)
        self.bottleneck = Dense(64, activation=embedding_activation, name=name + "_bottleneck")
    
    def call(self, x, training=None):
        encoded = self.encoded_1(x)
        #encoded = self.e1_drop(encoded)
        encoded = self.encoded_2(encoded)
        #encoded = self.e2_drop(encoded)
        encoded = self.encoded_3(encoded)
        #encoded = self.e3_drop(encoded)
        return self.bottleneck(encoded)

    def call_for_reg(self, x):
        encoded = self.encoded_1(x)
        encoded = self.encoded_2(encoded)
        encoded = self.encoded_3(encoded)
        return self.bottleneck(encoded)
    
class Decoder(Model):
    """
    Decoder part of the model -> expand from compressed latent
        space back to the input space
    """
    def __init__(self, name, output_dim, output_act):
        super().__init__()
        #self.bottleneck_drop = Dropout(0.2)
        self.decoded_1 = Dense(128, activation='relu', name=name + "_d1")
        #self.d1_drop = Dropout(0.4)
        self.decoded_2 = Dense(256, activation='relu', name=name + "_d2")
        #self.d2_drop = Dropout(0.4)
        self.decoded_3 = Dense(512, activation='relu', name=name + "_d3")
        #self.d3_drop = Dropout(0.2)
        self.reconstruct = Dense(output_dim, activation=output_act, name=name + "_reconstruction")
    
    def call(self, x, training=None):
        decoded = self.decoded_1(x)
        decoded = self.decoded_2(decoded)
        decoded = self.decoded_3(decoded)
        return self.reconstruct(decoded)

    # def call_for_reg(self, x):
    #     x = self.bottleneck_drop(x)
    #     decoded = self.decoded_1(x)
    #     decoded = self.d1_drop(decoded)
    #     decoded = self.decoded_2(decoded)
    #     decoded = self.d2_drop(decoded)
    #     decoded = self.decoded_3(decoded)
    #     decoded = self.d3_drop(decoded)
    #     return self.reconstruct(decoded)

class CC_Recommender(Model):
    """
    AutoEncoder build as a recommender system based on the following idea:

        If our input is a binary vector where 1 represents the presence of an
        item in a collection, then an autoencoder trained.
    """
    def __init__(self,num_cards):
        super().__init__()
        self.N = num_cards
        self.encoder = Encoder("encoder", embedding_activation='linear')
        #sigmoid because input is a binary vector we want to reproduce
        self.decoder = Decoder("main", self.N, output_act='sigmoid')
        #softmax because the graph information is probabilities
        #self.input_noise = Dropout(0.5)
        #self.latent_noise = Dropout(0.2)
        self.decoder_for_reg = Decoder("reg",self.N,output_act='softmax')

    def call(self, inputs, training=None):
        """
        input contains two things:
            input[0] = the binary vectors representing the collections
            input[1] = a diagonal matrix of size (self.N X self.N)

        We run the same encoder for each type of input, but with different
        decoders. This is because the goal is to make sure that the compression
        for collections still does a reasonable job compressing individual items.
        So a penalty term (regularization) is added to the model in the ability to
        reconstruct the probability distribution (adjacency matrix) on the item level
        from the encoding.

        The hope is that this regularization enforces this conditional probability to be
        embedded in the recommendations. As the individual items must pull towards items
        represented strongly within the graph.
        """
        if isinstance(inputs, tuple):
            x, identity = inputs
            encode_for_reg = self.encoder(identity)
            # reconstructed_for_reg = self.decoder(encode_for_reg) + 1e-08
            # reconstructed_for_reg = tf.math.divide_no_nan(reconstructed_for_reg, tf.reduce_sum(reconstructed_for_reg, 1, keepdims=True))
            # latent_for_reg = self.latent_noise(encode_for_reg)
            decoded_for_reg = self.decoder_for_reg(encode_for_reg)
        else:
            x = inputs
        # x = self.input_noise(x)
        encoded = self.encoder(x)
        # latent_for_reconstruct = self.latent_noise(encoded)
        reconstruction = self.decoder(encoded)
        if isinstance(inputs, tuple):
            return reconstruction, decoded_for_reg
            # return reconstruction, reconstructed_for_reg, decoded_for_reg
        else:
            return reconstruction
