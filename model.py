import tensorflow as tf
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.models import Model

class Encoder(Model):
    """
    Encoder part of the model -> compress dimensionality
    """
    def __init__(self):
        super().__init__()
        self.encoded_1 = Dense(512, activation='relu')
        self.encoded_2 = Dense(256, activation='relu')
        self.encoded_3 = Dense(128, activation='relu')
        self.bottleneck = Dense(64, activation='relu')
    
    def call(self, x):
        encoded = self.encoded_1(x)
        encoded = self.encoded_2(x)
        encoded = self.encoded_3(x)
        return self.bottleneck(encoded)
    
class Decoder(Model):
    """
    Decoder part of the model -> expand from compressed latent
        space back to the input space
    """
    def __init__(self, output_dim, output_act):
        super().__init__()
        self.decoded_1 = Dense(128, activation='relu')
        self.decoded_2 = Dense(256, activation='relu')
        self.decoded_3 = Dense(512, activation='relu')
        self.reconstruct = Dense(output_dim, activation=output_act)
    
    def call(self, x):
        decoded = self.decoded_1(x)
        decoded = self.decoded_2(x)
        decoded = self.decoded_3(x)
        return self.reconstruct(decoded)

class CC_Recommender(Model):
    """
    AutoEncoder build as a recommender system based on the following idea:

        If our input is a binary vector where 1 represents the presence of an
        item in a collection, then an autoencoder trained 
    """
    def __init__(self,num_cards):
        super().__init__()
        self.N = num_cards
        self.encoder = Encoder()
        #sigmoid because input is a binary vector we want to reproduce
        self.decoder = Decoder(self.N,output_act='sigmoid')
        #softmax because the graph information is probabilities
        self.decoder_for_reg = Decoder(self.N,output_act='softmax')
    
    def call(self, input):
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
        x,identity = input
        reconstruction = self.recommend(x)
        encode_for_reg = self.encoder(identity)
        decoded_for_reg = self.decoder_for_reg(encode_for_reg)
        return reconstruction, decoded_for_reg
    
    def recommend(self, x):
        """
        recommend function is pulled outside of `call` in order to 
        allow calling a recommendation without passing the diagonal
        matrix.

        Note, this recommend function had not been pulled out in the first
        iteration of this model, which is why the ml_recommend.py script
        does not call this function
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded