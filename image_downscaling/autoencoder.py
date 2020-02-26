import keras
from keras.layers import Input, Dense
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from tensorflow import set_random_seed
import os
import math

def seedy(s):
    np.random.seed(s)
    set_random_seed(s)

class AutoEncoder:
    def __init__(self, x, y, encoding_dim=8):
        self.encoding_dim = encoding_dim
        r = lambda: np.random.randint(1, 3)
        self.x = x
        self.y = y

        self.input_shape = self.x[0].shape
        self.output_shape = self.y[0].shape

        print(self.x)

    def _encoder(self):
        inputs = Input(shape=self.input_shape)

        # current_dim = self.x[0].shape[0]


        temp = Dense(8 * self.encoding_dim, activation='relu')(inputs)
        temp = Dense(4 * self.encoding_dim, activation='relu')(temp)
        temp = Dense(2 * self.encoding_dim, activation='relu')(temp)
        encoded = Dense(self.encoding_dim, activation='relu')(temp)


        # temp = inputs
        # while (current_dim > self.encoding_dim):
        #     temp = Conv2D(64, (3, 3), activation='relu', padding='same')(temp)
        #     temp = MaxPooling2D((2, 2), activation='relu', padding='same')(temp)
        #     current_dim = math.ceil(current_dim / 2)
        # encoded = temp

        model = Model(inputs, encoded)
        self.encoder = model
        return model

    def _decoder(self):
        inputs = Input(shape=(self.encoding_dim, ))

        temp = Dense(2 * self.encoding_dim, activation='relu')(inputs)
        temp = Dense(4 * self.encoding_dim, activation='relu')(temp)
        temp = Dense(8 * self.encoding_dim, activation='relu')(temp)
        decoded = Dense(self.output_shape[0], activation='relu')(temp)

        # decoded = Dense(3)(inputs)
        model = Model(inputs, decoded)
        self.decoder = model
        return model

    def encoder_decoder(self):
        ec = self._encoder()
        dc = self._decoder()

        inputs = Input(shape=self.input_shape)
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        model = Model(inputs, dc_out)

        self.model = model

        print("Encoder: ")
        print(ec.summary())
        print ("Decoder: ")
        print(dc.summary())
        print("Autoencoder: ")
        print(self.model.summary())


        print("Encoder output: ")
        print(ec_out)
        print("Decoder output: ")
        print(dc_out)

        return model

    def fit(self, batch_size=2, epochs=300):
        self.model.compile(optimizer='sgd', loss='mse')
        log_dir = './log/'
        tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir,
                                                 histogram_freq=0,
                                                 write_graph=True,
                                                 write_images=True)

        # need to change output here later on if we want to compare with lower resolution which is a different data set
        self.model.fit(self.x, self.y, epochs=epochs, batch_size=batch_size, callbacks=[tbCallBack])


    def save(self):
        if not os.path.exists(r'./weights'):
            os.mkdir(r'./weights')

        self.encoder.save(r'./weights/encoder_weights.h5')
        self.decoder.save(r'./weights/decoder_weights.h5')
        self.model.save(r'./weights/ae_weights.h5')

