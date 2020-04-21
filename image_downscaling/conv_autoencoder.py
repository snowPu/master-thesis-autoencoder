import keras
from keras.layers import Input, Dense
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate, BatchNormalization, Conv2DTranspose
from keras.layers import ReLU
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
# from tensorflow import set_random_seed
import tensorflow as tf
import os
# import math
from keras.applications.vgg19 import VGG19
import keras.backend as keras_backend
from matplotlib import pyplot as plt
from datetime import datetime
from keras.callbacks import CSVLogger

# Define VGG Model

vgg_model = VGG19(include_top=False, weights='imagenet', input_shape=(100, 100, 3))

ADADELTA = keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
# ADAM = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
SGD = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
ADAMAX = keras.optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
# NADAM = keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)

#
# def seedy(s):
#     np.random.seed(s)
#     set_random_seed(s)


def ADAM(lr, beta_1, beta_2, epsilon, decay):
    return keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay)

def NADAM(lr, beta_1, beta_2):
    return keras.optimizers.Nadam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2)

def VGGloss(y_true, y_pred): # Note the parameter order
    f_p = vgg_model(y_pred)
    f_t = vgg_model(y_true)
    return keras_backend.mean(keras_backend.square(f_p - f_t))

def SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def MSSSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, 1.0, power_factors=(0.0448, 0.2856, 0.3001, 0.2363)))

def VGG_SSIM_Loss(y_true, y_pred):
    return 0.65 * VGGloss(y_true, y_pred) + 0.35 * SSIMLoss(y_true, y_pred)

def VGG_MSSSIM_Loss(y_true, y_pred):
    return 0.6 * VGGloss(y_true, y_pred) + 0.4 * MSSSIMLoss(y_true, y_pred)

def VGG_MSE_Loss(y_true, y_pred):
    return 0.8 * VGGloss(y_true, y_pred) + 0.2 * tf.keras.losses.MSE(y_true, y_pred)

def VGG_SSIM_MSE_Loss(y_true, y_pred):
    return VGGloss(y_true, y_pred) * .5 + SSIMLoss(y_true, y_pred) * .4 + tf.keras.losses.MSE(y_true, y_pred) * .1

def VGG_MSSSIM_MSE_Loss(y_true, y_pred):
    return VGGloss(y_true, y_pred) * .5 + MSSSIMLoss(y_true, y_pred) * .4 + tf.keras.losses.MSE(y_true, y_pred) * .1


class AutoEncoder:
    def __init__(self, x, y, encoder_weights=None, decoder_weights=None):
        # self.encoding_dim = encoding_dim
        r = lambda: np.random.randint(1, 3)
        self.x = x
        self.y = y
        self.init_encoder = encoder_weights
        self.init_decoder = decoder_weights

        # ADAM
        self.adam_parameters = {
            'lr': 0.0001,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-08,
            'decay': 0.0
        }

        self.nadam_parameters = {
            'lr': 0.0002,
            'beta_1': 0.9,
            'beta_2': 0.999,
        }

        self.optimizer_parameters_map = {
            'adam': self.adam_parameters,
            'nadam': self.nadam_parameters
        }


        self.input_shape = self.x[0].shape
        self.output_shape = self.y[0].shape

        self.encoding_dim = int(self.input_shape[0] / 8)

        self.losses = {
            'perceptual': VGGloss,
            'ssim': SSIMLoss,
            'msssim': MSSSIMLoss,
            'mse': 'mse',
            'perceptual_ssim': VGG_SSIM_Loss,
            'perceptual_msssim': VGG_MSSSIM_Loss,
            'perceptual_ssim_mse': VGG_SSIM_MSE_Loss,
                'perceptual_msssim_mse': VGG_MSSSIM_MSE_Loss,
            'perceptual_mse': VGG_MSE_Loss
            # 'perceptual_ssim_mse': [VGGloss, SSIMLoss, 'mse']
        }

        self.optimizers = {
            'SGD': SGD,
            'adadelta': ADADELTA,
            'adam': ADAM(self.adam_parameters['lr'],
                              self.adam_parameters['beta_1'],
                              self.adam_parameters['beta_2'],
                              self.adam_parameters['epsilon'],
                              self.adam_parameters['decay']),
            'adamax': ADAMAX,
            'nadam': NADAM(self.nadam_parameters['lr'],
                           self.nadam_parameters['beta_1'],
                           self.nadam_parameters['beta_2'])
        }


        print(self.x)

    def _encoder(self):
        inputs = Input(shape=(None, None, 3))

        current_dim = self.input_shape[0]

        # temp = inputs
        cnt = 0

        conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D((2, 2), padding='same')(conv1)
        conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D((2, 2), padding='same')(conv2)
        conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
        encoded = MaxPooling2D((2, 2), padding='same')(conv3)

        model = Model(inputs, encoded)
        if self.init_encoder is not None:
            model.load_weights(self.init_encoder)
        self.encoder = model
        return model

    def _decoder(self):
        inputs = Input(shape=(None, None, 64))

        current_dim = self.encoding_dim
        # temp = inputs
        cnt = 0

        temp = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        temp = Conv2D(64, (3, 3), activation='relu', padding='same')(temp)
        temp = UpSampling2D((2, 2))(temp)
        temp = Conv2D(32, (3, 3), activation='relu', padding='same')(temp)
        temp = Conv2D(32, (3, 3), activation='relu', padding='same')(temp)
        temp = UpSampling2D((2, 2))(temp)
        # temp = Conv2D(32, (3, 3), activation='relu', padding='same')(temp)
        # temp = UpSampling2D((2, 2))(temp)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(temp)
        model = Model(inputs, decoded)
        if self.init_decoder is not None:
            model.load_weights(self.init_decoder)
        self.decoder = model
        return model

    def encoder_decoder(self):
        inputs = Input(shape=(None, None, 3))
        # 256 x 256

        # encoder

        conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = BatchNormalization(conv1)
        conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization(conv1)
        pool1 = MaxPooling2D((2, 2), padding='same')(conv1) # 128
        conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = BatchNormalization(conv2)
        conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization(conv2)
        pool2 = MaxPooling2D((2, 2), padding='same')(conv2) # 64
        conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = BatchNormalization(conv3)
        conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization(conv3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling2D((2, 2), padding='same')(drop3) # 32


        conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
        drop4 = Dropout(0.5)(conv4)

        # decoder

        up5 = Conv2D(64, (3, 3), activation='relu', padding='same')(Conv2DTranspose((2, 2))(drop4)) # 64
        merge5 = concatenate([drop3, up5], axis=3)
        conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge5)
        conv5 = BatchNormalization(conv5)
        conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
        conv5 = BatchNormalization(conv5)

        up6 = Conv2D(32, (3, 3), activation='relu', padding='same')(Conv2DTranspose((2, 2))(conv5)) # 128
        merge6 = concatenate([conv2, up6], axis=3)
        conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(merge6)
        conv6 = BatchNormalization(conv6)
        conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv6)
        conv6 = BatchNormalization(conv6)
        conv6 = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv6)


        model = Model(inputs, conv6)

        self.model = model

        print("Autoencoder: ")
        print(self.model.summary())

        return model

    def fit(self, batch_size=5, epochs=300, optimizer='SGD', loss='mse'):
        self.optimizer = optimizer
        self.model.compile(optimizer=self.optimizers[optimizer], loss=self.losses[loss])
        log_file = './logs/' + 'log_' + str(epochs) + '_' + \
                   str(self.optimizer_parameters_map[self.optimizer]['lr']) + '_' + \
                   str(self.optimizer_parameters_map[self.optimizer]['beta_1']) + '_' + \
                   str(self.optimizer_parameters_map[self.optimizer]['beta_2']) + '_' + \
                   str(datetime.timestamp(datetime.now())) + '.csv'
        # tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir,
        #                                          histogram_freq=0,
        #                                          write_graph=True,
        #                                          write_images=True)
        # callbacks=[tbCallBack],
        # need to change output here later on if we want to compare with lower resolution which is a different data set
        csv_logger = CSVLogger(log_file, append=True, separator=';')
        return self.model.fit(self.x, self.y,
                              epochs=epochs,
                              batch_size=batch_size,
                              callbacks=[csv_logger],
                              validation_split=0.2)

    def plot_history(self, trained, epochs, fig_folder):
        loss = trained.history['loss']
        val_loss = trained.history['val_loss']

        if not os.path.exists(fig_folder):
            os.mkdir(fig_folder)
        fig_path = fig_folder + '/' + 'plot_' + str(epochs) + '_' + \
                   str(self.optimizer_parameters_map[self.optimizer]['lr']) + '_' + \
                   str(self.optimizer_parameters_map[self.optimizer]['beta_1']) + '_' + \
                   str(self.optimizer_parameters_map[self.optimizer]['beta_2']) + '_' + \
                   str(datetime.timestamp(datetime.now())) + '.png'

        epoch_set = range(epochs)

        plt.figure()
        plt.plot(epoch_set, loss, color='orange', label='Training loss')
        plt.plot(epoch_set, val_loss, color='blue', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(fig_path)
        plt.show()



    def save(self, weights_folder='weights'):
        # if not os.path.exists(weights_folder):
        #     os.mkdir(weights_folder)
        current_weights_folder = weights_folder + '_' + \
                                 str(self.optimizer_parameters_map[self.optimizer]['lr']) + '_' + \
                                 str(self.optimizer_parameters_map[self.optimizer]['beta_1']) + '_' + \
                                 str(self.optimizer_parameters_map[self.optimizer]['beta_2']) + '_' + \
                                 str(datetime.timestamp(datetime.now()))
        os.mkdir(current_weights_folder)

        encoder_file = current_weights_folder + '/encoder_weights.h5'
        decoder_file = current_weights_folder + '/decoder_weights.h5'
        ae_file = current_weights_folder + '/ae_weights.h5'

        # self.encoder.save(encoder_file)
        # self.decoder.save(decoder_file)
        self.model.save(ae_file)

