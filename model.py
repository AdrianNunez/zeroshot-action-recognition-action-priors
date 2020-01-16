import gc 
import os
import sys
import h5py
import keras
import pickle
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Input, Model
from keras import regularizers
from keras.layers import (Layer, MaxPooling2D, Conv2D,
                          Flatten, Activation, Dense,
                          Dropout, ZeroPadding2D,
                          Lambda, concatenate, Input,
                          GlobalAveragePooling2D,
                          multiply, add, Reshape,
                          GaussianNoise, ConvLSTM2D,
                          TimeDistributed, RepeatVector,
                          Bidirectional, BatchNormalization,
                          Conv2DTranspose, Permute)
from keras import initializers
import scipy.io as sio
from keras.applications.resnet50 import ResNet50

# For debug purposes
def print_layer_names(model):
    """
    Takes a model (instance of the Model class of Keras) and prints the
    information of each layer (name and output shape)
    -Input:
        * model (Keras Model object)
    """
    for count, layer in enumerate(model.layers):
        print(count, layer.name, layer.output_shape)

def deploy_network(config, training_params, class_weights=None):
    """
    Instantiate the neural network used as a detector. Load a ResNet50 with
    Imagenet weights and a ConvLSTM. The final classifier is a Fully Connected
    layer.
    - Input:
        * config (configuration dictionary)
        * training_params (parameters dictionary)
        * (optional) class_weights (array of floats)
    - Output:
        * final_model (Keras Model object)
    """
    # REGULARISERS ==================================
    regulariser = None
    if (
        training_params['l2_reg_beta'] > 0. and
        training_params['l1_reg_beta'] > 0.
    ):
        print('L1_L2 regularisation applied')
        resgulariser = regularizers.l1_l2(l1=training_params['l1_reg_beta'],
                                          l2=training_params['l2_reg_beta'])
    elif training_params['l1_reg_beta'] > 0.:
        print('L1 regularisation applied')
        regulariser = regularizers.l2(training_params['l1_reg_beta'])
    elif training_params['l2_reg_beta'] > 0.:
        print('L2 regularisation applied')
        regulariser = regularizers.l2(training_params['l2_reg_beta'])

    # INPUTS =========================================
    inputs = Input(shape=(
        training_params['sequence_length'],) +
        tuple(config['input_shape']) + (3,)
    )
       
    # MODEL ===========================================
    # Call the ResNet50 model of keras with weights (mandatory to include the
    # top layers that need to be removed)
    cnn = ResNet50(include_top=True, weights='imagenet')
    # Take only the feature extraction layers
    end = -3
    model = Model(input=cnn.input,
                            output=cnn.layers[end].output)
    
    # Freeze layers
    upper_limit = training_params['last_layer_to_freeze_conv']
    for count, layer in enumerate(model.layers[:upper_limit]):
        layer.trainable = False
    
    # Create a model that accepts a sequence and applies the same CNN to
    # each element of the sequence
    x = TimeDistributed(model)(inputs)
    
    if training_params['add_1x1conv']:
        x = TimeDistributed(Conv2D(training_params['add_1x1conv_ch'],
                        1, 1,
                        subsample=(1,1),
                        border_mode='same',
                        activation='relu',
                        kernel_regularizer=regulariser,
                        kernel_initializer='he_normal', 
                        ),name='conv_att_1x1')(x)

    return_sequences = False
    if training_params['num_convlstms'] > 1:
        return_sequences = True
    padding = 'same'
    convlstm_input = x  
    recurrent_dropout = training_params['convlstm_recurrent_dropout_rate']  
    dropout = training_params['convlstm_dropout_rate'] 
    hidden_units = training_params['convlstm_hidden']  

    # ConvLSTM layer: (None,timesteps,7,7,channels) => 
    #                 (None,timesteps,5,5,channels)  
    # timesteps are configurable and channels depend on previous options,
    # by default the output of the network has 2048 channels  
    for i in range(training_params['num_convlstms']):
        if i+1 == training_params['num_convlstms']:
            return_sequences = False
        convlstm_output = ConvLSTM2D(hidden_units,
                                3, 
                                strides=(1, 1),
                                padding=padding,
                                activation='tanh',
                                recurrent_activation='hard_sigmoid',
                                kernel_initializer='glorot_uniform',
                                recurrent_initializer='orthogonal',
                                bias_initializer='zeros',
                                kernel_regularizer=regulariser,
                                return_sequences=return_sequences,
                                return_state=False,
                                go_backwards=False, 
                                stateful=False,
                                dropout=dropout,
                                recurrent_dropout=recurrent_dropout,
                                name='convlstm_{}'.format(i+1)
                            )(convlstm_input)
        # Apply a convolution between ConvLSTM layers
        if (
            i+1 < training_params['num_convlstms'] and
            training_params['apply_conv_betweenconvlstm']
        ):
            convlstm_output = TimeDistributed(
                                Conv2D(hidden_units,
                                        3, 3,
                                        subsample=(1,1),
                                        border_mode='same',
                                        activation='relu',
                                        kernel_regularizer=regulariser,
                                        kernel_initializer='he_normal', 
                                        ),
                                name='conv_between_convlstm'.format(i))(
                                convlstm_output)
        convlstm_input = convlstm_output

    convlstm = convlstm_output

    # Apply a convolution after the ConvLSTM
    if training_params['apply_conv_afterconvlstm']:
        convlstm = Conv2D(training_params['convlstm_hidden'],
                        3, 3,
                        subsample=(1,1),
                        border_mode='same',
                        activation='relu',
                        kernel_regularizer=regulariser,
                        kernel_initializer='he_normal', 
                        name='conv_post_convlstm')(convlstm)
    x = GlobalAveragePooling2D()(convlstm)

    if training_params['dropout_rate'] > 0.:
        dropout = Dropout(training_params['dropout_rate'])(x)
    dense = Dense(int(training_params['num_classes']),
                    kernel_regularizer=regulariser)(dropout 
                                    if training_params['dropout_rate'] > 0.
                                    else x)
    softmax = Activation('softmax')(dense)
    final_model = Model(inputs=inputs, outputs=softmax)
    return final_model