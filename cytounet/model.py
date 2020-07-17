import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

# https://stackoverflow.com/questions/49785133/keras-dice-coefficient-loss-function-is-negative-and-increasing-with-
# https://stats.stackexchange.com/questions/195006/is-the-dice-coefficient-the-same-as-
# https://stackoverflow.com/questions/52946110/u-net-low-contrast-test-images-predict-output-is-grey-box
smooth = 1


def dice_coef(y_true, y_pred):
    """

    :param y_true: Train ground truth
    :param y_pred: Predicted
    :return: Returns the dice coefficient

    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    """

       :param y_true: Train ground truth
       :param y_pred: Predicted
       :return: Returns dice loss

       """
    return 1 - dice_coef(y_true, y_pred)

# bias variance tradeoff
# https://blog.insightdatascience.com/bias-variance-tradeoff-explained-fa2bc28174c4
def unet_simple(pretrained_weights=None, input_size=(256, 256, 1),optimiser="Adam",
                learning_rate=3e-6, loss="binary_crossentropy",
         metrics=["accuracy"],dropout_rate = 0.5):
    """
    :param pretrained_weights: If a pretrained model exists, provide it here for fine tuning
    :param input_size: size of the first layer(input_layer). Defaults to (256, 256, 1)
    :param optimiser: Optimiser to use. One of SGD or Adam. Defaults to Adam.
    :param learning_rate: Learning rate to use with the Adam optimiser. Defaults to 3e-6
    :param loss:  Loss function to use. Defaults to binary_crossentropy
    :param metrics: Metrics to use. Defaults to dice_coef
    :return: A simple(r) unet model.

    """
    inputs = Input(shape=(256, 256, 1))
    conv1 = Conv2D(64, 3, activation="relu", padding="same",kernel_initializer="he_normal")(inputs)
    conv1 =BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation="relu", padding="same",kernel_initializer="he_normal")(conv1)
    conv1 =BatchNormalization()(conv1)

    conv2 = Conv2D(128, 3, activation="relu", padding="same",kernel_initializer='he_normal')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation="relu", padding="same",kernel_initializer='he_normal')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(dropout_rate)(conv2)

    up3 = Conv2D(64, 2, activation="relu", padding="same",kernel_initializer="he_normal")(conv2)
    merge3 = concatenate([up3, conv1], axis=3)
    conv4 = Conv2D(1, 1, activation="relu")(conv2)

    model = Model(inputs=inputs, outputs=conv4)

    optimiser_list = {'Adam': Adam(lr=learning_rate),
                      'SGD': SGD(learning_rate=learning_rate, momentum=0.99) }

    model.compile(optimizer=optimiser_list[optimiser],
                  loss=loss, metrics = metrics)

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)

    return model



def unet(pretrained_weights=None, input_size=(256, 256, 1),optimiser="SGD", learning_rate=1e-4, loss=dice_coef_loss,
         metrics=[dice_coef], dropout_rate = 0.2):
    """

    :param pretrained_weights: If a pretrained model exists, provide it here for fine tuning
    :param input_size: size of the first layer(input_layer). Defaults to (256, 256, 1)
    :param optimiser: Optimiser to use. One of SGD or Adam. Defaults to SGD.
    :param learning_rate: Learning rate to use with the Adam optimiser. Defaults to 1e-4
    :param loss:  Loss function to use. Defaults to dice_coef_loss
    :param metrics: Metrics to use. Defaults to dice_coef
    :return: A unet model.

    """
    # momentum chosen based on the original paper
    optimiser_list = {'Adam': Adam(lr=learning_rate), 'SGD': SGD(learning_rate=learning_rate,momentum=0.99)}
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(dropout_rate)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(dropout_rate)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, 3, activation='relu',padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(64, 2, activation='relu',padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, 3,activation='relu',  padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(2, 3,activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(1, 1,activation='relu')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=optimiser_list[optimiser], loss=loss, metrics=metrics)

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
