import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from data import generate_test_data


# https://stackoverflow.com/questions/49785133/keras-dice-coefficient-loss-function-is-negative-and-increasing-with-
# https://stats.stackexchange.com/questions/195006/is-the-dice-coefficient-the-same-as-
# https://stackoverflow.com/questions/52946110/u-net-low-contrast-test-images-predict-output-is-grey-box


def dice_coef(y_true, y_pred, smooth=1):
    """
    :param smooth Prevent zero division error
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
def unet_simple(pretrained_weights=None, metrics=['accuracy'], input_size=(256, 256, 1), optimiser="Adam",
                learning_rate=3e-6, loss="binary_crossentropy", model_name="unet_simple"):
    """
    :param pretrained_weights: If a pretrained model exists, provide it here for fine tuning
    :param input_size: size of the first layer(input_layer). Defaults to (256, 256, 1)
    :param optimiser: Optimiser to use. One of SGD or Adam. Defaults to Adam.
    :param learning_rate: Learning rate to use with the Adam optimiser. Defaults to 3e-6
    :param loss:  Loss function to use. Defaults to binary_crossentropy
    :param metrics: Metrics to use. Defaults to accuracy
    :param model_name Name of the model, defaults to unet_simple
    :return: A simple(r) unet model.
    """
    inputs = Input(shape=input_size)
    conv1 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv1)

    conv2 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer='he_normal')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)

    up3 = Conv2D(64, 2, activation="relu", padding="same", kernel_initializer="he_normal")(conv2)
    merge3 = concatenate([up3, conv1], axis=3)
    conv4 = Conv2D(1, 1, activation="relu")(merge3)

    model = Model(inputs=inputs, outputs=conv4, name=model_name)

    optimiser_list = {'Adam': Adam(lr=learning_rate),
                      'SGD': SGD(learning_rate=learning_rate, momentum=0.99)}

    model.compile(optimizer=optimiser_list[optimiser],
                  loss=loss, metrics=metrics)
    #model.name = model_name

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)

    return model


def unet(pretrained_weights=None, metrics=['accuracy'], input_size=(256, 256, 1), optimiser="Adam",
         learning_rate=3e-6, loss="binary_crossentropy", model_name = "unet_complex"):
    """
  :param pretrained_weights: If a pretrained model exists, provide it here for fine tuning
  :param input_size: size of the first layer(input_layer). Defaults to (256, 256, 1)
  :param optimiser: Optimiser to use. One of SGD or Adam. Defaults to Adam.
  :param learning_rate: Learning rate to use with the Adam optimiser. Defaults to 3e-6
  :param loss:  Loss function to use. Defaults to binary_crossentropy
  :param metrics: Metrics to use. Defaults to accuracy
  :param model_name Defaults to unet_complex
  :return: A failry complex unet model.
  """
    # momentum chosen based on the original paper
    optimiser_list = {'Adam': Adam(lr=learning_rate), 'SGD': SGD(learning_rate=learning_rate, momentum=0.90)}

    inputs = Input(input_size)

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    bnorm4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bnorm4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    bnorm5 = BatchNormalization()(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(bnorm5))
    merge6 = Concatenate(axis=3)([bnorm4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10, name=model_name)

    model.compile(optimizer=optimiser_list[optimiser], loss=loss, metrics=metrics)

    #model.name = model_name
    return model

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def predict(model_object=None, test_path=None, model_weights=None, image_length=None, target_size=(256, 256),
            image_suffix="tif"):
    """
    :param model_object: Model object eg unet() or unet_simple()
    :param test_path: Path to test file
    :param model_weights: A pretrained hdf5 model
    :param image_length: Number of images in test
    :param target_size: Target output image size
    :param image_suffix: Defaults to tif
    :return: Predictions

    """
    generate_test = generate_test_data(test_path=test_path, num_image=image_length, image_suffix=image_suffix,
                                       target_size=target_size)
    # init model
    model = model_object
    model.load_weights(model_weights)
    # predictions
    return model.predict(generate_test,verbose=1, batch_size=None, steps = image_length)


def train(model_object=None, train_generator=None, steps_per_epoch=200, epochs=5, save_as=None, **kwargs):
    """

    :param train_generator: From generate_train_data
    :param model_object: model_object: Model object eg unet() or unet_simple()

    :param epochs: see Model.fit
    :param save_as: If not None, saves model weights
    :param kwargs: Other arguments to Model.fit
    :return: A model object

    """
    model_object.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, **kwargs)

    if save_as is not None:
        model_object.save(save_as)


