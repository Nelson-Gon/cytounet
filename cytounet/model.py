from keras.models import *
# Just need to be explicit * imports can be ambiguous.
from keras.models import load_model
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
from data import generate_test_data
from keras.regularizers import l1, l2, l1_l2
from keras.callbacks import ModelCheckpoint


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
def unet_simple(pretrained_weights=None, metrics=None, input_size=(256, 256, 1), optimiser="Adam",
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
    if metrics is None:
        metrics = ['accuracy']
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
    # model.name = model_name

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)

    return model


def unet(metrics=None, input_size=(256, 256, 1), optimiser="Adam",
         learning_rate=3e-6, loss="binary_crossentropy", model_name="unet_complex",
         regularizer="l2", regularizer_rate=1e-2):
    """
   :param regularizer_rate: Regularization rate. Defaults to 1e-2.
  :param regularizer: Regularizer to use for penalizing weights(kernel_regularizer). Supports l1 and l2.
  :param input_size: size of the first layer(input_layer). Defaults to (256, 256, 1)
  :param optimiser: Optimiser to use. One of SGD or Adam. Defaults to Adam.
  :param learning_rate: Learning rate to use with the Adam optimiser. Defaults to 3e-6
  :param loss:  Loss function to use. Defaults to binary_crossentropy
  :param metrics: Metrics to use. Defaults to accuracy
  :param model_name Defaults to unet_complex
  :return: A fairly complex unet model.
  """

    if metrics is None:
        metrics = ['accuracy']
    optimiser_list = {'Adam': Adam(lr=learning_rate), 'SGD': SGD(learning_rate=learning_rate, momentum=0.90)}
    regularizer_list = {'l1': l1(regularizer_rate), 'l2': l2(regularizer_rate)}
    use_regularizer = regularizer_list[regularizer]
    inputs = Input(input_size)
    conv1 = Conv2D(16, 3, activation='relu', kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=use_regularizer)(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(16, 3, activation='relu', kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=use_regularizer)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=use_regularizer)(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=use_regularizer)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=use_regularizer)(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=use_regularizer)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=use_regularizer)(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=use_regularizer)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, 3, activation='relu', kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=use_regularizer)(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, 3, activation='relu', kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=use_regularizer)(conv5)

    up6 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv4], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=use_regularizer)(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=use_regularizer)(conv6)

    up7 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3])
    conv7 = Conv2D(64, 2, activation='relu', kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=use_regularizer)(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=use_regularizer)(conv7)

    up8 = Conv2DTranspose(32, 3, strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2])
    conv8 = Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=use_regularizer)(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=use_regularizer)(conv8)

    up9 = Conv2DTranspose(16, 3, strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(16, 3, activation='relu', kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=use_regularizer)(up9)
    conv9 = Conv2D(16, 3, activation='relu', kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=use_regularizer)(conv9)
    outputs = Conv2D(1, 1, activation='sigmoid', padding='same')(conv9)

    model = Model(inputs=inputs, outputs=outputs, name=model_name)

    model.compile(optimizer=optimiser_list[optimiser], loss=loss, metrics=metrics)

    return model


def predict(test_path=None, model_weights=None, train_seed=None, target_size=(256, 256),
            custom_loss=None):
    """
    :type custom_loss: dict
    :param custom_loss: If using a custom loss function, provide a dict with these functions e.g. \
     {'dice_coef':dice_coef,'dice_coef_loss':dice_coef_loss}
    :param train_seed: Same seed as used in generate_train_data
    :param test_path: Path to test file
    :param model_weights: A pretrained hdf5 model
    :param target_size: Target output image size
    :return: Predictions

    """
    generated_test = generate_test_data(test_path=test_path, train_seed=train_seed, target_size=target_size)

    model = load_model(model_weights, custom_objects=custom_loss)

    return model.predict(generated_test)


def train(model_object=None, train_generator=None, steps_per_epoch=200, epochs=5, **kwargs):
    """


    :param steps_per_epoch: Steps to use for each training epoch, defaults to 200.
    :param train_generator: From generate_train_data
    :param model_object: model_object: Model object eg unet() or unet_simple()
    :param epochs: see Model.fit
    :param kwargs: Other arguments to Model.fit. Of interest is validation_data and validation_steps that can be \
    helpful when using a validation dataset. Also useful is the callbacks argument that allows you to add callbacks \
    like ModelCheckpoint which may be useful to save a given model.
    :return: A model object

    """
    return model_object.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, **kwargs)


def finetune(pretrained_weights, model_object=None, train_generator=None, steps_per_epoch=200, epochs=5,
             monitor_metric="loss", save_best_only=True, **kwargs):
    """



      :param save_best_only: Save only the best weights? Defaults to True
      :param pretrained_weights: An hdf5 file containing pretrained weights.
      :param monitor_metric: Metric to monitor. Determines if weights are saved at the end of an epoch
      :param steps_per_epoch: Steps to use for each training epoch, defaults to 200.
      :param train_generator: From generate_train_data
      :param model_object: model_object: Model object eg unet() or unet_simple()
      :param epochs: see Model.fit
      :param kwargs: Other arguments to Model.fit. Of interest is validation_data and validation_steps that can be \
      helpful when using a validation dataset. Also useful is the callbacks argument that allows you to add callbacks \
      like ModelCheckpoint which may be useful to save a given model.
      :return: A model object

      """
    chkpoint = ModelCheckpoint(pretrained_weights, monitor=monitor_metric, verbose=1, save_best_only=save_best_only)
    train(model_object, train_generator, steps_per_epoch, epochs, callbacks=[chkpoint], **kwargs)
