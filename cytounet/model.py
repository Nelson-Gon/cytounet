from tensorflow.keras.models import *
# Just need to be explicit * imports can be ambiguous.
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as keras_backend
from data import generate_test_data
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.callbacks import ModelCheckpoint


def dice_coef(y_true, y_pred, smooth=1):
    """
    :param smooth Prevent zero division error
    :param y_true: Train ground truth
    :param y_pred: Predicted
    :return: Returns the dice coefficient
    """
    y_true_f = keras_backend.flatten(y_true)
    y_pred_f = keras_backend.flatten(y_pred)
    intersection = keras_backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (keras_backend.sum(y_true_f) + keras_backend.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    """
    :param y_true: Train ground truth
    :param y_pred: Predicted
    :return: Returns dice loss
    """
    return 1 - dice_coef(y_true, y_pred)


def apply_conv(input_tile, filters, kernel_regularizer, layered_conv: bool):
    result = Conv2D(filters, 3, activation='relu', kernel_initializer='he_normal', padding='same',
                    kernel_regularizer=kernel_regularizer)(input_tile)
    result = BatchNormalization()(result)
    if layered_conv:  # applies double-layered conv
        return Conv2D(filters, 3, activation='relu', kernel_initializer='he_normal', padding='same',
                      kernel_regularizer=kernel_regularizer)(result)
    # simple - applies single-layered conv
    return result


def unet(metrics=None, input_size=(256, 256, 1), optimiser="Adam",
         learning_rate=3e-6, loss="binary_crossentropy", model_name="unet_complex",
         use_regularizer=False, regularizer="l2", regularizer_rate=1e-2, layered_conv=True):
    """
    :param regularizer_rate: If using a regularizer, provide regularization rate here.
    :param use_regularizer: Boolean. Should kernel regularization be used? Defaults to False
    :param regularizer: One of l1 or l2 to use as the kernel regularizer
    :param input_size: size of the first layer(input_layer). Defaults to (256, 256, 1)
    :param optimiser: Optimiser to use. One of SGD or Adam. Defaults to Adam.
    :param learning_rate: Learning rate to use with the Adam optimiser. Defaults to 3e-6
    :param loss:  Loss function to use. Defaults to binary_crossentropy
    :param metrics: Metrics to use. Defaults to accuracy
    :param model_name Defaults to unet_complex
    :param layered_conv: Should double layered conv be used each stage? Defaults to True
    :return: A fairly complex unet model.
    """
    # momentum chosen based on the original paper
    if metrics is None:
        metrics = ['accuracy']
    
    optimiser_list = {'Adam': Adam(lr=learning_rate), 'SGD': SGD(
        learning_rate=learning_rate, momentum=0.90)}
    
    chosen_regularizer = None
    if use_regularizer:
        print("Using {} for kernel regularization".format(regularizer))
        regularizer_list = {
            'l1': l1(regularizer_rate), 'l2': l2(regularizer_rate)}
        chosen_regularizer = regularizer_list[regularizer]

    inputs = Input(input_size)
    stage1_conv = apply_conv(inputs, 16, chosen_regularizer, layered_conv)
    pool1 = MaxPooling2D((2, 2))(stage1_conv)
    stage2_conv = apply_conv(pool1, 32, chosen_regularizer, layered_conv)
    pool2 = MaxPooling2D((2, 2))(stage2_conv)
    stage3_conv = apply_conv(pool2, 64, chosen_regularizer, layered_conv)
    pool3 = MaxPooling2D((2, 2))(stage3_conv)
    stage4_conv = apply_conv(pool3, 128, chosen_regularizer, layered_conv)
    pool4 = MaxPooling2D(pool_size=(2, 2))(stage4_conv)
    stage5_conv = apply_conv(pool4, 256, chosen_regularizer, layered_conv)
    
    up6 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(stage5_conv)
    up6 = concatenate([up6, stage4_conv], axis=3)
    stage6_conv = apply_conv(up6, 128, chosen_regularizer, layered_conv)
    up7 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(stage6_conv)
    up7 = concatenate([up7, stage3_conv])
    stage7_conv = apply_conv(up7, 64, chosen_regularizer, layered_conv)
    up8 = Conv2DTranspose(32, 3, strides=(2, 2), padding='same')(stage7_conv)
    up8 = concatenate([up8, stage2_conv])
    stage8_conv = apply_conv(up8, 32, chosen_regularizer, layered_conv)
    up9 = Conv2DTranspose(16, 3, strides=(2, 2), padding='same')(stage8_conv)
    up9 = concatenate([up9, stage1_conv], axis=3)
    stage9_conv = apply_conv(up9, 16, chosen_regularizer, layered_conv)
    outputs = Conv2D(1, 1, activation='sigmoid', padding='same')(stage9_conv)

    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(
        optimizer=optimiser_list[optimiser], loss=loss, metrics=metrics)
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
    generated_test = generate_test_data(
        test_path=test_path, train_seed=train_seed, target_size=target_size)

    if model_weights is None:
        raise ValueError("An HDF5 file saved via model.save is required")

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
    chkpoint = ModelCheckpoint(
        pretrained_weights, monitor=monitor_metric, verbose=1, save_best_only=save_best_only)
    train(model_object, train_generator, steps_per_epoch,
          epochs, callbacks=[chkpoint], **kwargs)
