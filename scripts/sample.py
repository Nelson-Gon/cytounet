if __name__ == "__main__":
    from cytounet.model import *
    from cytounet.augmentation import *
    from cytounet.data import *
    from cytounet.post_model import *
    import argparse
    import os.path

    # Add relevant arguments

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("-t", "--train", type=str, help="Path to train directory",
                            required=True)
    arg_parser.add_argument("-tt", "--test", type=str, help="Path to test directory",
                            required=True)
    arg_parser.add_argument("-i", "--image", type=str,
                            help="Name of train image directory. Uses the same for validation directory",
                            required=True)
    arg_parser.add_argument("-m", "--mask", type=str, path="Name of train label/mask directory",
                            required=True)
    arg_parser.add_argument("-v", "--validation", type=str, help="Path to validation directory",
                            required=True)
    arg_parser.add_argument("-l", "--rate", type=int, help="Learning rate", required=True)
    arg_parser.add_argument("-s", "--size", type=int, help="Input size eg 512 for (512,512,1)", required=True)
    arg_parser.add_argument("-e", "--epochs", type=int, help="Number of train epochs", required=True)
    arg_parser.add_argument("-se", "--steps", type=int, help="Steps per epoch", required=True)
    arg_parser.add_argument("-b", "--batch", type=int, help="Batch Size", required=True)
    # TODO
    # Boolean with dice vs other metrics
    # Regularization
    # Control viewing of images read
    # Support different image extensions
    # Control model saving

    arguments = arg_parser.parse_args()

    os.path.join("Hi","there")



    # Read images
    x_train = read_images(os.path.join(arguments.train, arguments.image))
    y_train = read_images(os.path.join(arguments.train, arguments.mask))
    x_test = read_images(arguments.test)

    # Data Generation
    data_generator_args = dict(rotation_range=0.1,
                               rescale=1. / 255,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.05,
                               zoom_range=0.05,
                               horizontal_flip=True,
                               fill_mode='nearest')
    my_generator = generate_train_data(arguments.batch, arguments.train, arguments.image, arguments.mask,
                                       data_generator_args,
                                       save_to_dir=None, seed=2,
                                       target_size=(arguments.size, arguments.size))
    valid_generator = generate_validation_data(arguments.batch, arguments.validation, arguments.image,
                                               arguments.mask,
                                               data_generator_args,
                                               save_to_dir=None, seed=12,
                                               target_size=(arguments.size, arguments.size))

    # Training
    model = unet(learning_rate=arguments.rate, input_size=(arguments.size, arguments.size, 1),
                 metrics=dice_coef, loss=dice_coef_loss, use_regularizer=False)
    history = train(model_object=model, train_generator=my_generator,
                    epochs=arguments.epochs, steps_per_epoch=arguments.steps, batch_size=arguments.batch)

    # Results
    model.save("a549_scratch_github.hdf5")
    results = predict(test_path="test", model_weights="a549_scratch_github.hdf5", train_seed=12,
                      custom_loss={'dice_coef': dice_coef,
                                   'dice_coef_loss': dice_coef_loss}, target_size=(arguments.size, arguments.size))

    show_images(x_test, results, number=10, titles=['truth', 'predicted'], figure_size=(20, 20))
