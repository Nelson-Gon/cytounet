if __name__ == "__main__":
    from cytounet.model import *
    from cytounet.augmentation import *
    from cytounet.data import *
    from cytounet.post_model import *
    import argparse
    import os.path

    # Run like this
    # python scripts/sample.py -t "examples/original_data/a549" -i "images" -m "masks" -v
    # "examples/original_data/a549/validation" -l "1e-8" -s 512 -e 5 -se 50 -b 8 -tt
    # "examples/original_data/a549/test/images" -w "models/a549_test"

    # Add relevant arguments

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("-t", "--train", type=str, help="Path to train directory",
                            required=True)
    arg_parser.add_argument("-tt", "--test", type=str, help="Path to test directory",
                            required=True)
    arg_parser.add_argument("-i", "--image", type=str,
                            help="Name of train image directory. Uses the same for validation directory",
                            required=True)
    arg_parser.add_argument("-m", "--mask", type=str, help="Name of train label/mask directory",
                            required=True)
    arg_parser.add_argument("-v", "--validation", type=str, help="Path to validation directory",
                            required=True)
    arg_parser.add_argument("-b", "--batch", type=int, help="Batch Size", required=True)
    arg_parser.add_argument("-l", "--rate", type=str, help="Learning rate", required=True)
    arg_parser.add_argument("-o", "--optimizer", type=str, help="Optimizer, defaults to Adam", required=True,
                            default="Adam")
    arg_parser.add_argument("-mt", "--metric", type=str, help="Metric to use, defaults to dice_coef", required=True,
                            default="dice_coef")
    arg_parser.add_argument("-ls", "--loss", type=str, help="Loss to minimize. Defaults to dice_coef_loss",
                            required=True, default="dice_coef_loss")
    arg_parser.add_argument("-sd", "--seed", type=str, help="Seed to use for the training/prediction. Defaults to 2",
                            required=True, default=2)
    arg_parser.add_argument("-w", "--weights", type=str, help="Path to save model weights to.",
                            required=True)

    arg_parser.add_argument("-s", "--size", type=int, help="Input size eg 512 for (512,512,1)", required=True)
    arg_parser.add_argument("-e", "--epochs", type=int, help="Number of train epochs", required=True)
    arg_parser.add_argument("-se", "--steps", type=int, help="Steps per epoch", required=True)
    
    # TODO
    # Boolean with dice vs other metrics
    # Regularization
    # Control viewing of images read
    # Support different image extensions
    # Control model saving

    arguments = arg_parser.parse_args()

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
                                       save_to_dir=None, seed=arguments.seed,
                                       target_size=(arguments.size, arguments.size))
    valid_generator = generate_validation_data(arguments.batch, arguments.validation, arguments.image,
                                               arguments.mask,
                                               data_generator_args,
                                               save_to_dir=None, seed=arguments.seed,
                                               target_size=(arguments.size, arguments.size))
    use_loss = arguments.loss
    use_metric = arguments.metric
    use_custom_loss = None

    if arguments.metric == "dice_coef":
        use_loss = dice_coef_loss
        use_metric = dice_coef
        use_custom_loss = {'dice_coef': dice_coef,
                           'dice_coef_loss': dice_coef_loss}

    model = unet(optimiser=arguments.optimizer,
                 learning_rate=float(arguments.rate), input_size=(arguments.size, arguments.size, 1),
                 metrics=use_metric, loss=use_loss, use_regularizer=False)
    history = train(model_object=model, train_generator=my_generator,
                    epochs=arguments.epochs, steps_per_epoch=arguments.steps, batch_size=arguments.batch)

    save_weights_as = os.path.join(arguments.weight, ".hdf5")
    model.save(save_weights_as)
    results = predict(test_path="test", model_weights=save_weights_as, train_seed=arguments.seed,
                      custom_loss=use_custom_loss,
                      target_size=(arguments.size, arguments.size))

    show_images(x_test, results, number=10, titles=['truth', 'predicted'], figure_size=(20, 20))
