if __name__ == "__main__":
    from cytounet.model import *
    from cytounet.augmentation import *
    from cytounet.data import *
    from cytounet.post_model import *

    # Read images
    x_train = read_images("examples/original_data/a549/train/images")
    y_train = read_images("examples/original_data/a549/train/masks")
    x_test = read_images("test/images")

    # Data Generation
    data_generator_args = dict(rotation_range=0.1,
                               rescale=1. / 255,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.05,
                               zoom_range=0.05,
                               horizontal_flip=True,
                               fill_mode='nearest')
    my_generator = generate_train_data(8, 'examples/original_data/a549/train', 'images', 'masks', data_generator_args,
                                       save_to_dir=None, seed=2,
                                       target_size=(512, 512))
    valid_generator = generate_validation_data(8, "examples/original_data/a549/validation", "images", "masks",
                                               data_generator_args,
                                               save_to_dir=None, seed=12,
                                               target_size=(512, 512))

    # Training
    model = unet(learning_rate=1e-4, input_size=(512, 512, 1), metrics=dice_coef, loss=dice_coef_loss,
                 use_regularizer=False)
    history = train(model_object=model, train_generator=my_generator, epochs=10, steps_per_epoch=120, batch_size=8)

    # Results
    model.save("a549_scratch_github.hdf5")
    results = predict(test_path="test", model_weights="a549_scratch_github.hdf5", train_seed=12,
                      custom_loss={'dice_coef': dice_coef,
                                   'dice_coef_loss': dice_coef_loss}, target_size=(512, 512))

    show_images(x_test, results, number=10, titles=['truth', 'predicted'], figure_size=(20, 20))
