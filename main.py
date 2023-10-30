from absl import app, flags
from absl.flags import FLAGS
import tensorflow as tf
import os
import numpy as np
import data_management as dam
from training import *
import classification_models as cm

def main(_argv):

    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs:", len(physical_devices))

    path_dataset = FLAGS.path_dataset
    type_training = FLAGS.type_training
    name_model = FLAGS.name_model
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    results_dir = FLAGS.results_dir
    learning_rate = FLAGS.learning_rate
    backbone = FLAGS.backbone
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy
    metrics = ["accuracy", tf.keras.metrics.Precision(name='precision'),
               tf.keras.metrics.Recall(name='recall')]

    if type_training == 'eager_training':
        dataset_dict = dam.load_dataset_from_directory(path_dataset)
        unique_classes = np.unique([dataset_dict[k]['class'] for k in dataset_dict.keys()])
        tf_ds = dam.make_tf_image_dataset(dataset_dict, training_mode=True, input_size=[224, 224], batch_size=batch_size)
        model = cm.simple_classifier(len(unique_classes))
        model.summary()
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        eager_train(model, tf_ds, epochs, batch_size)

    elif type_training == 'custome_training':
        custome_training(name_model, path_dataset, epochs, patience=15, batch_size=batch_size, backbone_network=backbone,
                         loss=loss, metrics=metrics, optimizer=optimizer)
    else:
        print('olakease')


if __name__ == '__main__':

    flags.DEFINE_string('name_model', '', 'name of the model')
    flags.DEFINE_string('path_dataset', '', 'directory dataset')
    flags.DEFINE_string('type_training', '', 'eager_train or custome_training')
    flags.DEFINE_integer('batch_size', 8, 'batch size')
    flags.DEFINE_integer('epochs', 1, 'epochs')
    flags.DEFINE_string('results_dir', os.path.join(os.getcwd(), 'results'), 'directory to save the results')
    flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
    flags.DEFINE_boolean('analyze_data', True, 'analyze the data after the experiment')
    flags.DEFINE_string('backbone', 'resnet101', 'A list of the nets used as backbones: resnet101, resnet50, densenet121, vgg19')
    try:
        app.run(main)
    except SystemExit:
        pass