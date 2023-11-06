import os
import tensorflow as tf
from absl import app, flags
from absl.flags import FLAGS
from utils.data_management import datasets as dam
from utils.data_management import file_management as fam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Nadam
import datetime
from utils import data_analysis as daa
from models.model_utils import *
from models.gan_classification import *
from models.simple_models import *
import time
import numpy as np
import pandas as pd
import tqdm
import shutil
import random

input_sizes_models = {'vgg16': (224, 224), 'vgg19': (224, 224), 'inception_v3': (299, 299),
                      'resnet50': (224, 224), 'resnet101': (224, 224), 'mobilenet': (224, 224),
                      'densenet121': (224, 224), 'xception': (299, 299),
                      'resnet152': (224, 224), 'densenet201': (224, 224)}


def custom_train_simple_model(name_model, path_dataset, mode='fit', backbones=['resnet101'], gan_model='checkpoint_charlie',
                              epochs=2, batch_size=1, learning_rate=0.001, path_pretrained_model=None, val_data=None,
                              test_data=None, eval_val_set=False, eval_train_set=False,
                              results_dir=os.path.join(os.getcwd(), 'results'), gpus_available=None, analyze_data=False,
                              specific_domain=None, prepare_finished_experiment=False, k_folds={None}, val_division=0.2,
                              gan_selection=None, patience=35):

    @tf.function
    def train_step(images):
        with tf.GradientTape() as tape:
            input_ims = tf.image.resize(images, input_sizes_models[backbones[0]], method='area')
            pred_teacher = teacher_model(images, training=False)
            labels = tf.argmax(pred_teacher, axis=1)
            predictions = model(input_ims, training=True)
            #loss = loss_object(y_true=labels, y_pred=predictions)
            loss = loss_object(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss_val = train_loss(loss)
        train_accuracy_val = train_accuracy(labels, predictions)
        return train_loss_val, train_accuracy_val

    @tf.function
    def valid_step(images, labels):
        # pred_teacher = teacher_model(images, training=False)
        # labels = tf.argmax(pred_teacher, axis=1)
        input_ims = tf.image.resize(images, input_sizes_models[backbones[0]], method='area')
        predictions = model(input_ims, training=False)
        v_loss = loss_object(labels, predictions)

        val_loss = valid_loss(v_loss)
        val_acc = valid_accuracy(labels, predictions)

        return val_loss, val_acc

    @tf.function
    def prediction_step(images):
        input_ims = tf.image.resize(images, input_sizes_models[backbones[0]], method='area')
        predictions = model(input_ims, training=False)
        return predictions

    unique_class = ['HGC', 'HLT', 'LGC', 'NTL']
    multioutput = True
    list_datasets = os.listdir(os.path.join(os.getcwd(), 'datasets'))
    if path_dataset in list_datasets:
        path_dataset = os.path.join(os.getcwd(), 'datasets', path_dataset)

    list_subdirs_dataset = os.listdir(path_dataset)
    # define loss and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')
    ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

    # ID name for the folder and results
    backbone_model = ''.join([name_model + '_' for name_model in backbones])
    new_results_id = dam.generate_experiment_ID(name_model=name_model, learning_rate=learning_rate,
                                                batch_size=batch_size, backbone_model=backbone_model,
                                                mode=mode, specific_domain=specific_domain)
    path_pretrained_model_name = os.path.split(os.path.normpath(path_pretrained_model))[-1]
    # the information needed for the yaml
    training_date_time = datetime.datetime.now()
    dataset_name = os.path.split(os.path.normpath(path_dataset))[-1]
    information_experiment = {'experiment folder': new_results_id,
                              'date': training_date_time.strftime("%d-%m-%Y %H:%M"),
                              'name model': 'semi_supervised_resnet101',
                              'backbone': backbone_model,
                              'domain data used': specific_domain,
                              'batch size': int(batch_size),
                              'learning rate': float(learning_rate),
                              'backbone gan': gan_model,
                              'dataset': dataset_name,
                              'teacher_model': path_pretrained_model_name}

    results_directory = ''.join([results_dir, '/', new_results_id, '/'])
    # if results experiment doesn't exists create it
    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)

    path_experiment_information = os.path.join(results_directory, 'experiment_information.yaml')
    fam.save_yaml(path_experiment_information, information_experiment)

    train_summary_writer = tf.summary.create_file_writer(os.path.join(results_directory, 'summaries', 'train'))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(results_directory, 'summaries', 'val'))

    A_img_paths = list()
    ya_train_labels = list()
    A_img_paths_val = list()
    ya_val_labels = list()
    A_img_paths_test = list()
    ya_test_labels = list()

    if 'train' in list_subdirs_dataset:
        path_train_dataset = os.path.join(path_dataset, 'train')
    else:
        path_train_dataset = path_dataset

    if 'val' in list_subdirs_dataset:
        path_val_dataset = os.path.join(path_dataset, 'val')
    else:
        path_val_dataset = val_data

    img_list_train, dictionary_labels_train = dam.load_data_from_directory(path_train_dataset, )
    img_list_val, dictionary_labels_val = dam.load_data_from_directory(path_val_dataset)

    for img_name in img_list_train:
        A_img_paths.append(dictionary_labels_train[img_name]['path_file'])
        #ya_train_labels.append(dictionary_labels_train[img_name]['img_class'])

    #unique_class = list(np.unique(ya_train_labels)).remove('nan')
    #ya_train_labels = [unique_class.index(x) for x in ya_train_labels]
    num_classes = 4

    # A_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'testA'), '*.jpg')
    # B_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'testB'), '*.jpg')

    for img_name in img_list_val:
        A_img_paths_val.append(dictionary_labels_val[img_name]['path_file'])
        #ya_val_labels.append(dictionary_labels_val[img_name]['img_class'])

    #ya_val_labels = [unique_class.index(x) for x in ya_val_labels]
    # ya_val_labels = data.convert_labels_to_vector(ya_val_labels, len(unique_class))
    # b_val_labels = data.convert_labels_to_vector(yb_val_labels, len(unique_class))

    len_dataset = len(img_list_train) // batch_size
    train_dataset, _ = dam.make_tf_dataset(img_list_train, dictionary_labels_train, batch_size,
                                                    training=True, multi_output=multioutput, custom_training=True,
                                                   ignore_labels=True, specific_domain=specific_domain)
                                                   #num_repeat=len_dataset, )

    valid_dataset, num_class = dam.make_tf_dataset(img_list_val, dictionary_labels_val, batch_size,
                                                          training=False, multi_output=multioutput,
                                                         custom_training=True, specific_domain=specific_domain, )

    # define the models
    teacher_model, _ = load_model(path_pretrained_model)
    model = simple_classifier(backbone=backbones[0], num_classes=num_classes)

    # checkpoint
    checkpoint_dir = os.path.join(results_directory, 'checkpoints')

    gan_pretrained_weights = os.path.join(os.getcwd(), 'scripts', 'models', 'weights_gans', gan_model)

    patience = patience
    wait = 0
    best = 0
    # start training

    for epoch in range(epochs):
        t = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()
        step = 0
        for x, domain in train_dataset:
            step += 1
            images = x
            train_loss_value, t_acc = train_step(images)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

            template = 'ETA: {} - epoch: {} loss: {:.5f}  acc: {:.5f}'
            print(template.format(
                round((time.time() - t) / 60, 2), epoch + 1,
                train_loss_value, float(train_accuracy.result()),
            ))

        for x, valid_labels in valid_dataset:
            valid_images = x[0]
            valid_step(valid_images, valid_labels)
            with val_summary_writer.as_default():
                tf.summary.scalar('loss', valid_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', valid_accuracy.result(), step=epoch)

        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
              "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                  epochs,
                                                                  train_loss.result(),
                                                                  train_accuracy.result(),
                                                                  valid_loss.result(),
                                                                  valid_accuracy.result()))

        # writer.flush()
        #checkpoint.save(epoch)

        wait += 1
        if epoch == 0:
            best = valid_loss.result()
        if valid_loss.result() < best:
            best = valid_loss.result()
            wait = 0
        if wait >= patience:
            print('Early stopping triggered: wait time > patience')
            break


    model_dir = os.path.join(results_directory, 'model_weights')
    os.mkdir(model_dir)
    model_dir = ''.join([model_dir, '/saved_weights'])
    model.save(filepath=model_dir, save_format='tf')
    print(f'model saved at {model_dir}')

    path_test_data = os.path.join(path_dataset, 'test', 'test_0')
    csvfile_test_director = [f for f in os.listdir(path_test_data) if f.endswith('.csv')].pop()
    dataframe_test = pd.read_csv(os.path.join(path_test_data, csvfile_test_director))
    df_test = dataframe_test.set_index('image_name').T.to_dict()

    img_list_test, dictionary_labels_test = dam.load_data_from_directory(path_test_data)
    for img_name in img_list_test:
        A_img_paths_test.append(dictionary_labels_test[img_name]['path_file'])
        ya_test_labels.append(dictionary_labels_test[img_name]['img_class'])

    ya_test_labels = [unique_class.index(x) for x in ya_test_labels]
    test_dataset, len_test_dataset = dam.make_tf_dataset(img_list_test, dictionary_labels_test, 1,
                                                          training=False, multi_output=True,
                                                         custom_training=True)

    # C = module.simple_classifier(backbone=backbone, num_classes=len(unique_class))
    # tl.Checkpoint(dict(C=C), py.join(output_dir, 'checkpoints')).restore()

    list_all_imgs = list()
    list_all_classes = list()
    list_imaging_type = list()

    # now make predictions
    i = 0
    dictionary_predictions = {}

    for j, data_batch in enumerate(tqdm.tqdm(test_dataset, desc='Making predictions of test dataset')):
        x = data_batch[0]
        img = x[0]
        pred = prediction_step(img)
        name_img = os.path.split(A_img_paths_test[i])[-1]
        prediction = list(pred.numpy()[0])
        prediction_label = unique_class[prediction.index(np.max(prediction))]
        list_all_imgs.append(name_img)
        list_all_classes.append(df_test[name_img]['tissue type'])
        list_imaging_type.append(df_test[name_img]['imaging type'])
        dictionary_predictions[name_img] = prediction
        i += 1

    header_column = ['class_' + str(i + 1) for i in range(len(unique_class))]
    x_pred = [[] for _ in range(len(unique_class))]

    header_column.insert(0, 'fname')
    header_column.append('img_domain')
    header_column.append('over all')
    header_column.append('real values')

    df = pd.DataFrame(columns=header_column)

    print(len(list_all_imgs))
    predicted_class = list()
    for img in list_all_imgs:
        if img in dictionary_predictions:
            list_results = dictionary_predictions[img]
            predicted_class.append(unique_class[list_results.index(np.max(list_results))])
            for h in range(len(list_results)):
                x_pred[h].append(list_results[h])

    for i in range(len(unique_class)):
        df['class_' + str(i + 1)] = x_pred[i]

    df['fname'] = list_all_imgs
    df['real values'] = list_all_classes
    df['img_domain'] = list_imaging_type
    df['over all'] = predicted_class

    date_time_test = datetime.datetime.now()
    output_name_csv = ''.join(
        ['predictions_', 'complete_GAN_training_', dataset_name, '_', date_time_test.strftime("%d_%m_%Y_%H_%M"),
         '_.csv'])
    path_results_csv_file = os.path.join(results_directory, output_name_csv)
    print(f'csv file with results saved: {path_results_csv_file}')
    df.to_csv(path_results_csv_file, index=False)

    if analyze_data is True:
        gt_data_file = [f for f in os.listdir(path_test_data) if f.endswith('.csv')].pop()
        path_gt_data_file = os.path.join(path_test_data, gt_data_file)
        path_experiments_dir = os.path.join(results_directory)
        daa.analyze_multiclass_experiment(path_gt_data_file, path_experiments_dir, plot_figure=True,
                                          analyze_training_history=False, k_folds=False)

    if prepare_finished_experiment:
        compressed_results = os.path.join(os.getcwd(), 'results', 'bladder_tissue_classification_v3_compressed')
        transfer_results = os.path.join(os.getcwd(), 'results', 'bladder_tissue_classification_v3_transfer')
        path_folder = results_directory
        fam.compress_files(path_folder, destination_dir=compressed_results)
        # make a temporal folder with only the files you want to compress
        experiment_folder = os.path.split(os.path.normpath(path_folder))[-1]
        temporal_folder_dir = os.path.join(os.getcwd(), 'results', 'remove_folders', experiment_folder)
        os.mkdir(temporal_folder_dir)
        # move files to temporal folder
        list_files = [f for f in os.listdir(path_folder) if f.endswith('.png') or
                      f.endswith('.yaml') or f.endswith('.csv')]
        # temporal folder to delete
        for file_name in list_files:
            shutil.copyfile(os.path.join(path_folder, file_name), os.path.join(temporal_folder_dir, file_name))
        # compress the file
        fam.compress_files(temporal_folder_dir, destination_dir=transfer_results)
        print(f'Experiment finished, results compressed and saved in {transfer_results}')


def custom_train_gan_model(name_model, path_dataset, mode='fit', backbones=['resnet101'], gan_model='checkpoint_charlie',
                              epochs=2, batch_size=1, learning_rate=0.001, path_pretrained_model=None, val_data=None,
                              test_data=None, eval_val_set=False, eval_train_set=False,
                              results_dir=os.path.join(os.getcwd(), 'results'), gpus_available=None, analyze_data=False,
                              specific_domain=None, prepare_finished_experiment=False, k_folds={None}, val_division=0.2,
                              gan_selection=None, patience=35):

    @tf.function
    def train_step(images, domains):
        with tf.GradientTape() as tape:
            pred_teacher = teacher_model(images, training=False)
            labels = tf.argmax(pred_teacher, axis=1)
            #input_ims = tf.image.resize(images, input_sizes_models[backbones[0]], method='area')
            predictions = model([images, domains], training=True)
            #loss = loss_object(y_true=labels, y_pred=predictions)
            loss = loss_object(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss_val = train_loss(loss)
        train_accuracy_val = train_accuracy(labels, predictions)
        return train_loss_val, train_accuracy_val

    @tf.function
    def valid_step(images, domains, labels):
        # pred_teacher = teacher_model(images, training=False)
        # labels = tf.argmax(pred_teacher, axis=1)
        #input_ims = tf.image.resize(images, input_sizes_models[backbones[0]], method='area')
        predictions = model([images, domains], training=False)
        v_loss = loss_object(labels, predictions)

        val_loss = valid_loss(v_loss)
        val_acc = valid_accuracy(labels, predictions)

        return val_loss, val_acc

    @tf.function
    def prediction_step(images, domains):
        #input_ims = tf.image.resize(images, input_sizes_models[backbones[0]], method='area')
        predictions = model([images, domains], training=False)
        return predictions

    unique_class = ['HGC', 'HLT', 'LGC', 'NTL']
    multioutput = True
    list_datasets = os.listdir(os.path.join(os.getcwd(), 'datasets'))
    if path_dataset in list_datasets:
        path_dataset = os.path.join(os.getcwd(), 'datasets', path_dataset)

    list_subdirs_dataset = os.listdir(path_dataset)
    # define loss and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')
    ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

    # ID name for the folder and results
    backbone_model = ''.join([name_model + '_' for name_model in backbones])
    new_results_id = dam.generate_experiment_ID(name_model=name_model, learning_rate=learning_rate,
                                                batch_size=batch_size, backbone_model=backbone_model,
                                                mode=mode, specific_domain=specific_domain)
    path_pretrained_model_name = os.path.split(os.path.normpath(path_pretrained_model))[-1]
    # the information needed for the yaml
    training_date_time = datetime.datetime.now()
    dataset_name = os.path.split(os.path.normpath(path_dataset))[-1]
    information_experiment = {'experiment folder': new_results_id,
                              'date': training_date_time.strftime("%d-%m-%Y %H:%M"),
                              'name model': name_model,
                              'backbone': backbone_model,
                              'domain data used': specific_domain,
                              'batch size': int(batch_size),
                              'learning rate': float(learning_rate),
                              'backbone gan': gan_model,
                              'dataset': dataset_name,
                              'teacher_model': path_pretrained_model_name}

    results_directory = ''.join([results_dir, '/', new_results_id, '/'])
    # if results experiment doesn't exists create it
    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)

    path_experiment_information = os.path.join(results_directory, 'experiment_information.yaml')
    fam.save_yaml(path_experiment_information, information_experiment)

    train_summary_writer = tf.summary.create_file_writer(os.path.join(results_directory, 'summaries', 'train'))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(results_directory, 'summaries', 'val'))

    A_img_paths = list()
    A_img_paths_val = list()
    A_img_paths_test = list()
    ya_test_labels = list()

    if 'train' in list_subdirs_dataset:
        path_train_dataset = os.path.join(path_dataset, 'train')
    else:
        path_train_dataset = path_dataset

    if 'val' in list_subdirs_dataset:
        path_val_dataset = os.path.join(path_dataset, 'val')
    else:
        path_val_dataset = val_data

    img_list_train, dictionary_labels_train = dam.load_data_from_directory(path_train_dataset, )
    img_list_val, dictionary_labels_val = dam.load_data_from_directory(path_val_dataset)

    for img_name in img_list_train:
        A_img_paths.append(dictionary_labels_train[img_name]['path_file'])
        #ya_train_labels.append(dictionary_labels_train[img_name]['img_class'])

    #unique_class = list(np.unique(ya_train_labels)).remove('nan')
    #ya_train_labels = [unique_class.index(x) for x in ya_train_labels]
    num_classes = 4

    for img_name in img_list_val:
        A_img_paths_val.append(dictionary_labels_val[img_name]['path_file'])
        #ya_val_labels.append(dictionary_labels_val[img_name]['img_class'])


    len_dataset = len(img_list_train) // batch_size
    train_dataset, _ = dam.make_tf_dataset(img_list_train, dictionary_labels_train, batch_size,
                                                    training=True, multi_output=multioutput, custom_training=True,
                                                   ignore_labels=True, specific_domain=specific_domain)
                                                   #num_repeat=len_dataset, )

    valid_dataset, num_class = dam.make_tf_dataset(img_list_val, dictionary_labels_val, batch_size,
                                                          training=False, multi_output=multioutput,
                                                         custom_training=True, specific_domain=specific_domain, )

    gan_pretrained_weights = gan_pretrained_weights = os.path.join(os.getcwd(), 'scripts', 'models', 'weights_gans',
                                                                   gan_model)
    # define the models
    teacher_model, _ = load_model(path_pretrained_model)
    #model = simple_classifier(backbone=backbones[0], num_classes=num_classes)
    #model = build_gan_model_separate_features(num_classes, backbones=backbones,
    #                                                          gan_weights=gan_pretrained_weights)
    if name_model == 'semi_supervised_gan_separate_features':
        model = build_gan_model_separate_features(num_classes, backbones=backbones,
                                                              gan_weights=gan_pretrained_weights)
    elif name_model == 'semi_supervised_gan_separate_features_b2':
        model = build_gan_model_separate_features_b2(num_classes, backbones=backbones,
                                                              gan_weights=gan_pretrained_weights)
    elif name_model == 'semi_supervised_gan_separate_features_b3':
        model = build_gan_model_separate_features_b3(num_classes, backbones=backbones,
                                                              gan_weights=gan_pretrained_weights)
    #model = build_only_gan_model_joint_features_and_domain(num_classes, backbones=backbones,
    #                                                          gan_weights=gan_pretrained_weights)
    # checkpoint
    #checkpoint_dir = os.path.join(results_directory, 'checkpoints')
    #checkpoint = Checkpoint(dict(C=model,
    #                                C_optimizer=optimizer,
    #                                ep_cnt=ep_cnt),
    #                           os.path.join(results_directory, 'checkpoints'),
    #                           max_to_keep=3)

    patience = patience
    wait = 0
    best = 0
    # start training

    for epoch in range(epochs):
        t = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()
        step = 0
        for x, domain in train_dataset:
            step += 1
            images = x
            train_loss_value, t_acc = train_step(images, domain)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

            template = 'ETA: {} - epoch: {} loss: {:.5f}  acc: {:.5f}'
            print(template.format(
                round((time.time() - t) / 60, 2), epoch + 1,
                train_loss_value, float(train_accuracy.result()),
            ))

        for x, valid_labels in valid_dataset:
            valid_images = x[0]
            valid_domains = x[1]
            valid_step(valid_images, valid_domains, valid_labels)
            with val_summary_writer.as_default():
                tf.summary.scalar('loss', valid_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', valid_accuracy.result(), step=epoch)

        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
              "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                  epochs,
                                                                  train_loss.result(),
                                                                  train_accuracy.result(),
                                                                  valid_loss.result(),
                                                                  valid_accuracy.result()))

        # writer.flush()
        #checkpoint.save(epoch)

        wait += 1
        if epoch == 0:
            best = valid_loss.result()
        if valid_loss.result() < best:
            best = valid_loss.result()
            wait = 0
        if wait >= patience:
            print('Early stopping triggered: wait time > patience')
            break


    model_dir = os.path.join(results_directory, 'model_weights')
    os.mkdir(model_dir)
    model_dir = ''.join([model_dir, '/saved_weights'])
    model.save(filepath=model_dir, save_format='tf')
    print(f'model saved at {model_dir}')

    path_test_data = os.path.join(path_dataset, 'test', 'test_0')
    csvfile_test_director = [f for f in os.listdir(path_test_data) if f.endswith('.csv')].pop()
    dataframe_test = pd.read_csv(os.path.join(path_test_data, csvfile_test_director))
    df_test = dataframe_test.set_index('image_name').T.to_dict()

    img_list_test, dictionary_labels_test = dam.load_data_from_directory(path_test_data)
    for img_name in img_list_test:
        A_img_paths_test.append(dictionary_labels_test[img_name]['path_file'])
        ya_test_labels.append(dictionary_labels_test[img_name]['img_class'])

    ya_test_labels = [unique_class.index(x) for x in ya_test_labels]
    test_dataset, len_test_dataset = dam.make_tf_dataset(img_list_test, dictionary_labels_test, 1,
                                                          training=False, multi_output=True,
                                                         custom_training=True)

    # C = module.simple_classifier(backbone=backbone, num_classes=len(unique_class))
    # tl.Checkpoint(dict(C=C), py.join(output_dir, 'checkpoints')).restore()

    list_all_imgs = list()
    list_all_classes = list()
    list_imaging_type = list()

    # now make predictions
    i = 0
    dictionary_predictions = {}

    for j, data_batch in enumerate(tqdm.tqdm(test_dataset, desc='Making predictions of test dataset')):
        x = data_batch[0]
        img = x[0]
        domain = x[1]
        pred = prediction_step(img, domain)
        name_img = os.path.split(A_img_paths_test[i])[-1]
        prediction = list(pred.numpy()[0])
        prediction_label = unique_class[prediction.index(np.max(prediction))]
        list_all_imgs.append(name_img)
        list_all_classes.append(df_test[name_img]['tissue type'])
        list_imaging_type.append(df_test[name_img]['imaging type'])
        dictionary_predictions[name_img] = prediction
        i += 1

    header_column = ['class_' + str(i + 1) for i in range(len(unique_class))]
    x_pred = [[] for _ in range(len(unique_class))]

    header_column.insert(0, 'fname')
    header_column.append('img_domain')
    header_column.append('over all')
    header_column.append('real values')

    df = pd.DataFrame(columns=header_column)

    print(len(list_all_imgs))
    predicted_class = list()
    for img in list_all_imgs:
        if img in dictionary_predictions:
            list_results = dictionary_predictions[img]
            predicted_class.append(unique_class[list_results.index(np.max(list_results))])
            for h in range(len(list_results)):
                x_pred[h].append(list_results[h])

    for i in range(len(unique_class)):
        df['class_' + str(i + 1)] = x_pred[i]

    df['fname'] = list_all_imgs
    df['real values'] = list_all_classes
    df['img_domain'] = list_imaging_type
    df['over all'] = predicted_class

    date_time_test = datetime.datetime.now()
    output_name_csv = ''.join(
        ['predictions_', 'complete_GAN_training_', dataset_name, '_', date_time_test.strftime("%d_%m_%Y_%H_%M"),
         '_.csv'])
    path_results_csv_file = os.path.join(results_directory, output_name_csv)
    print(f'csv file with results saved: {path_results_csv_file}')
    df.to_csv(path_results_csv_file, index=False)

    if analyze_data is True:
        gt_data_file = [f for f in os.listdir(path_test_data) if f.endswith('.csv')].pop()
        path_gt_data_file = os.path.join(path_test_data, gt_data_file)
        path_experiments_dir = os.path.join(results_directory)
        daa.analyze_multiclass_experiment(path_gt_data_file, path_experiments_dir, plot_figure=True,
                                          analyze_training_history=False, k_folds=False)

    if prepare_finished_experiment:
        compressed_results = os.path.join(os.getcwd(), 'results', 'bladder_tissue_classification_v3_compressed')
        transfer_results = os.path.join(os.getcwd(), 'results', 'bladder_tissue_classification_v3_transfer')
        path_folder = results_directory
        fam.compress_files(path_folder, destination_dir=compressed_results)
        # make a temporal folder with only the files you want to compress
        experiment_folder = os.path.split(os.path.normpath(path_folder))[-1]
        temporal_folder_dir = os.path.join(os.getcwd(), 'results', 'remove_folders', experiment_folder)
        os.mkdir(temporal_folder_dir)
        # move files to temporal folder
        list_files = [f for f in os.listdir(path_folder) if f.endswith('.png') or
                      f.endswith('.yaml') or f.endswith('.csv')]
        # temporal folder to delete
        for file_name in list_files:
            shutil.copyfile(os.path.join(path_folder, file_name), os.path.join(temporal_folder_dir, file_name))
        # compress the file
        fam.compress_files(temporal_folder_dir, destination_dir=transfer_results)
        print(f'Experiment finished, results compressed and saved in {transfer_results}')


def custom_train_gan_model_v2(name_model, path_dataset, mode='fit', backbones=['resnet101'], gan_model='checkpoint_charlie',
                              epochs=2, batch_size=1, learning_rate=0.001, path_pretrained_model=None, val_data=None,
                              test_data=None, eval_val_set=False, eval_train_set=False,
                              results_dir=os.path.join(os.getcwd(), 'results'), gpus_available=None, analyze_data=False,
                              specific_domain=None, prepare_finished_experiment=False, k_folds={None}, val_division=0.2,
                              gan_selection=None, patience=35):

    @tf.function
    def train_step(images, domains):
        with tf.GradientTape() as tape:
            pred_teacher = teacher_model([images, domains], training=False)
            labels = tf.argmax(pred_teacher, axis=1)
            #input_ims = tf.image.resize(images, input_sizes_models[backbones[0]], method='area')
            predictions = model([images, domains], training=True)
            #loss = loss_object(y_true=labels, y_pred=predictions)
            loss = loss_object(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss_val = train_loss(loss)
        train_accuracy_val = train_accuracy(labels, predictions)
        return train_loss_val, train_accuracy_val

    @tf.function
    def valid_step(images, domains, labels):
        # pred_teacher = teacher_model(images, training=False)
        # labels = tf.argmax(pred_teacher, axis=1)
        #input_ims = tf.image.resize(images, input_sizes_models[backbones[0]], method='area')
        predictions = model([images, domains], training=False)
        v_loss = loss_object(labels, predictions)

        val_loss = valid_loss(v_loss)
        val_acc = valid_accuracy(labels, predictions)

        return val_loss, val_acc

    @tf.function
    def prediction_step(images, domains):
        #input_ims = tf.image.resize(images, input_sizes_models[backbones[0]], method='area')
        predictions = model([images, domains], training=False)
        return predictions

    unique_class = ['HGC', 'HLT', 'LGC', 'NTL']
    multioutput = True
    list_datasets = os.listdir(os.path.join(os.getcwd(), 'datasets'))
    if path_dataset in list_datasets:
        path_dataset = os.path.join(os.getcwd(), 'datasets', path_dataset)

    list_subdirs_dataset = os.listdir(path_dataset)
    # define loss and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')
    ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

    # ID name for the folder and results
    backbone_model = ''.join([name_model + '_' for name_model in backbones])
    new_results_id = dam.generate_experiment_ID(name_model=name_model, learning_rate=learning_rate,
                                                batch_size=batch_size, backbone_model=backbone_model,
                                                mode=mode, specific_domain=specific_domain)
    path_pretrained_model_name = os.path.split(os.path.normpath(path_pretrained_model))[-1]
    # the information needed for the yaml
    training_date_time = datetime.datetime.now()
    dataset_name = os.path.split(os.path.normpath(path_dataset))[-1]
    information_experiment = {'experiment folder': new_results_id,
                              'date': training_date_time.strftime("%d-%m-%Y %H:%M"),
                              'name model': name_model,
                              'backbone': backbone_model,
                              'domain data used': specific_domain,
                              'batch size': int(batch_size),
                              'learning rate': float(learning_rate),
                              'backbone gan': gan_model,
                              'dataset': dataset_name,
                              'teacher_model': path_pretrained_model_name}

    results_directory = ''.join([results_dir, '/', new_results_id, '/'])
    # if results experiment doesn't exists create it
    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)

    path_experiment_information = os.path.join(results_directory, 'experiment_information.yaml')
    fam.save_yaml(path_experiment_information, information_experiment)

    train_summary_writer = tf.summary.create_file_writer(os.path.join(results_directory, 'summaries', 'train'))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(results_directory, 'summaries', 'val'))

    A_img_paths = list()
    A_img_paths_val = list()
    A_img_paths_test = list()
    ya_test_labels = list()

    if 'train' in list_subdirs_dataset:
        path_train_dataset = os.path.join(path_dataset, 'train')
    else:
        path_train_dataset = path_dataset

    if 'val' in list_subdirs_dataset:
        path_val_dataset = os.path.join(path_dataset, 'val')
    else:
        path_val_dataset = val_data

    img_list_train, dictionary_labels_train = dam.load_data_from_directory(path_train_dataset, )
    img_list_val, dictionary_labels_val = dam.load_data_from_directory(path_val_dataset)

    for img_name in img_list_train:
        A_img_paths.append(dictionary_labels_train[img_name]['path_file'])
        #ya_train_labels.append(dictionary_labels_train[img_name]['img_class'])

    #unique_class = list(np.unique(ya_train_labels)).remove('nan')
    #ya_train_labels = [unique_class.index(x) for x in ya_train_labels]
    num_classes = 4

    for img_name in img_list_val:
        A_img_paths_val.append(dictionary_labels_val[img_name]['path_file'])
        #ya_val_labels.append(dictionary_labels_val[img_name]['img_class'])


    len_dataset = len(img_list_train) // batch_size
    train_dataset, _ = dam.make_tf_dataset(img_list_train, dictionary_labels_train, batch_size,
                                                    training=True, multi_output=multioutput, custom_training=True,
                                                   ignore_labels=True, specific_domain=specific_domain)
                                                   #num_repeat=len_dataset, )

    valid_dataset, num_class = dam.make_tf_dataset(img_list_val, dictionary_labels_val, batch_size,
                                                          training=False, multi_output=multioutput,
                                                         custom_training=True, specific_domain=specific_domain, )

    gan_pretrained_weights = gan_pretrained_weights = os.path.join(os.getcwd(), 'scripts', 'models', 'weights_gans',
                                                                   gan_model)
    # define the models
    #teacher_model, _ = load_model(path_pretrained_model)
    #teacher_model = build_gan_model_separate_features(num_classes, backbones=backbones,
    #                                                          gan_weights=gan_pretrained_weights)

    teacher_model, _ = load_model(path_pretrained_model)
    #model = simple_classifier(backbone=backbones[0], num_classes=num_classes)
    model = build_gan_model_separate_features(num_classes, backbones=backbones,
                                                              gan_weights=gan_pretrained_weights)
    #model = build_only_gan_model_joint_features_and_domain(num_classes, backbones=backbones,
    #                                                          gan_weights=gan_pretrained_weights)
    # checkpoint
    #checkpoint_dir = os.path.join(results_directory, 'checkpoints')
    #checkpoint = Checkpoint(dict(C=model,
    #                                C_optimizer=optimizer,
    #                                ep_cnt=ep_cnt),
    #                           os.path.join(results_directory, 'checkpoints'),
    #                           max_to_keep=3)

    patience = patience
    wait = 0
    best = 0
    # start training

    for epoch in range(epochs):
        t = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()
        step = 0
        for x, domain in train_dataset:
            step += 1
            images = x
            train_loss_value, t_acc = train_step(images, domain)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

            template = 'ETA: {} - epoch: {} loss: {:.5f}  acc: {:.5f}'
            print(template.format(
                round((time.time() - t) / 60, 2), epoch + 1,
                train_loss_value, float(train_accuracy.result()),
            ))

        for x, valid_labels in valid_dataset:
            valid_images = x[0]
            valid_domains = x[1]
            valid_step(valid_images, valid_domains, valid_labels)
            with val_summary_writer.as_default():
                tf.summary.scalar('loss', valid_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', valid_accuracy.result(), step=epoch)

        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
              "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                  epochs,
                                                                  train_loss.result(),
                                                                  train_accuracy.result(),
                                                                  valid_loss.result(),
                                                                  valid_accuracy.result()))

        # writer.flush()
        #checkpoint.save(epoch)

        wait += 1
        if epoch == 0:
            best = valid_loss.result()
        if valid_loss.result() < best:
            best = valid_loss.result()
            wait = 0
        if wait >= patience:
            print('Early stopping triggered: wait time > patience')
            break


    model_dir = os.path.join(results_directory, 'model_weights')
    os.mkdir(model_dir)
    model_dir = ''.join([model_dir, '/saved_weights'])
    model.save(filepath=model_dir, save_format='tf')
    print(f'model saved at {model_dir}')

    path_test_data = os.path.join(path_dataset, 'test', 'test_0')
    csvfile_test_director = [f for f in os.listdir(path_test_data) if f.endswith('.csv')].pop()
    dataframe_test = pd.read_csv(os.path.join(path_test_data, csvfile_test_director))
    df_test = dataframe_test.set_index('image_name').T.to_dict()

    img_list_test, dictionary_labels_test = dam.load_data_from_directory(path_test_data)
    for img_name in img_list_test:
        A_img_paths_test.append(dictionary_labels_test[img_name]['path_file'])
        ya_test_labels.append(dictionary_labels_test[img_name]['img_class'])

    ya_test_labels = [unique_class.index(x) for x in ya_test_labels]
    test_dataset, len_test_dataset = dam.make_tf_dataset(img_list_test, dictionary_labels_test, 1,
                                                          training=False, multi_output=True,
                                                         custom_training=True)

    # C = module.simple_classifier(backbone=backbone, num_classes=len(unique_class))
    # tl.Checkpoint(dict(C=C), py.join(output_dir, 'checkpoints')).restore()

    list_all_imgs = list()
    list_all_classes = list()
    list_imaging_type = list()

    # now make predictions
    i = 0
    dictionary_predictions = {}

    for j, data_batch in enumerate(tqdm.tqdm(test_dataset, desc='Making predictions of test dataset')):
        x = data_batch[0]
        img = x[0]
        domain = x[1]
        pred = prediction_step(img, domain)
        name_img = os.path.split(A_img_paths_test[i])[-1]
        prediction = list(pred.numpy()[0])
        prediction_label = unique_class[prediction.index(np.max(prediction))]
        list_all_imgs.append(name_img)
        list_all_classes.append(df_test[name_img]['tissue type'])
        list_imaging_type.append(df_test[name_img]['imaging type'])
        dictionary_predictions[name_img] = prediction
        i += 1

    header_column = ['class_' + str(i + 1) for i in range(len(unique_class))]
    x_pred = [[] for _ in range(len(unique_class))]

    header_column.insert(0, 'fname')
    header_column.append('img_domain')
    header_column.append('over all')
    header_column.append('real values')

    df = pd.DataFrame(columns=header_column)

    print(len(list_all_imgs))
    predicted_class = list()
    for img in list_all_imgs:
        if img in dictionary_predictions:
            list_results = dictionary_predictions[img]
            predicted_class.append(unique_class[list_results.index(np.max(list_results))])
            for h in range(len(list_results)):
                x_pred[h].append(list_results[h])

    for i in range(len(unique_class)):
        df['class_' + str(i + 1)] = x_pred[i]

    df['fname'] = list_all_imgs
    df['real values'] = list_all_classes
    df['img_domain'] = list_imaging_type
    df['over all'] = predicted_class

    date_time_test = datetime.datetime.now()
    output_name_csv = ''.join(
        ['predictions_', 'complete_GAN_training_', dataset_name, '_', date_time_test.strftime("%d_%m_%Y_%H_%M"),
         '_.csv'])
    path_results_csv_file = os.path.join(results_directory, output_name_csv)
    print(f'csv file with results saved: {path_results_csv_file}')
    df.to_csv(path_results_csv_file, index=False)

    if analyze_data is True:
        gt_data_file = [f for f in os.listdir(path_test_data) if f.endswith('.csv')].pop()
        path_gt_data_file = os.path.join(path_test_data, gt_data_file)
        path_experiments_dir = os.path.join(results_directory)
        daa.analyze_multiclass_experiment(path_gt_data_file, path_experiments_dir, plot_figure=True,
                                          analyze_training_history=False, k_folds=False)

    if prepare_finished_experiment:
        compressed_results = os.path.join(os.getcwd(), 'results', 'bladder_tissue_classification_v3_compressed')
        transfer_results = os.path.join(os.getcwd(), 'results', 'bladder_tissue_classification_v3_transfer')
        path_folder = results_directory
        fam.compress_files(path_folder, destination_dir=compressed_results)
        # make a temporal folder with only the files you want to compress
        experiment_folder = os.path.split(os.path.normpath(path_folder))[-1]
        temporal_folder_dir = os.path.join(os.getcwd(), 'results', 'remove_folders', experiment_folder)
        os.mkdir(temporal_folder_dir)
        # move files to temporal folder
        list_files = [f for f in os.listdir(path_folder) if f.endswith('.png') or
                      f.endswith('.yaml') or f.endswith('.csv')]
        # temporal folder to delete
        for file_name in list_files:
            shutil.copyfile(os.path.join(path_folder, file_name), os.path.join(temporal_folder_dir, file_name))
        # compress the file
        fam.compress_files(temporal_folder_dir, destination_dir=transfer_results)
        print(f'Experiment finished, results compressed and saved in {transfer_results}')


def main(_argv):

    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs:", len(physical_devices))
    path_dataset = FLAGS.path_dataset
    name_model = FLAGS.name_model
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    results_dir = FLAGS.results_dir
    learning_rate = FLAGS.learning_rate
    analyze_data = FLAGS.analyze_data
    backbones = FLAGS.backbones
    specific_domain = FLAGS.specific_domain
    prepare_finished_experiment = FLAGS.prepare_finished_experiment
    k_folds = FLAGS.k_folds
    gan_weights = FLAGS.gan_weights
    gan_selection = FLAGS.gan_selection
    path_pretrained_model = FLAGS.path_pretrained_model
    training_mode = FLAGS.training_mode

    if training_mode == 'simple_training':
        custom_train_simple_model(name_model, path_dataset, batch_size=batch_size, gpus_available=physical_devices,
                    epochs=epochs, results_dir=results_dir, learning_rate=learning_rate, analyze_data=analyze_data,
                    backbones=backbones, specific_domain=specific_domain, gan_model=gan_weights,
                    path_pretrained_model=path_pretrained_model, prepare_finished_experiment=prepare_finished_experiment,
                    gan_selection=gan_selection)

    elif training_mode == 'gan_based_training':
        #name_model = 'semi_supervised_gan_separate_features'
        #name_model = 'semi_supervised_only_gan'
        custom_train_gan_model(name_model, path_dataset, batch_size=batch_size, gpus_available=physical_devices,
                    epochs=epochs, results_dir=results_dir, learning_rate=learning_rate, analyze_data=analyze_data,
                    backbones=backbones, specific_domain=specific_domain, gan_model=gan_weights,
                    path_pretrained_model=path_pretrained_model, prepare_finished_experiment=prepare_finished_experiment,
                    gan_selection=gan_selection)

    elif training_mode == 'gan_based_training_v2':
        print(path_pretrained_model)
        #name_model = 'semi_supervised_only_gan'
        custom_train_gan_model_v2(name_model, path_dataset, batch_size=batch_size, gpus_available=physical_devices,
                    epochs=epochs, results_dir=results_dir, learning_rate=learning_rate, analyze_data=analyze_data,
                    backbones=backbones, specific_domain=specific_domain, gan_model=gan_weights,
                    path_pretrained_model=path_pretrained_model, prepare_finished_experiment=prepare_finished_experiment,
                    gan_selection=gan_selection)


if __name__ == '__main__':

    flags.DEFINE_string('training_mode', '', 'name of the trinaing')
    flags.DEFINE_string('name_model', '', 'name of the model')
    flags.DEFINE_string('path_dataset', '', 'directory dataset')
    flags.DEFINE_string('val_dataset', '', 'directory val dataset')
    flags.DEFINE_string('test_dataset', '', 'directory test dataset')
    flags.DEFINE_integer('batch_size', 8, 'batch size')
    flags.DEFINE_integer('epochs', 1, 'epochs')
    flags.DEFINE_string('results_dir', os.path.join(os.getcwd(), 'results'), 'directory to save the results')
    flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
    flags.DEFINE_boolean('analyze_data', True, 'analyze the data after the experiment')
    flags.DEFINE_list('backbones', ['resnet101'], 'A list of the nets used as backbones: resnet101, resnet50, densenet121, vgg19')
    flags.DEFINE_string('specific_domain', None, 'In case a specific domain wants to be selected for training and validation data ops:'
                                                 '[NBI, WLI]')
    flags.DEFINE_boolean('prepare_finished_experiment', False, 'either compress the files and move them to the transfer folder or not')
    flags.DEFINE_boolean('k_folds', False, 'in case k-folds cross validation is taking palce')
    flags.DEFINE_string('gan_weights', 'checkpoint_charlie', 'weights for the generator')
    flags.DEFINE_string('gan_selection', None,
                        'In case you are using a GAN generated dataset, the options are:'
                        '[converted, reconverted, generated, all+converted, all+reconverted, all+converted_NBI]')
    flags.DEFINE_string('path_pretrained_model', None, 'path to the teacher model')

    try:
        app.run(main)
    except SystemExit:
        pass