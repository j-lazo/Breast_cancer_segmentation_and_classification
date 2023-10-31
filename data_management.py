
import numpy as np
import tensorflow as tf
import random
import os
import yaml
import datetime
import tqdm


def load_dataset_from_directory(path_frames):
    output_dict = {}
    class_folders = [f for f in os.listdir(path_frames) if os.path.isdir(os.path.join(path_frames, f))]

    # read dictionary of list (cases) each list has a dictionary of frames
    for folder in tqdm.tqdm(class_folders, desc=f"Loading data"):
        path_case = os.path.join(path_frames, folder)
        list_imgs = os.listdir(path_case)
        list_imgs = [i for i in list_imgs if i.endswith('.png') and 'mask' not in i]
        dict_frames = {i: {'Frame_id': i, 'Path_img': os.path.join(path_case, i), 'class': folder} for i in list_imgs}

        output_dict = {**output_dict, **dict_frames}
        # option b with ChainMap, check which one is more efficient
        # output_dict = dict(ChainMap({}, output_dict, dict_frames))

    print(f'Dataset with {len(output_dict)} elements')
    return output_dict


def make_tf_image_dataset(dictionary_labels, batch_size=2, training_mode=False,
                          num_repeat=None, analyze_dataset=False, input_size=[224, 224]):
    list_files = list(dictionary_labels.keys())
    path_imgs = list()
    images_class = list()

    def decode_image(file_name):
        image = tf.io.read_file(file_name)
        if tf.io.is_jpeg(image):
            image = tf.io.decode_jpeg(image, channels=3)
        else:
            image = tf.image.decode_png(image, channels=3)

        return image

    def rand_degree(lower, upper):
        return random.uniform(lower, upper)

    def rotate_img(img, lower=0, upper=180):

        upper = upper * (np.pi / 180.0)  # degrees -> radian
        lower = lower * (np.pi / 180.0)
        img = tf.keras.layers.RandomRotation(rand_degree(lower, upper), fill_mode='nearest')(img)
        return img

    def parse_image(filename, input_size=input_size):

        image = decode_image(filename)
        image = tf.image.resize(image, input_size)

        if training_mode:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            # image = rotate_img(image)

        return image

    def configure_for_performance(dataset):
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        if num_repeat:
            dataset = dataset.repeat()
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    if training_mode:
        random.shuffle(list_files)

    for img_id in list_files:
        path_imgs.append(dictionary_labels[img_id]['Path_img'])
        images_class.append(dictionary_labels[img_id]['class'])

    filenames_ds = tf.data.Dataset.from_tensor_slices(path_imgs)
    images_ds = filenames_ds.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    unique_classes = list(np.unique(images_class))
    labels = [unique_classes.index(v) for v in images_class]

    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    # path_files = tf.data.Dataset.from_tensor_slices()

    if analyze_dataset:
        ds = tf.data.Dataset.zip((images_ds, labels_ds), filenames_ds)
    else:
        ds = tf.data.Dataset.zip(images_ds, labels_ds)

    if training_mode:
        ds = configure_for_performance(ds)
    else:
        ds = ds.batch(batch_size)

    print(f'TF dataset with {len(path_imgs)} elements')

    return ds


def generate_experiment_ID(name_model='', learning_rate='na', batch_size='na', backbone_model='', prediction_model=''):
    """
    Generate a ID name for the experiment considering the name of the model, the learning rate,
    the batch size, and the date of the experiment

    :param name_model: (str)
    :param learning_rate: (float)
    :param batch_size: (int)
    :param backbone_model: (str)
    :return: (str) id name
    """
    if type(learning_rate) == list:
        lr = learning_rate[0]
    else:
        lr = learning_rate

    if prediction_model == '':
        training_date_time = datetime.datetime.now()
        if backbone_model != '':
            name_mod = ''.join([name_model, '+', backbone_model])
        else:
            name_mod = name_model

        id_name = ''.join([name_mod, '_',  '_lr_', str(lr),
                          '_bs_', str(batch_size), '_',
                           training_date_time.strftime("%d_%m_%Y_%H_%M")
                           ])
    else:
        predictions_date_time = datetime.datetime.now()
        id_name = ''.join([prediction_model, '_predictions_', predictions_date_time.strftime("%d_%m_%Y_%H_%M")])

    return id_name


def check_path_exists(path, default_ext):
    name, ext = os.path.splitext(path)
    if ext == '':
        if default_ext[0] == '.':
            default_ext = default_ext[1:]
        path = name + '.' + default_ext
    return path


def save_yaml(path, data, **kwargs):
    path = check_path_exists(path, 'yml')
    with open(path, 'w') as f:
        yaml.dump(data, f, **kwargs)