from absl import app, flags
from absl.flags import FLAGS
from models.building_blocks import *
from models.model_utils import *
from utils.data_management import datasets as dam
from models.simple_models import *
import keras
import os
import tqdm
import numpy as np
import pandas as pd
from utils import data_analysis as daa
from utils.data_management import file_management as fam
from matplotlib import pyplot as plt
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import scipy

import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm
from skimage.transform import resize
from keras.applications.resnet import preprocess_input

# calculate frechet inception distance


def calculate_fid(model, images1, images2):
    # calculate activations

    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)


def calculate_fid_dataset(path_dataset, gan_path, classifier_network, specific_domain=None, batch_size=1):

    fid_list = list()
    test_x, dataset_dictionary = dam.load_data_from_directory(path_dataset)
    test_dataset, _ = dam.make_tf_dataset(test_x, dataset_dictionary, batch_size=batch_size, multi_output=True,
                                          specific_domain=specific_domain)

    pretrained_backbone = build_model_fid(classifier_network)
    #pretrained_backbone = applications.resnet50(include_top=False, pooling='avg', weights='imagenet')
    #pretrained_backbone = applications.resnet.ResNet101(include_top=False, pooling='avg', weights='imagenet')

    # Generator Models
    G_A2B = ResnetGenerator(input_shape=(256, 256, 3))
    G_B2A = ResnetGenerator(input_shape=(256, 256, 3))

    Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), gan_path).restore()

    print(f'Tensorflow Dataset of {len(test_dataset)} elements found')
    for i, x in enumerate(tqdm.tqdm(test_dataset, desc='Making predictions')):
        name_file = test_x[i]
        inputs_model = x[0]
        label = x[1].numpy()
        input_classifier_1 = tf.image.resize(inputs_model[0], input_sizes_models[classifier_network], method='bilinear')

        if inputs_model[-1].numpy()[0] == 0:
            gen_img = G_A2B(inputs_model[0])
        else:
            gen_img = G_B2A(inputs_model[0])

        generated_img = tf.math.scalar_mul(255., tf.math.divide(tf.math.subtract(gen_img, tf.reduce_min(gen_img)),
                                                                   tf.math.subtract(tf.reduce_max(gen_img),
                                                                                    tf.math.reduce_min(gen_img))))
        generated_img = tf.cast(generated_img, tf.float32)
        input_classifier_2 = tf.image.resize(generated_img, input_sizes_models[classifier_network], method='bilinear')

        input_classifier_1 = get_preprocess_input_backbone(classifier_network, input_classifier_1)
        input_classifier_2 = get_preprocess_input_backbone(classifier_network, input_classifier_2)

        fid = calculate_fid(pretrained_backbone, input_classifier_1, input_classifier_2)
        fid_list.append(fid)

        # fid between act1 and act1
        #print('FID (same): %.3f' % fid1)
        # fid between act1 and act2
        #print('FID (different): %.3f' % fid2)
    return fid_list


def main(_argv):

    path_dataset = FLAGS.path_dataset
    path_gan = FLAGS.path_gan
    path_cnn = FLAGS.path_cnn

    gan_model = path_gan.split('/')[-2]
    if gan_model == 'checkpoints':
        gan_model = path_gan.split('/')[-3]

    fid_all = calculate_fid_dataset(path_dataset, path_gan, path_cnn, batch_size=7)
    fid_nbi = calculate_fid_dataset(path_dataset, path_gan, path_cnn, specific_domain='NBI', batch_size=3)
    fid_wli = calculate_fid_dataset(path_dataset, path_gan, path_cnn, specific_domain='WLI', batch_size=10)

    name_file = ''.join(['analysis_', gan_model, '.yaml'])
    path_experiment_information = os.path.join(os.getcwd(), 'results', name_file)

    information_experiment = {'GAN model': gan_model,
                              'FID ALL': float(np.mean(fid_all)),
                              'FID NBI': float(np.mean(fid_nbi)),
                              'FID WLI': float(np.mean(fid_wli))}

    fam.save_yaml(path_experiment_information, information_experiment)
    print(f'results  saved at {path_experiment_information}')

if __name__ == '__main__':

    flags.DEFINE_string('path_dataset', None, 'Select option of analysis to perform')
    flags.DEFINE_string('path_gan', None, 'Absolute path to the directory with the files needed to analyze')
    flags.DEFINE_string('path_cnn', None, 'In case only a single file is going to be analyzed')
    flags.DEFINE_string('specific_domain', None, 'In case only a single file is going to be analyzed')
    flags.DEFINE_integer('batch_size', 1, 'In case only a single file is going to be analyzed')

    try:
        app.run(main)
    except SystemExit:
        pass
