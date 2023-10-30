import time
import os
import tensorflow as tf
from tensorflow import keras
import data_management as dam
import classification_models
import datetime
import numpy as np


def eager_train(model, train_dataset, epochs, batch_size):
    # Instantiate an optimizer to train the model.
    optimizer = keras.optimizers.SGD(learning_rate=1e-3)
    # Instantiate a loss function.
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Prepare the metrics.
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:
                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                logits = model(x_batch_train, training=True)
                # Compute the loss value for this minibatch.
                loss_value = loss_fn(y_batch_train, logits)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                # Log every 200 batches.
                if step % 200 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %s samples" % ((step + 1) * batch_size))


def custome_training(model_name, path_dataset, max_epochs, patience=15, results_directory=os.getcwd(), batch_size=2,
                     learning_rate=0.0001, results_dir=os.path.join(os.getcwd(), 'results'), backbone_network='resnet101',
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=[],
                     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)):

    dataset_dict = dam.load_dataset_from_directory(path_dataset)
    train_dataset = dam.make_tf_image_dataset(dataset_dict, training_mode=True, input_size=[224, 224],
                                          batch_size=batch_size)
    unique_classes = np.unique([dataset_dict[k]['class'] for k in dataset_dict.keys()])

    if model_name == 'simple_classifier':
        model = simple_classifier(len(unique_classes), backbone=backbone_network)
        model.summary()
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    loss_fn = loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')
    ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

    # ID name for the folder and results
    backbone_model = backbone_network
    new_results_id = dam.generate_experiment_ID(name_model=name_model, learning_rate=learning_rate,
                                                batch_size=batch_size, backbone_model=backbone_model,
                                                mode=mode, specific_domain=specific_domain)
    path_pretrained_model_name = os.path.split(os.path.normpath(path_pretrained_model))[-1]
    # the information needed for the yaml
    training_date_time = datetime.datetime.now()
    information_experiment = {'experiment folder': new_results_id,
                              'date': training_date_time.strftime("%d-%m-%Y %H:%M"),
                              'name model': 'semi_supervised_resnet101',
                              'backbone': backbone_model,
                              'batch size': int(batch_size),
                              'learning rate': float(learning_rate),
                              'teacher_model': path_pretrained_model_name}

    results_directory = ''.join([results_dir, '/', new_results_id, '/'])
    # if results experiment doesn't exists create it
    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)

    path_experiment_information = os.path.join(results_directory, 'experiment_information.yaml')
    fam.save_yaml(path_experiment_information, information_experiment)

    train_summary_writer = tf.summary.create_file_writer(os.path.join(results_directory, 'summaries', 'train'))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(results_directory, 'summaries', 'val'))

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            t_loss = loss_fn(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(t_loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss_val = train_loss(t_loss)
        train_accuracy_val = train_accuracy(labels, predictions)
        return train_loss_val, train_accuracy_val

    @tf.function
    def valid_step(images, labels):
        # pred_teacher = teacher_model(images, training=False)
        # labels = tf.argmax(pred_teacher, axis=1)
        predictions = model(images, training=False)
        v_loss = loss_fn(labels, predictions)
        val_loss = valid_loss(v_loss)
        val_acc = valid_accuracy(labels, predictions)

        return val_loss, val_acc

    @tf.function
    def prediction_step(images):
        predictions = model(images, training=False)
        return predictions

    # if path_dataset in list_datasets:
    #    path_dataset = os.path.join(os.getcwd(), 'datasets', path_dataset)

    patience = patience
    wait = 0
    # start training
    valid_dataset = train_dataset
    best_loss = 999

    for epoch in range(max_epochs):
        t = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()
        step = 0
        template = 'ETA: {} - epoch: {} loss: {:.5f}  acc: {:.5f}'
        for x, train_labels in train_dataset:
            step += 1
            images = x
            train_loss_value, t_acc = train_step(images, train_labels)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

            print(template.format(round((time.time() - t) / 60, 2), epoch + 1, train_loss_value,
                                  float(train_accuracy.result())))

        for x, valid_labels in valid_dataset:
            valid_images = x
            valid_step(valid_images, valid_labels)
            with val_summary_writer.as_default():
                tf.summary.scalar('loss', valid_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', valid_accuracy.result(), step=epoch)

        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
              "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                  max_epochs,
                                                                  train_loss.result(),
                                                                  train_accuracy.result(),
                                                                  valid_loss.result(),
                                                                  valid_accuracy.result()))

        # checkpoint.save(epoch)
        # writer.flush()

        wait += 1
        if epoch == 0:
            best_loss = valid_loss.result()
        if valid_loss.result() < best_loss:
            best_loss = valid_loss.result()
            # model.save_weights('model.h5')
            wait = 0
        if wait >= patience:
            print('Early stopping triggered: wait time > patience')
            break