import os
os.environ['PYTHONHASHSEED'] = '1'
from numpy.random import seed
seed(1)
import tensorflow as tf
tf.set_random_seed(1)
import random as rn
rn.seed(1)
import numpy as np
import cv2
import glob
import gc
import h5py
import time
import sys
import psutil
import shutil
import json
import random
from keras import backend as K
from keras.objectives import categorical_crossentropy
from keras.models import model_from_json
from keras.optimizers import Adam, SGD, RMSprop
from keras.applications import VGG16
from keras.models import Model, load_model
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from collections import Counter, OrderedDict
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import normalize
import itertools
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
from scipy.spatial.distance import euclidean
from keras.callbacks import ModelCheckpoint

from utils import *
from model import *
from data import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# ============================================================
# VARIABLES TO MODIFY
# ============================================================
variables_file = 'variables.json'
# Choose the GPU you want to use
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# ============================================================

def main(config): 
    use_data_augmentation = True
    # This is the main configuration object
    training_params = {
        # Name of the split created
        'split': 'name_of_split',
        # Label: 'verb' or 'object'
        'label': 'verb',
        # Execute a quick run: 1 batch for training and 2 videos for test
        'toy_execution': False,
        # If the evaluation is already done and saved, whether to repeat it
        'redo_evaluation': False,
        # Oversample minority classes
        'oversampling': False,
        # Warm up: do 'warmup_epochs' epochs with learnign rate 'warmup_lr'
        'use_warmup': False,
        'warmup_epochs': 3,
        'warmup_lr': 0.01,
        # From 0 to 1, percentage of offset at the beginning and end to sample
        'frame_sampling_offset': 0.,
        # Number of times to run the experiment (averaged at the end)
        'runs': 3,
        'epochs': 100,
        # Skip connection from last layer before ConvLSTM to output hidden
        # states of the ConvLSTM. It requires a 1x1 conv with a number of
        # channels equal to the number of hidden units of the ConvLSTM
        'skip_connect': False,
        # Number of timesteps
        'sequence_length': 25,
        'learning_rate': 0.0001,
        'batch_size': 16,
        # Number of layers to freeze, starting from 0
        # 142 to freeze everything except for the last conv block
        # 0 would set all layers as trainable
        # -1 to freeze all
        'last_layer_to_freeze_conv': 142,
        'optimizer': 'adam',
        # Maximum value to clip the gradient (to avoid large changes)
        'gradient_clipvalue': 0.,
        # Criterion to use for early stopping and also to choose the best model
        'stop_criterion': 'val_f1_metric',
        # Patience for early stopping
        'patience': 10,
        # Number of hidden states used in the ConvLSTM, i.e., number of 
        # output channels
        'num_convlstms': 1,
        'convlstm_hidden': 256,
        'convlstm_add_initial_state': False,
        'apply_conv_betweenconvlstm': False,
        'apply_conv_afterconvlstm': False,
        'last_layer_to_freeze': 0,
        'non_uniform_sampling': False,
        # Normalise input to the ConvLSTM with L2 Normalisation
        'convlstm_normalise_input': False,
        'dropout_rate': 0.,
        'convlstm_dropout_rate': 0.,
        'convlstm_recurrent_dropout_rate': 0.,
        'spatial_dropout_rate': 0.,
        'use_average_pool': True,
        'use_data_augmentation': use_data_augmentation,
        'random_horizontal_flipping': True,
        'random_corner_cropping': True,
        'random_lighting': False,
        # Apply class weighting in the loss function using the training set
        # class distribution as a prior
        'apply_class_weighting': True,
        # Regularisation L1/L2 in the loss. If both are used then L1_L2
        # regularisation is used (keras)
        'l1_reg_beta': 0.,
        'l2_reg_beta': 0.,
        # Add a 1x1 conv after the last conv block but before the non local
        # block in order to reduce the number of channels (ch)
        'add_1x1conv': True,
        'add_1x1conv_ch': 256,
        'min_frames': -1,
        # Activates debug mode: inputs to the network are saved in the folder
        # pointed out by 'debug_folder' below
        'debug_mode': False,
        'debug_folder': 'debug_folder',
        'visualisation_mode': False
    }

    # Name of the experiment (e.g. split_R_verb_detector)
    exp_name = 'split_{}_{}_detector'.format(
        training_params['split'], training_params['label']
    )

    root_path = config['split_path'] + training_params['split'] + '/'

    if training_params['label'] == 'verb':
        training_params['num_classes'] = (
             len(open(root_path + config['train_verbs_file'], 
                 'r').readlines())) 
        training_params['train_classes_file'] = (
            root_path + config['train_verbs_file']
        )
        training_params['val_classes_file'] = (
            root_path + config['val_verbs_file']
        )
        training_params['test_classes_file'] = (
            root_path + config['test_verbs_file']
        )

    elif training_params['label'] == 'object':
        training_params['num_classes'] = (
             len(open(root_path + config['train_objects_file'], 
                 'r').readlines())) 
        training_params['train_classes_file'] = (
            root_path + config['train_objects_file']
        )
        training_params['val_classes_file'] = (
            root_path + config['val_objects_file']
        )
        training_params['test_classes_file'] = (
            root_path + config['test_objects_file']
        )

    init_time = time.time()  

    # For reproducibility
    tf.set_random_seed(1)
    os.environ['PYTHONHASHSEED'] = '1'
    seed(1)
    rn.seed(1)
    
    # Path to folders to save plots and checkpoints
    checkpoints_folder = (config['project_folder'] +
        config['checkpoints_folder'] + 
        '{}/{}/'.format(training_params['split'], exp_name)
    )
    plots_folder = (config['project_folder'] + config['plots_folder'] +
        '{}/{}/'.format(training_params['split'], exp_name)
    )

    # Create any necessary folder
    if not os.path.exists(plots_folder): 
        os.makedirs(plots_folder)
    if not os.path.exists(checkpoints_folder): 
        os.makedirs(checkpoints_folder) 

    # Save training parameters
    with open(plots_folder + 'training_params.json', 'w') as fp:
        json.dump(training_params, fp, indent=4)
    
    # ===============================================================
    # LOAD THE DATA
    # ===============================================================

    # Compute number of videos from each set: train, validation and test
    train_file = root_path + config['train_file']
    val_file = root_path + config['val_file']
    test_file = root_path + config['test_file']
    nb_videos_train = num_sequences(config, training_params, 'train', 
                                    'training', train_file)
    nb_videos_val = num_sequences(config, training_params, 'val',
                                 'training', val_file)
    nb_videos_test = num_sequences(config, training_params, 'test',
                                   'training', test_file) 

    # Compute number of mini-batches for each set
    # Add an extra mini-batch in case that the number of samples
    # is not divisible by the mini-batch size
    nb_batches_train = nb_videos_train // training_params['batch_size']
    if nb_videos_train % training_params['batch_size'] > 0:
        nb_batches_train += 1
    nb_batches_val = nb_videos_val // training_params['batch_size']
    if nb_videos_val % training_params['batch_size'] > 0:
        nb_batches_val += 1
    nb_batches_test = nb_videos_test // training_params['batch_size']
    if nb_videos_test % training_params['batch_size'] > 0:
        nb_batches_test += 1  

    # Necessary to load the model
    custom_objects = {'f1_metric': f1_metric}
    
    if training_params['use_data_augmentation']:
        print('train: using data augmentation')

    # Instantiate the generators of batches for training and validation
    train_generator = BatchGenerator(config, 'train', train_file,
                                     training_params, nb_batches_train)
    val_generator = BatchGenerator(config, 'val', val_file, 
                                   training_params, nb_batches_val)

    total_videos = float(nb_videos_train+nb_videos_val+nb_videos_test)
    
    if training_params['apply_class_weighting']:
        print('Class weighting applied')
    print('Number of videos to => train: {}, val: {}, test: {}'.format(
        nb_videos_train, nb_videos_val, nb_videos_test)
        )
    print('% of videos to => train: {}, val: {}, test: {}'.format(
        nb_videos_train/total_videos*100, nb_videos_val/total_videos*100,
        nb_videos_test/total_videos*100)
    )
    
    if not os.path.exists(plots_folder + 'results.json'):
        all_run_results = dict()
    
    # Vectors to accumulate the results of each run
    accuracy_by_input, accuracy_by_video = dict(), dict()
    f1_by_input, f1_by_video = dict(), dict()

    # ===============================================================
    # EXECUTE THE N RUNS OF THE EXPERIMENT
    # ===============================================================
    verbose = 1
    if training_params['runs'] > 1:
        verbose = 0

    classes_val, indices_val = get_classes_ordered(
        training_params['val_classes_file']
    )
    # Compute train labels to obtain class weights (for the loss function)
    labels_by_video, _ = load_labels(
        config, training_params, 'val', val_file,
        training_params['val_classes_file']
    )
    plot_class_distribution(plots_folder, labels_by_video, classes_val, 'val')
    del labels_by_video
    gc.collect()

    classes_test, indices_test = get_classes_ordered(
        training_params['test_classes_file']
    )
    labels_by_video, _ = load_labels(
        config, training_params, 'test', test_file,
        training_params['test_classes_file']
    )
    plot_class_distribution(plots_folder, labels_by_video, classes_test, 'test')
    del labels_by_video
    gc.collect()

    classes_train, indices_train = get_classes_ordered(
        training_params['train_classes_file']
    )
    labels_by_video, indices_by_video = load_labels(
        config, training_params, 'train', 
        train_file, training_params['train_classes_file']
    )
    plot_class_distribution(plots_folder, labels_by_video,
                            classes_train, 'train')

    if training_params['apply_class_weighting']:
        class_weights = compute_class_weight('balanced',
                                             np.unique(indices_by_video),
                                             indices_by_video)
        plot_weights_distribution(plots_folder, class_weights,
                                 classes_train, 'train')
    
    histories = []
    for run in range(training_params['runs']):
        print('EXECUTING RUN {} ----------------------------------'.format(run))
        run_folder = plots_folder + 'run_{}/'.format(run)
        if not os.path.exists(run_folder):
            os.makedirs(run_folder)
        
        save_best_weights_file = (checkpoints_folder + 
            'best_weights_{}.h5'.format(run) 
        ) 
 
        if os.path.exists(plots_folder + 'results.json'):
            with open(plots_folder + 'results.json', 'r') as json_file:
                all_run_results = json.load(json_file)
        if not 'run_{}'.format(run) in all_run_results:
            all_run_results['run_{}'.format(run)] = dict()

        training_skipped = False
        # If this run has already been computed, skip training
        if not 'training_result' in all_run_results['run_{}'.format(run)]:
            model = deploy_network(config, training_params)
            
            if training_params['epochs'] > 0:
                clipvalue = None
                if training_params['gradient_clipvalue'] > 0.:
                    clipvalue = training_params['gradient_clipvalue']
                    print('Clipping gradient to {}'.format(
                        training_params['gradient_clipvalue']
                    ))

                if training_params['optimizer'] == 'adam':
                    print('Using Adam optimizer')
                    optimizer = Adam(lr=training_params['learning_rate'],
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                            clipvalue=clipvalue)
                elif training_params['optimizer'] == 'sgd':
                    print('Using SGD optimizer')
                    optimizer = SGD(lr=training_params['learning_rate'],
                                    momentum=0.9, decay=1e-6, nesterov=True, 
                                    clipvalue=clipvalue)
                elif training_params['optimizer'] == 'rmsprop':
                    print('Using RMSprop optimizer')
                    optimizer = RMSprop(lr=training_params['learning_rate'])
                
                metric_list = config['metrics'][1:] + [f1_metric]
                model.compile(optimizer=optimizer,
                              loss='categorical_crossentropy',
                              metrics=metric_list)
                model.summary()

                print('Exp {}, run {}'.format(exp_name, run))

                apply_cw = training_params['apply_class_weighting'] 
                # Optional warmup training
                if training_params['use_warmup']:
                    warmup_epochs = training_params['warmup_epochs']
                    history = model.fit_generator(generator=train_generator,
                                            validation_data=val_generator,
                                            epochs=warmup_epochs,
                                            max_queue_size=10,
                                            workers=2, 
                                            verbose=1,
                                            class_weight=(class_weights
                                                if apply_cw
                                                else None),
                                            shuffle=False,
                                            use_multiprocessing=False)

                # Type of criterion for the ModelCheckpoint and EarlyStopping
                # depending on the metric used for the EarlStopping
                if training_params['stop_criterion'] == 'val_loss':
                    mode = 'min'
                else:
                    mode = 'max'
                c = ModelCheckpoint(str(save_best_weights_file),
                                    monitor=training_params['stop_criterion'], 
                                    save_best_only=True, 
                                    save_weights_only=False, 
                                    mode=mode,
                                    period=1)
                e = EarlyStopping(monitor=training_params['stop_criterion'],
                                min_delta=0,
                                patience=training_params['patience'],
                                verbose=0,
                                mode=mode,
                                baseline=None)
                callbacks = [c, e]

                train_time = time.time()
                steps = None
                if training_params['toy_execution']:
                    steps = 1
                history = model.fit_generator(generator=train_generator,
                                            steps_per_epoch=steps,
                                            validation_data=val_generator,
                                            validation_steps=steps,
                                            epochs=training_params['epochs'],
                                            max_queue_size=10,
                                            workers=2, 
                                            verbose=1,
                                            class_weight=(class_weights
                                                if apply_cw
                                                else None),
                                            shuffle=False,
                                            use_multiprocessing=False,
                                            callbacks=callbacks)
            
                # Save the history of training 
                histories.append(history.history)
                print('TRAINING PHASE ENDED')
                metric = training_params['stop_criterion']
                # Depending on the metric to stop the training, choose whether
                # the minimum or maximum value must be chosen
                if metric == 'val_loss':
                    func = np.argmin
                else:
                    func = np.argmax
                best_epoch = func(history.history[metric])

                # Save results to the dictionary
                k1, k2 = 'run_{}'.format(run), 'training_result'
                all_run_results[k1][k2] = dict()
                all_run_results[k1][k2]['best_epoch'] = best_epoch
                all_run_results[k1][k2][
                    'best_epoch_val_loss'] = history.history[
                                                'val_loss'][best_epoch]
                all_run_results[k1][k2][
                    'best_epoch_val_acc'] = history.history[
                                                'val_acc'][best_epoch]
                all_run_results[k1][k2][
                    'best_epoch_val_f1'] = history.history[
                                                'val_f1_metric'][best_epoch]
                # Save intermediate result
                with open(plots_folder + 'results.json', 'w') as f:
                    json.dump(all_run_results, f, indent=4)
                print('Time to train for {} epochs: {}s'.format(
                    training_params['epochs'], time.time()-train_time))
        else:
            training_skipped = True
                                
        # TEST ========================
        print('='*20)
        print('TEST PHASE')
        print('training_skipped', training_skipped)
        # If training was not skipped, save the histories (loss, accuracy and
        # f1 per epoch, per run)
        if not training_skipped:
            save_history(run_folder, history.history)
            del model
        # If training was skipped, load the history of this run
        else:
            histories.append(load_history(run_folder, run))

        # Load the best model for evaluation
        model = load_model(str(save_best_weights_file),
                           custom_objects=custom_objects)
        print('Loaded the checkpoint at {}'.format(save_best_weights_file))
    
        class_list = classes_train
        print('Exp {}, run {}'.format(exp_name, run))
        res_dict = dict()
        for mode in ['train', 'val', 'test']:
            # If the evaluation was already saved and 'redo_evaluation' is not
            # set to True
            if (
                ('evaluation_{}'.format(mode) in 
                    all_run_results['run_{}'.format(run)]) and
                not training_params['redo_evaluation']
            ):
                if not mode in accuracy_by_video:
                	accuracy_by_video[mode] = []
                if not mode in f1_by_video:
                    f1_by_video[mode] = []
                k1, k2 = 'run_{}'.format(run), 'evaluation_{}'.format(mode)
                _f1_by_video = all_run_results[k1][k2]['f1']
                _accuracy_by_video = all_run_results[k1][k2]['accuracy']
                accuracy_by_video[mode].append(_accuracy_by_video/100.)
                #f1_by_input[mode].append(_f1_by_input)
                f1_by_video[mode].append(_f1_by_video/100.)
                print('{}: Accuracy per video: {}, '.format(
                        mode, _accuracy_by_video) +
                      'Macro-F1 per video: {}'.format( _f1_by_video)
                )
                continue

            if training_params['use_data_augmentation']:
                print('{}: using data augmentation'.format(mode))
            
            #if not results_loaded:
            if mode == 'train':
                if training_params['oversampling']:
                    nb_videos_train = num_sequences(config, training_params, 
                                                    'train', 'evaluation',
                                                    train_file)
                nb_videos = nb_videos_train
                generator = load_gaze_plus_sequences(config, 'train',
                                                     train_file,
                                                     training_params)
                classes_train, indices_train = get_classes_ordered(
                    training_params['train_classes_file']
                )
            elif mode == 'val':
                nb_videos = nb_videos_val
                generator = load_gaze_plus_sequences(config, 'val',
                                                     val_file,
                                                     training_params) 
                classes_train, indices_train = get_classes_ordered(
                    training_params['train_classes_file']
                )
            elif mode == 'test':
                nb_videos = nb_videos_test
                generator = load_gaze_plus_sequences(config, 'test',
                                                     test_file,
                                                     training_params) 
                classes_test, indices_test = get_classes_ordered(
                    training_params['train_classes_file']
                )
                            
            if training_params['toy_execution']:
                nb_videos = 2

            predictions_by_video, ground_truth_by_video = [], []
            length_of_videos = dict()
            predictions_by_class = []
            for _ in range(training_params['num_classes']):
                predictions_by_class.append([])
            info = dict()           
            
            # Process video by video
            print('Processing {}, {} videos'.format(mode, nb_videos))
            for i in tqdm(range(nb_videos)):
                batch_x, batch_y, video_name, length = generator.next()
                predictions = model.predict(batch_x)[0]

                # Dictionary to save results by length of video
                if not length in length_of_videos:
                    length_of_videos[length] = []

                # Save class predicted by the model and ground truth
                predicted = np.argmax(predictions,0)
                predictions_by_video.append(predicted)
                ground_truth_by_video.append(batch_y)
            
                # Save prediction by class
                predictions_by_class[batch_y].append(predicted)

                # Save results by video
                info[video_name] = dict()
                info[video_name]['ground_truth_index'] = batch_y
                info[video_name]['ground_truth_class'] = class_list[batch_y]
                info[video_name]['prediction_index'] = predicted
                info[video_name]['prediction_softmax'] = predictions[0]
                info[video_name]['prediction_class'] = class_list[predicted]
                info[video_name]['length'] = length
                info[video_name]['classes'] = class_list
            
            ground_truth_by_video = np.squeeze(np.stack(ground_truth_by_video))
            predictions_by_video = np.squeeze(np.stack(predictions_by_video))

            cm_by_video = confusion_matrix(ground_truth_by_video,
                                        predictions_by_video,
                                        labels=range(
                                            training_params['num_classes']
                                        ))
            _accuracy_by_video = accuracy_score(ground_truth_by_video,
                                                predictions_by_video)
            _f1_by_video = f1_score(ground_truth_by_video, predictions_by_video,
                                    average='macro')

            k1, k2 = 'run_{}'.format(run), 'evaluation_{}'.format(mode)
            all_run_results[k1][k2] = dict()
            all_run_results[k1][k2]['num_videos'] = nb_videos
            all_run_results[k1][k2]['accuracy'] = _accuracy_by_video*100.
            all_run_results[k1][k2]['f1'] = _f1_by_video*100.

            print('{}: Accuracy per video: {}, Macro-F1 per video: {}'.format(
                mode, _accuracy_by_video, _f1_by_video)
            )

            plot_confusion_matrix(
                cm_by_video, class_list,
                run_folder + '_normalized_by_video_{}_{}_{}.pdf'.format(
                    mode, exp_name, run),
                normalize=True,
                title='Normalized confusion matrix for {} set'.format(mode),
                cmap='coolwarm'
            )
            plot_confusion_matrix(
                cm_by_video, class_list,
                run_folder + '_by_video_{}_{}_{}.pdf'.format(
                    mode, exp_name, run),
                normalize=False, 
                title='Confusion matrix for {} set'.format(mode),
                cmap='coolwarm'
            )
          
            # Compute and save results by class                
            for i in range(len(predictions_by_class)):
                if len(predictions_by_class[i]) > 0:
                    pred = predictions_by_class[i]
                    acc = accuracy_score([i]*len(pred),pred)
                    f1 = f1_score([i]*len(pred),pred, average='macro')
                    predictions_by_class[i] = [acc, f1,
                                            len(predictions_by_class[i])]    
                else:
                    predictions_by_class[i] = [0., 0.,
                                            len(predictions_by_class[i])]
            save_results(run_folder, mode, predictions_by_class,
                         class_list, run)

            # Save general info
            save_in_csv(run_folder + '{}_run{}_evaluation_info.csv'.format(
                mode,run), info)
        
            if not mode in accuracy_by_video:
                accuracy_by_video[mode] = []
                f1_by_video[mode] = []

            accuracy_by_video[mode].append(_accuracy_by_video)
            f1_by_video[mode].append(_f1_by_video)

            del generator
            gc.collect()

            with open(plots_folder + 'results.json', 'w') as f:
                json.dump(all_run_results, f, indent=4)
        
        # END OF THE EVALUATION ===========================================
 
        del model
        gc.collect()
        K.clear_session()
        tf.set_random_seed(1)

    # END OF ALL THE RUNS ===========================================
    if not training_skipped: 
        del val_generator, train_generator
        gc.collect()
    
    plot_training_info(plots_folder,
                       exp_name, 
                       config['metrics'] + ['f1_metric'],
                       True,
                       histories)
    # ===============================================================
    # SHOW THE AVERAGED RESULTS
    # ===============================================================

    results_dict = dict()
    print('='*20)
    results_file = open(plots_folder + 'results.txt', 'w')
    for mode in ['train', 'val', 'test']:
        res_msg = '='*20 + '\n'
        res_msg += '{}: AVERAGE RESULTS OF {} RUNS\n'.format(
            mode, training_params['runs'])
        res_msg += '='*20 + '\n'

        results_dict[mode] = dict()
        results_dict[mode]['accuracy_by_video'] = accuracy_by_video[mode]
        results_dict[mode]['f1_by_video'] = f1_by_video[mode]

        res_msg += 'ACCURACY: {:.2f} (-+{:.2f}), MACRO F1: {:.2f} (-+{:.2f})\n'.format(
            np.mean(accuracy_by_video[mode])*100, np.std(accuracy_by_video[mode])*100,
            np.mean(f1_by_video[mode])*100, np.std(f1_by_video[mode])*100
            )

        res_msg += 'RESULTS PER RUN\n'
        res_msg += '----------------\n'
        res_msg += '\nAccuracy by video:\n'
        res_msg += ', '.join(str(x) for x in accuracy_by_video[mode])
        res_msg += '\nMacro F1 by video:\n'
        res_msg += ', '.join(str(x) for x in f1_by_video[mode])
        res_msg += '\n'
        print(res_msg)
        results_file.write(res_msg)
    results_file.close()

    res_msg = '\n\nTime for training and evaluation of every run: {}s'.format(time.time()-init_time)
    
    final_results = dict()
    for run in range(training_params['runs']):
        k1 = 'run_{}'.format(run)
        final_results[k1] = dict()
        for mode in ['train', 'val', 'test']:
            final_results[k1][mode] = dict()
            final_results[k1][mode]['accuracies'] = [x*100 for x in accuracy_by_video[mode]]
            final_results[k1][mode]['f1s'] = [x*100 for x in f1_by_video[mode]]
    
    with open(plots_folder + 'overall_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)
           
if __name__ == '__main__':
    with open(variables_file) as f:
        config = json.load(f)
    res = main(config)
