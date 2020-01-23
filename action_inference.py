import os
import json
import numpy as np
from tqdm import tqdm
from keras.models import load_model
from sklearn.metrics import (confusion_matrix, f1_score,
                             accuracy_score, precision_score,
                             recall_score)

from data import *
from model import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# ============================================================
# VARIABLES TO BE MODIFIED
# ============================================================
mode = 'test'
# Evaluate on the top-k predictions (value of k)
k_fortopk = 1
# ============================================================

custom_objects = {'f1_metric': f1_metric}

def execute_run(run,
                exp_name,
                config,
                training_params,
                path,
                mode,
                video_dict,
                verb_exp,
                object_exp,
                predictions_save_path):

    root_path = config['split_path'] + training_params['split'] + '/'

    int_to_actions,action_indices = get_classes_ordered(
        root_path + config['actions_file']
    )
    # Include a class for actions (verb and object combinations) that are not
    # in the dataset and have no label
    int_to_actions.append('other')
    other_class = len(int_to_actions)-1
    action_indices.append(other_class)

    int_to_objects,_ = get_classes_ordered(
        root_path + config['train_objects_file']
    )
    # reverse dictionary
    objects_to_int = {
        v: k for k, v in zip(range(len(int_to_objects)), int_to_objects)
    }
    int_to_verbs,_ = get_classes_ordered(root_path + config['train_verbs_file'])
    verbs_to_int = {v: k for k, v in zip(range(len(int_to_verbs)),int_to_verbs)}       
    index_to_action = dict(zip(action_indices,int_to_actions))
    action_to_int = {v: k for k, v in index_to_action.items()} 
 
    predictions, ground_truth = [], []
    # If not saved, compute and save results
    if not 'run_{}'.format(run) in video_dict:
        object_model = (config['project_folder'] +
                            config['checkpoints_folder'] +
                            '{}/best_weights_{}.h5'.format(
                                object_exp, run)
        )
        # Load object detector
        object_detector = load_model(str(object_model),
                                    custom_objects=custom_objects)

        verb_model = (config['project_folder'] +
                        config['checkpoints_folder'] +
                        '{}/best_weights_{}.h5'.format(verb_exp, run))
        # Load verb detector
        verb_detector = load_model(str(verb_model),
                                    custom_objects=custom_objects)
        
        # Load dataset
        data_file = root_path + config['{}_file'.format(mode)]
        nb_videos = num_sequences(config, training_params, mode,
                                 'evaluation', data_file) 
        # Load the video generator
        generator = load_gaze_plus_sequences(config,
                                            mode,
                                            data_file,
                                            training_params) 
        # Dictionary to save results
        video_dict['run_{}'.format(run)] = dict()
        video_dict['run_{}'.format(run)]['inference'] = dict()
        
        for i in tqdm(range(nb_videos)):
            batch_x, batch_y, video_name, length = generator.next()
            video_name = video_name[video_name.rfind('/')+1:]
            video_dict['run_{}'.format(run)]['inference'][video_name] = dict()
            video_dict['run_{}'.format(run)]['inference'][video_name][
                'length'] = length
            video_dict['run_{}'.format(run)]['inference'][video_name][
                'label'] = batch_y
        
            # Obtain object and verb softmax predictions and save them
            obj_predictions = object_detector.predict(batch_x)[0]
            verb_predictions = verb_detector.predict(batch_x)[0]
            video_dict['run_{}'.format(run)]['inference'][video_name][
                'verb_predictions'] = [float(x) for x in list(verb_predictions)]
            video_dict['run_{}'.format(run)]['inference'][video_name][
                'obj_predictions'] = [float(x) for x in list(obj_predictions)]

    # Load the correspoding prior path
    ngram = training_params['n_gram']
    if training_params['use_cookbook_prior']:
        prior_path = (
            config['project_folder'] + 
            config['cookbook_{}gram'.format(ngram)]
        )
    elif training_params['use_perfect_prior']:
        prior_path = root_path + config['perfect_prior']
    elif training_params['use_google_prior']:
        prior_path = config['project_folder'] + config['google_prior']
    elif training_params['use_phrasefinder_prior']:
        prior_path = config['project_folder'] + config['phrasefinder_prior']
    
    # Load the prior
    if (
        training_params['use_cookbook_prior'] or
        training_params['use_perfect_prior'] or
        training_params['use_google_prior'] or
        training_params['use_phrasefinder_prior']
    ):
        with open(prior_path, 'r') as json_file:
            action_prior = json.load(json_file)   

    options = [
        'use_cookbook_prior',
        'use_perfect_prior',
        'use_google_prior',
        'use_phrasefinder_prior'
    ]
    
    class_confusions = dict()
    # For each video
    for key in tqdm(video_dict['run_{}'.format(run)]['inference'].keys()):
        batch_y = video_dict['run_{}'.format(run)]['inference'][key]['label']
        ground_truth.append(batch_y)

        # Get softmax predictions
        verb_preds = video_dict['run_{}'.format(run)]['inference'][key][
            'verb_predictions']
        obj_preds = video_dict['run_{}'.format(run)]['inference'][key][
            'obj_predictions']
        verb_indices = np.argsort(verb_preds)
        obj_indices = np.argsort(obj_preds)

        actions, action_probs = [], []
        # For each verb in the dataset
        for verb in int_to_verbs:
            verb_idx = verbs_to_int[verb]
            verb_prob = verb_preds[verb_idx]

            for obj in int_to_objects:
                obj_idx = objects_to_int[obj] 
                obj_prob = obj_preds[obj_idx]

                action = verb + ' ' + obj                       

                # If any prior is used
                if (
                    training_params['use_cookbook_prior'] or
                    training_params['use_perfect_prior'] or
                    training_params['use_google_prior'] or
                    training_params['use_phrasefinder_prior']
                ):
                    action_prob = action_prior[action] 
                    prob = verb_prob * obj_prob * action_prob
                else:
                    prob = verb_prob * obj_prob
                actions.append(action)
                action_probs.append(prob)
        
        # Get the highest score actions first
        indices = np.argsort(action_probs)[::-1]
        found = False
        # Check the first 'k_fortopk' actions
        for idx in indices[:k_fortopk]:
            predicted_action = actions[idx]
            if predicted_action == int_to_actions[batch_y]:
                predictions.append(batch_y)
                found = True
        if not found:
            idx = indices[0]
            predicted_action = actions[idx]
            if predicted_action in int_to_actions:
                    predicted_action = action_to_int[actions[idx]]
                    predictions.append(predicted_action)
            else:
                predictions.append(other_class)

        # Class confusions are saved individually
        if k_fortopk == 1:
            if not int_to_actions[batch_y] in class_confusions:
                class_confusions[int_to_actions[batch_y]] = dict()
            if not 'actions' in class_confusions[int_to_actions[batch_y]]:
                class_confusions[int_to_actions[batch_y]]['actions'] = dict()
            if not 'verbs' in class_confusions[int_to_actions[batch_y]]:
                class_confusions[int_to_actions[batch_y]]['verbs'] = dict()
            if not 'objects' in class_confusions[int_to_actions[batch_y]]:
                class_confusions[int_to_actions[batch_y]]['objects'] = dict()
            if not actions[idx] in class_confusions[int_to_actions[batch_y]]['actions']:
                class_confusions[int_to_actions[batch_y]]['actions'][actions[idx]] = 0.

            idx = indices[0]
            class_confusions[int_to_actions[batch_y]]['actions'][actions[idx]] += 1
            verb = actions[idx][:actions[idx].rfind(' ')]
            if not verb in class_confusions[int_to_actions[batch_y]]['verbs']:
                class_confusions[int_to_actions[batch_y]]['verbs'][verb] = 0.
            class_confusions[int_to_actions[batch_y]]['verbs'][verb] += 1
            obj = actions[idx][actions[idx].rfind(' ')+1:]
            if not obj in class_confusions[int_to_actions[batch_y]]['objects']:
                class_confusions[int_to_actions[batch_y]]['objects'][obj] = 0.
            class_confusions[int_to_actions[batch_y]]['objects'][obj] += 1
        
    predictions = np.asarray(predictions)
    ground_truth = np.asarray(ground_truth)

    original_classes,_ = get_classes_ordered(
        root_path + config['actions_file']
    )

    accuracy_by_video = accuracy_score(
        ground_truth,
        predictions
    )

    f1_by_video = f1_score(
        ground_truth,
        predictions,
        average='macro',
        labels=original_classes
    )
    precision_by_video = precision_score(
        ground_truth,
        predictions,
        average='macro',
        labels=original_classes
    )
    recall_by_video = recall_score(
        ground_truth,
        predictions,
        average='macro',
        labels=original_classes
    )

    video_dict['run_{}'.format(run)]['evaluation'] = dict()
    video_dict['run_{}'.format(run)]['evaluation']['accuracy'] = accuracy_by_video
    video_dict['run_{}'.format(run)]['evaluation']['macro_f1'] = f1_by_video
    video_dict['run_{}'.format(run)]['evaluation']['precision'] = precision_by_video
    video_dict['run_{}'.format(run)]['evaluation']['recall'] = recall_by_video

    print(exp_name)
    print('Accuracy: {:.2f}, Macro-F1: {:.2f}'.format(
        accuracy_by_video*100, f1_by_video*100
    ))
    print('Precision: {:.2f}, Recall: {:.2f}'.format(
        precision_by_video*100, recall_by_video*100
    ))

    with open(path + 'results.txt', 'w') as f:
        f.write('Accuracy: {:.2f}, '.format(accuracy_by_video*100.) +
                'Macro-F1: {:.2f}, '.format(f1_by_video*100.) +
                'Precision: {:.2f}, '.format(precision_by_video*100.) +
                'Recall: {:.2f}'.format(recall_by_video*100.)
        )

    res = dict()
    res['accuracy'] = accuracy_by_video
    res['f1'] = f1_by_video
    res['precision'] = precision_by_video
    res['recall'] = recall_by_video

    classes,_ = get_classes_ordered(root_path + config['actions_file'])
    classes.append('other')  
    
    # Compute and save confusion matrices
    cm_by_video = confusion_matrix(
        ground_truth,
        predictions,
        labels=range(len(classes))
    )
    
    # Save class confusions
    if k_fortopk == 1:
        with open(path + 'class_confusions.csv', 'w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
            for key in class_confusions.keys():
                writer.writerow([key])
                for label in ['actions', 'verbs', 'objects']:
                    writer.writerow([label])
                    total = 0.
                    for candidate_class in class_confusions[key][label].keys():
                        total += class_confusions[key][label][candidate_class]
                    for candidate_class in class_confusions[key][label].keys():
                        preds = class_confusions[key][label][candidate_class]
                        writer.writerow([candidate_class, 
                                        preds, '{:.2f}'.format(preds/float(total))])
                writer.writerow([])

    # Plot and save confusion matrices
    plot_confusion_matrix(
        cm_by_video,
        classes, path + 'normalized_confusion_matrix_exp{}_run{}.pdf'.format(
            exp_name,run
        ),
        normalize=True,
        title='Normalized confusion matrix for {} set'.format(mode),
        cmap='coolwarm',
        numbers=False,
        ticks_fontsize=2
    )
    plot_confusion_matrix(
        cm_by_video,
        classes, path + 'confusion_matrix_exp{}_run{}.pdf'.format(exp_name,run),
        normalize=False,
        title='Confusion matrix for {} set'.format(mode),
        cmap='coolwarm',
        numbers=False,
        ticks_fontsize=2
    )

    cm2 = confusion_matrix(
        ground_truth,
        predictions,
        labels=range(len(classes))
    )

    # Save results by class
    with open(path + 'results_by_class_top{}.txt'.format(k_fortopk), 'w') as f:
        test_classes,_ = get_classes_ordered(root_path + config['test_actions_file'])
        #test_accuracies = []
        cm = np.asarray(np.copy(cm2), dtype=np.float32)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis].astype('float')
        cm[np.isnan(cm)] = 0
        diag = cm.diagonal()
        for i in range(len(classes)):
            f.write('{} - {:6f} - {} samples\n'.format(classes[i],
                    diag[i],
                    int(np.sum(cm_by_video[i,:]))
            ))

    return video_dict, res

def run_inference(config):
    global mode
    use_data_augmentation = True
    stacks_overlapping = False
    training_params = {
        'split': 'name_of_split',
        # Evaluate the zero-shot problem on actions
        'label': 'action',

        # Parameteres related to action priors
        'n_gram': 4,
        'use_cookbook_prior': False,
        'use_perfect_prior': False,
        'use_google_prior': False,
        'use_phrasefinder_prior': False,

        'oversampling': False,
        # From 0 to 1, percentage of offset at the beginning and end
        'frame_sampling_offset': 0.,
        # Number of times to run the experiments (averaged at the end)
        'runs': 3,
        # Number of timesteps
        'sequence_length': 25,
        # Number of layers to freeze, starting from 0
        # 142 to freeze everything except for the last conv block
        # None would set all layers as trainable
        # -1 to freeze all
        #'last_layer_to_freeze_conv': 142,
        # Number of hidden states used in the ConvLSTM, i.e., number of 
        # output channels
        #'num_convlstms': 1,
        #'convlstm_hidden': 256,
        #'convlstm_add_initial_state': False,
        #'apply_conv_betweenconvlstm': False,
        #'apply_conv_afterconvlstm': False,
        #'last_layer_to_freeze': 0,
        'non_uniform_sampling': False,
        # Normalise input to the ConvLSTM with L2 Normalisation
        #'convlstm_normalise_input': False,
        #'dropout_rate': 0.,
        #'convlstm_dropout_rate': 0.,
        #'convlstm_recurrent_dropout_rate': 0.,
        #'spatial_dropout_rate': 0.,
        #'use_average_pool': True,
        'use_data_augmentation': use_data_augmentation,
        'random_horizontal_flipping': True,
        'random_corner_cropping': True,
        'random_lighting': False,
        # Add a 1x1 conv after the last conv block but before the non local
        # block in order to reduce the number of channels (ch)
        #'add_1x1conv': True,
        #'add_1x1conv_ch': 256,
        'min_frames': -1,
        # Activates debug mode: inputs to the network are saved in the folder
        # pointed out by 'debug_folder' below
        'debug_mode': False,
        'debug_folder': 'debug_folder',
        'visualisation_mode': False
    }

    root_path = config['split_path'] + training_params['split'] + '/'
    training_params['label'] = 'action'
    training_params['num_classes'] = (
            len(open(root_path + config['train_actions_file'], 
                     'r').readlines())) 
    training_params['classes_file'] = root_path + config['train_actions_file']

    predictions_save_path = config['project_folder'] + 'action_predictions/'
    predictions_file = root_path + 'predictions_{}_split{}.json'.format(
        mode, training_params['split'])

    object_exp = '{}/split_{}_object_detector'.format(
        training_params['split'], training_params['split'])
    verb_exp = '{}/split_{}_verb_detector'.format(
        training_params['split'], training_params['split'])

    video_dict = dict()
    
    if os.path.exists(predictions_file):
        with open(predictions_file, 'r') as json_file:
            video_dict = json.load(json_file)        

    # Experiment setup
    options = [
        'use_cookbook_prior',
        'use_perfect_prior',
        'use_google_prior',
        'use_phrasefinder_prior'
    ]

    # Define every experiment
    experiments = []

    # Baseline
    opt_dict = dict(zip(options, [False]*len(options)))
    exp_name = 'baseline_no_text'
    experiments.append([opt_dict, exp_name])

    # Cookbook prior
    opt_dict = dict(zip(options, [False]*len(options)))
    opt_dict['use_cookbook_prior'] = True
    opt_dict['n_gram'] = 4
    exp_name = 'cookbook_prior_4gram'
    experiments.append([opt_dict, exp_name])

    # Google prior
    opt_dict = dict(zip(options, [False]*len(options)))
    opt_dict['use_google_prior'] = True
    exp_name = 'google_prior'
    experiments.append([opt_dict, exp_name])

    # Phrasefinder prior
    opt_dict = dict(zip(options, [False]*len(options)))
    opt_dict['use_phrasefinder_prior'] = True
    exp_name = 'phrasefinder_prior'
    experiments.append([opt_dict, exp_name])

    # Perfect prior
    opt_dict = dict(zip(options, [False]*len(options)))
    opt_dict['use_perfect_prior'] = True
    exp_name = 'perfect_prior'
    experiments.append([opt_dict, exp_name])
    
    # Create CSV file to save results
    csv_file = open(root_path + 'results_{}_{}_top-{}.csv'.format(
        training_params['split'], mode, k_fortopk), 'w')
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['Experiment name', 'Accuracy',
                         'Macro-F1', 'Precision', 'Recall'])

    for experiment in experiments:
        options, name = experiment
        exp_name = 'split{}_{}_{}'.format(
            training_params['split'], mode, name
        )
        plots_folder = (config['project_folder'] + 
                    config['plots_folder'] +
                    '{}/{}/'.format(training_params['split'], exp_name)
        )
        # Set up the parameters for each experiment
        for key in options.keys():
            training_params[key] = options[key]
        accuracies, f1s, precisions, recalls = [], [], [], []

        # Repeat for N runs
        for run in range(training_params['runs']):
            path = plots_folder + 'run_{}/'.format(run)
            if not os.path.exists(path):
                os.makedirs(path)

            print('Run {} ================================'.format(run+1))
            video_dict, res = execute_run(
                run,
                exp_name,
                config,
                training_params,
                path,
                mode,
                video_dict,
                verb_exp,
                object_exp,
                predictions_save_path
            )
            accuracies.append(res['accuracy'])
            f1s.append(res['f1'])
            precisions.append(res['precision'])
            recalls.append(res['recall'])

            # Save intermediate predictions
            with open(predictions_file, 'w') as f:
                json.dump(video_dict, f, indent=4)
        
        # Save results in the CSV
        csv_writer.writerow([name,
            '{:.2f} (-+{:.2f})'.format(
                np.mean(accuracies)*100., np.std(accuracies)*100.
            ),
            '{:.2f} (-+{:.2f})'.format(
                np.mean(f1s)*100., np.std(f1s)*100.
            ),
            '{:.2f} (-+{:.2f})'.format(
                np.mean(precisions)*100., np.std(precisions)*100.
            ),
            '{:.2f} (-+{:.2f})'.format(
                np.mean(recalls)*100., np.std(recalls)*100.
            )
        ])

        # Save results in txt
        with open(plots_folder + 'results_top{}.txt'.format(k_fortopk), 'w') as f:
            f.write('Mean Accuracy: {:.2f} (-+{:.2f}), '.format(
                        np.mean(accuracies)*100., np.std(accuracies)*100.
                    ) +
                    'mean Macro-F1: {:.2f} (-+{:.2f}), '.format(
                        np.mean(f1s)*100., np.std(f1s)*100.
                    ) + 
                    'mean Precision: {:.2f} (-+{:.2f}), '.format(
                        np.mean(precisions)*100., np.std(precisions)*100.
                    ) +
                    'mean Recall: {:.2f} (-+{:.2f})\n'.format(
                        np.mean(recalls)*100., np.std(recalls)*100.
                    )  
            )
    
            f.write('\nResults per run:\n')
            f.write('='*10 + '\n')
            for i in range(len(accuracies)):
                f.write('{} => Accuracy: {:.2f}, '.format(i, accuracies[i]*100.) +
                        'Macro-F1: {:.2f}, '.format(f1s[i]*100) +
                        'Precision: {:.2f}, '.format(precisions[i]*100.) + 
                        'Recall: {:.2f}\n'.format(recalls[i]*100.)
                )
    csv_file.close()

if __name__ == '__main__':
    variables_file = 'variables.json'
    with open(variables_file) as f:
        config = json.load(f)
        
    run_inference(config)
