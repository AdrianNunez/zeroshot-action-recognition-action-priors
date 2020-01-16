import os
import csv
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import itertools
from matplotlib.patches import Rectangle
import keras.backend as K
from collections import Counter
import operator
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

def limit_threads(threads_number='1'):                                         
    """Limit the number of threads for a python process.                       
                                                                               
       Args:                                                                   
            threads_number (int, optional): number of threads.                 
    """                                                                        
                                                                               
    print("Python process limited to " + threads_number + " thread")           
                                                                               
    os.environ['MKL_NUM_THREADS'] = '1'                                        
    os.environ['OPENBLAS_NUM_THREADS'] = '1'                                   
    os.environ["MKL_DYNAMIC"]="FALSE";                                         
    os.environ["NUMEXPR_NUM_THREADS"]='1';                                     
    os.environ["VECLIB_MAXIMUM_THREADS"]='1';                                  
    os.environ["OMP_NUM_THREADS"] = '1';                                       
    #mkl.set_num_threads(1)   

def plot_lr_history(path, lr_history, step_size, steps_taken, epochs):
    plt.ioff()
    fig = plt.figure()
    ax = plt.figure().gca()
    legend = []
    plt.plot(lr_history)
    legend.extend(['Learning rate'])
    plt.title('Learning rate history  (1 epoch = {})'.format(2*step_size))
    plt.ylabel('Learning rate')
    plt.xlabel('Steps')

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    lgd = plt.legend(legend, bbox_to_anchor=(1.04,1), loc='upper left')
    plt.savefig(path + 'learning_rate.png', bbox_extra_artists=(lgd,),
                bbox_inches='tight')
    plt.gcf().clear()
    plt.close(fig)

def plot_class_distribution(folder, labels_by_video, classes, mode):
    dist = Counter(labels_by_video)
    ordered_dict = sorted(dist.items(), key=operator.itemgetter(1))
    keys, values = [], []
    for elem in ordered_dict:
        keys.append(elem[0])
        values.append(elem[1])
    
    plt.bar(range(len(values)), values)
    tick_marks_x = np.arange(len(values))
    tick_marks_y = np.arange(len(values))
    plt.title('Distribution of classes in {} by video'.format(mode))
    plt.xticks(tick_marks_x, sorted(classes), fontsize=4, rotation=90)
    plt.tight_layout()
    plt.ylabel('Number of videos')
    plt.xlabel('Classes')
    plt.savefig(folder + '{}_video_distribution.pdf'.format(mode), 
                bbox_inches='tight')
    plt.gcf().clear()

def plot_weights_distribution(folder, weights, classes, mode):
    plt.bar(range(len(classes)), weights)
    tick_marks_x = np.arange(len(classes))
    tick_marks_y = np.arange(len(classes))
    plt.title('Class weighting in {}'.format(mode))
    plt.xticks(tick_marks_x, classes, fontsize=4, rotation=90)
    plt.tight_layout()
    plt.ylabel('Weight value')
    plt.xlabel('Classes')
    plt.savefig(folder + '{}_weights_distribution.pdf'.format(mode),
                bbox_inches='tight')
    plt.gcf().clear()

def f1_metric(y_true, y_pred):
    """
    from: https://www.kaggle.com/applecer/use-f1-to-select-model-lstm-based
    """
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def plot_confusion_matrix(cm, classes, path, normalize=False,
                          title='Confusion matrix', cmap='coolwarm',
                          numbers=True, ticks_fontsize=4):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = np.asarray(cm, dtype=np.float32)
        for i in range(cm.shape[0]):
            if cm[i,:].sum() > 0:
                row_total = np.float(np.sum(cm[i,:]))
                for j in range(cm.shape[1]):
                    cm[i,j] = float(cm[i,j]) / float(row_total)
            else:
                cm[i,...] = np.zeros((cm.shape[1]))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks_x = np.arange(len(classes))
    tick_marks_y = np.arange(len(classes))
    plt.xticks(tick_marks_x, classes, fontsize=ticks_fontsize, rotation=90)
    plt.yticks(tick_marks_y, classes, fontsize=ticks_fontsize)

    if numbers:
        fmt = '.2f' if normalize else 'd'
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center", fontsize=2,
                    color="black")
      
    ax = plt.gca()
    for i in range(len(classes)):
        rect = Rectangle((-0.5+i, -0.5+i), 1, 1, fill=False, 
                         edgecolor='black', lw=0.2)
        ax.add_patch(rect)
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path, bbox_inches='tight')
    plt.gcf().clear()
    
def save_image(img, path):
    fig = plt.figure()
    plt.imshow(img, cmap='Greys', interpolation='nearest')
    plt.title(path)
    plt.ylabel('Height')
    plt.xlabel('Width')
    plt.colorbar()
    plt.savefig(path, bbox_inches='tight')
    plt.gcf().clear()
    plt.close(fig)

def plot_training_info(case, num_exp, metrics, save, history):
    '''
    Function to create plots for train and validation loss and accuracy
    Input:
    * case: name for the plot, an 'accuracy.png' or 'loss.png' will be concatenated after the name.
    * metrics: list of metrics to store: 'loss' and/or 'accuracy'
    * save: boolean to store the plots or only show them.
    * history: History object returned by the Keras fit function.
    '''
    # Summarise history for accuracy
    plt.ioff()
    if 'accuracy' in metrics:     
        fig = plt.figure()
        ax = plt.figure().gca()
        legend = []
        for i in range(len(history)):
            plt.plot(history[i]['acc'])
            plt.plot(history[i]['val_acc'], '-')
            legend.extend(['train run {}'.format(i), 'val run {}'.format(i)])
        plt.title('Exp {}: Model Accuracy'.format(num_exp))
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.yticks(np.arange(0, 1.0+0.1, 0.1))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        lgd = plt.legend(legend, bbox_to_anchor=(1.04,1), loc='upper left')
        if save == True:
            plt.savefig(case + '{}_accuracy.png'.format(num_exp),
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.gcf().clear()
        else:
            plt.show()
        plt.close(fig)

    # Summarise history for loss
    if 'loss' in metrics:
        fig = plt.figure()
        for i in range(len(history)):
            plt.plot(history[i]['loss'])
            plt.plot(history[i]['val_loss'], '-')
            legend.extend(['train run {}'.format(i), 'val run {}'.format(i)])
        plt.title('Exp {}: Model Loss'.format(num_exp))
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        lgd = plt.legend(legend, bbox_to_anchor=(1.04,1), loc='upper left')
        if save == True:
            plt.savefig(case + '{}_loss.png'.format(num_exp),
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.gcf().clear()
        else:
            plt.show()
        plt.close(fig)

    # Summarise history for macro f1
    if 'f1_metric' in metrics:
        fig = plt.figure()
        ax = plt.figure().gca()
        for i in range(len(history)):
            plt.plot(history[i]['f1_metric'])
            plt.plot(history[i]['val_f1_metric'], '-')
            legend.extend(['train run {}'.format(i), 'val run {}'.format(i)])
        plt.title('Exp {}: Model Macro-F1'.format(num_exp))
        plt.ylabel('Macro F1')
        plt.xlabel('Epoch')
        plt.yticks(np.arange(0, 1.0+0.1, 0.1))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        lgd = plt.legend(legend, bbox_to_anchor=(1.04,1), loc='upper left')
        if save == True:
            plt.savefig(case + '{}_f1.png'.format(num_exp),
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.gcf().clear()
        else:
            plt.show()
        plt.close(fig)

def save_history(path, history):
    np.save(path + 'acc.npy', history['acc'])
    np.save(path + 'val_acc.npy', history['val_acc'])

    np.save(path + 'loss.npy', history['loss'])
    np.save(path + 'val_loss.npy', history['val_loss'])

    np.save(path + 'f1_metric.npy', history['f1_metric'])
    np.save(path + 'val_f1_metric.npy', history['val_f1_metric'])

def load_history(run_folder, i):
    metrics = ['acc', 'val_acc', 'loss', 'val_loss',
               'f1_metric', 'val_f1_metric']
    history = dict()
    for metric in metrics:
        history[metric] = np.load(run_folder + '{}.npy'.format(metric))
    return history

def join_histories(prev, current):
    metrics = ['acc', 'val_acc', 'loss', 'f1_metric', 'val_f1_metric']
    for metric in metrics:
        prev[metric].extend(current[metric])
    return prev

def createHistory():
    '''
    Creates a history object, a dictionary with the loss and acc of training
    and validation per epoch.
    Output:
    * history
    '''
    history = dict()
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []
    history['f1_metric'] = []
    history['val_f1_metric'] = []
    return history
    
def addtoHistory(history, loss, val_loss, acc, val_acc, f1, val_f1):
    '''
    This function takes a history object (a dictionary) and adds new loss and
    accuracy values to it
    Input:
    * history: a dictionary with the loss and accuracy of training and
               validation per epoch.
    * loss, val_loss: training and validation loss in a specific epoch.
    * acc, val_acc: training and validation accuracy in a specific epoch.
    Output:
    * history: updated history object with loss, val_loss, acc and val_acc
               included.
    '''
    history['loss'].append(loss)
    history['val_loss'].append(val_loss)
    history['acc'].append(acc)
    history['val_acc'].append(val_acc)
    history['f1_metric'].append(f1)
    history['val_f1_metric'].append(val_f1)
    return history

def show_RAM():
    '''
    TODO
    '''
    values = psutil.virtual_memory()
    used = values.used / (1024*1024)
    active = values.active / (1024*1024)
    print('RAM: {}MB, {}MB'.format(used, active))

def save_results(path, mode, results_by_class, classes, run):
    with open(path + 'egtea_{}_results_by_class_run{}.csv'.format(mode,run),
             'w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Class', 'acc', 'F1 score'])
        global_acc, global_f1, total_samples = 0., 0., 0
        for i in range(len(results_by_class)):
            writer.writerow(
                [classes[i],
                '{:.2f}'.format(float(results_by_class[i][0])),
                '{:.2f}'.format(float(results_by_class[i][1])),
                '{}'.format(results_by_class[i][2])
                ]
            )
            global_acc += float(results_by_class[i][0])
            global_f1 += float(results_by_class[i][1])
            total_samples += results_by_class[i][2]
        if global_acc > 0.:
            global_acc /= len(results_by_class)
            global_f1 /= len(results_by_class)
        writer.writerow(
                ['AVERAGE', 
                '{:.2f}'.format(global_acc),
                '{:.2f}'.format(global_f1),
                '{}'.format(total_samples)]
            )

def save_in_csv(path, struct):
    with open(path,'w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        key = struct.keys()[0]
        writer.writerow(struct[key].keys())
        for key in struct.keys():
            writer.writerow(struct[key].values())

def compute_metrics(ground_truth, predictions, classes):
    accuracy, precision, recall, total = 0., 0., 0., 0.
    for p, gt in zip(predictions, ground_truth):
        if gt in classes: continue
        if p == gt:
            accuracy += 1.
        total += 1.
    return accuracy / total