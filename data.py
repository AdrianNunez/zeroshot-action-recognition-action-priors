import os
import random as rn
import glob
import cv2
import gc
import time
import random
import threading
import numpy as np
from keras.utils.np_utils import to_categorical
#from scipy.misc import imresize
from utils import *
import h5py
from scipy.spatial.distance import euclidean
from scipy.stats import multivariate_normal
from PIL import Image
from tqdm import tqdm
from keras.utils import Sequence
from keras.applications.resnet50 import preprocess_input

def get_classes(ind_file):
    '''
    Returns an array with all the class names.
    -Output:
    * classes: array of size num_classes with strings.
    '''
    classes = dict()
    with open(ind_file, 'r') as f:
         content = f.readlines()
         for c in content:
            #class_name, num = c.strip().split(' ')
            pos = c.rfind(' ')
            class_name = c[:pos].strip()
            num = c[pos+1:].strip()
            classes[class_name] = int(num)-1
    return classes

def get_classes_ordered(ind_file):
    '''
    Returns an array with all the class names.
    -Output:
    * classes: array of size num_classes with strings.
    '''
    classes, indices = [], []
    with open(ind_file, 'r') as f:
         content = f.readlines()
         for c in content:
            #class_name, num = c.strip().split(' ')
            pos = c.rfind(' ')
            class_name, num = c[:pos].strip(), int(c[pos+1:].strip())
            
            classes.append(class_name)
            indices.append(num-1)
    return classes, indices

def get_action_classes(ind_file):
    '''
    Returns an array with all the class names.
    -Output:
    * classes: array of size num_classes with strings.
    '''
    classes = []
    with open(ind_file, 'r') as f:
         content = f.readlines()
         for c in content:
            pos = c.rfind(' ')
            class_name = c[:pos].strip()
            #class_name, num = c.strip().split(' ')
            classes.append(class_name)
    return classes

def load_labels(config, training_params, mode, data_file, class_file,
                load_indices=False):
    class_to_int = get_classes(class_file)
    int_to_class = {v: k for k, v in class_to_int.items()}

    labels, indices = [], []
    with open(data_file, 'r') as f:
        content = f.readlines()
        for i in range(len(content)):
            folder, action, verb, object_code = content[i].strip().split(' ')
            if training_params['label'] == 'action':
                label = action
            elif training_params['label'] == 'verb':
                label = verb
            elif training_params['label'] == 'object':
                label = object_code
            indices.append(int(label)-1)   
            labels.append(int_to_class[int(label)-1])   

    # For oversampling
    if mode == 'train' and training_params['oversampling']:  
        num_classes = training_params['num_classes']
        cnt = Counter(labels)
        classes = range(num_classes)
        class_name, ocur = cnt.most_common(1)[0]
        for i in int_to_class.values():
            for _ in range(cnt[class_name]-cnt[i]):
                labels.append(i)
                indices.append(class_to_int[i])
    return labels, indices

def num_sequences(config, training_params, mode, phase, data_file):
    '''
    Outputs the number of stacks in each set: train, val and test (set in the
    'mode' parameter). The data_file is a .txt file where the path to folders
    and labels is included.
    For validation set, include the training data file.
    Output:
    * Integer: number of stacks in the given set.
    '''
    # Used in case of oversampling
    labels = []
    nb_videos = 0
    with open(data_file, 'r') as f:
        content = f.readlines()
        for i in range(len(content)):
            folder, action, verb, object_code = content[i].strip().split(' ')
            if training_params['label'] == 'action':
                label = action
            elif training_params['label'] == 'verb':
                label = verb
            elif training_params['label'] == 'object':
                label = object_code
            labels.append(int(label)-1)
            # Check if there is a minimum of frames to select the video for
            # training, if so and it has less than the minimum, do not take
            # it into account
            if mode == 'train' and training_params['min_frames'] != -1:
                frames = glob.glob(config['images_folder'] + folder + '/img*')
                if training_params['min_frames'] >= len(frames)+1:
                    continue
            nb_videos += 1

    if (
        phase == 'training' and mode == 'train' and
        training_params['oversampling']
    ):    
        cnt = Counter(labels)
        classes = np.unique(labels)
        nb_videos = cnt.most_common(1)[0][1]*len(classes)
    return nb_videos

def oversampling(X,Y):
    _Y = np.asarray(Y)
    classes = np.unique(_Y)
    cnt = Counter(_Y)
    # Find the number of samples of the most ocurring class
    most_common_class, ocurrences = cnt.most_common(1)[0]
    for i in classes:
        if i == most_common_class: continue
        pY = np.copy(_Y)
        # Create a probability array to sample only elements from class i
        inds = np.where(pY!=int(i))
        # Sample with replacement samples of class i to match the number of
        # samples of the most ocurring class
        indices = np.random.choice(list(inds[0]), ocurrences-cnt[i])
        # Add the samples to the original array
        for j in indices:
            X.append(X[j])
            Y.append(Y[j])
    return X,Y

def prepare_dataset(config, training_params, data_file, mode, phase):
    X, Y = [], []
    with open(data_file, 'r') as f:
        content = f.readlines()
        for i in range(len(content)):
            folder, action, verb, object_code = content[i].strip().split(' ')
            if training_params['label'] == 'action':
                label = action
            elif training_params['label'] == 'verb':
                label = verb
            elif training_params['label'] == 'object':
                label = object_code            
            X.append(config['images_folder'] + folder)
            Y.append(int(label)-1)

    if (
        phase == 'training' and
        mode == 'train' and
        training_params['oversampling']
    ):
        X, Y = oversampling(X, Y)
    return X, Y

def load_sequence_from_video(config, training_params, phase,
                             mode, images, label, video_name,
                             sample_all=False):
    
    sequence, labels = [], []
    nb_elements = len(images)
    
    start, end = 0, nb_elements
    if training_params['frame_sampling_offset']:
        offset = int(np.round(
            nb_elements*training_params['frame_sampling_offset']
        ))
        start += offset
        end -= offset
        nb_elements = end-start

    # In case 'non_uniform_sampling' is True, the frame sampled from a
    # segments of the video is random and not fixed
    if (
        phase == 'training' and
        mode == 'train' and
        training_params['non_uniform_sampling']
    ):
        inds = []
        steps = np.linspace(0, nb_elements-1,   
                        training_params['sequence_length']+1,
                        dtype=np.int32) + start
        for i in range(len(steps)-1):
            inds.append(np.random.randint(steps[i], steps[i+1]))
    else:
        # Uniform sampling of frames ('sequence_length' determines the amount)
        inds = np.linspace(0, nb_elements-1,
                        training_params['sequence_length'],
                        dtype=np.int32) + start

    if sample_all:
        inds = range(nb_elements)

    # For random lighting
    val = random.uniform(0.5, 1.5)
    # For each selected frame do
    for ind in inds:
        img = cv2.imread(images[ind])
        # Random lighting
        if (
            mode == 'train' and
            phase == 'training' and
            training_params['use_data_augmentation'] and
            training_params['random_lighting']
        ): 
            img = img*val
            img[img>255] = 255
            img[img<0] = 0
        if not (
            training_params['debug_mode'] or
            training_params['visualisation_mode']
        ):
            img = preprocess_input(img)

        # Resize if necessary
        if phase == 'training' and not (
            training_params['use_data_augmentation'] and
            training_params['random_corner_cropping']
        ):
            img = cv2.resize(img, tuple(config['input_shape']))
        if (
            phase == 'evaluation' and not (
                training_params['use_data_augmentation'] and
                training_params['random_corner_cropping']
            )
        ):
            img = cv2.resize(img, tuple(config['input_shape']))

        sequence.append(img)

    labels.append(label)

    return {'sequence': sequence, 'labels': labels, 'length': nb_elements,
            'video_name': video_name, 'inds': inds}

def load_data(config, 
              element):

    images = glob.glob(config['images_folder'] +
                        element[len(config['images_folder']):] + '/img*')       
    images.sort()
    return images

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator        
def load_gaze_plus_sequences(config, mode, data_file, training_params):  
    '''
    TODO
    '''
    X, Y = prepare_dataset(config, training_params, data_file, 
                           mode, 'evaluation')
   
    while True:     
        # p contains a class index, randomized by perm
        for p in range(len(X)):
            folder, label = X[p], Y[p]
            videoname = folder[folder.rfind('/')+1:]
            images = load_data(config, folder)
            
            # Obtain sequence, label and bounding boxes for the video
            res = load_sequence_from_video(config, training_params,
                                           'evaluation', mode,
                                           images,
                                           label, folder)
            
            if training_params['debug_mode']:
                save_path = config['project_folder'] + '{}/{}/{}/'.format(
                    training_params['debug_folder'], 'eval_' + mode,
                    videoname
                )
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
         
                for i in range(len(res['sequence'])):
                    cv2.imwrite(save_path + '{}_a.jpg'.format(i),
                                res['sequence'][i])

            if (
                training_params['use_data_augmentation'] and
                training_params['random_corner_cropping']
            ):  
                
                shape = config['image_shape']
                scale = config['crop_scales'][1]
                crop_size = (int(shape[0]*scale), int(shape[1]*scale))
                center_x = shape[1] // 2
                center_y = shape[0] // 2
                box_half_x = crop_size[1] // 2
                box_half_y = crop_size[0] // 2
                x1 = center_x - box_half_x
                y1 = center_y - box_half_y
                x2 = center_x + box_half_x
                y2 = center_y + box_half_y

                for i in range(len(res['sequence'])):
                    res['sequence'][i] = cv2.resize(
                        res['sequence'][i][y1:y2,x1:x2,:],
                        tuple(config['input_shape'])
                    )
                  
            yield (np.asarray(res['sequence'])[np.newaxis,...],
                label, folder, res['length'])
            del res, images
            gc.collect()

class BatchGenerator(Sequence):
    def __init__(self, config, mode, data_file, training_params, num_batches):
        self.config = config
        self.mode = mode
        self.data_file = data_file
        self.training_params = training_params
        self.num_batches = num_batches

        self.X, self.Y = prepare_dataset(config, training_params,
                                         data_file, mode, 'training')

        if mode == 'train':
            self.perm = np.random.permutation(len(self.X))
        else:
            self.perm = range(len(self.X))

        self.num_classes = self.training_params['num_classes']

        if training_params['use_data_augmentation']:
            self.crop_positions = config['crop_positions']
            self.crop_scales = config['crop_scales']

        self.epoch_nb = 1
        
    def __len__(self):
        return self.num_batches

    def on_epoch_end(self):
        # Randomise
        if self.mode == 'train':
            self.perm = np.random.permutation(len(self.X))
        self.epoch_nb += 1

    def __getitem__(self, idx):
        minibatch_size = self.training_params['batch_size']
        low, high = idx*minibatch_size, (idx+1)*minibatch_size
        if high > len(self.X):
            high = len(self.X)
        
        inds = list(self.perm[low:high])

        if self.mode == 'train':
            if len(inds) < self.training_params['batch_size']:
                diff = self.training_params['batch_size'] - len(inds)
                for _ in range(diff):
                    inds.append(np.random.choice(len(self.X)))

        batch, batch_labels = [], []
        for i in inds:
            folder, label = self.X[i], self.Y[i]
            videoname = folder[folder.rfind('/')+1:]
            flip_prob = np.random.rand(1)
            images = load_data(self.config, folder)
            
            video_masks = None

            res = load_sequence_from_video(self.config, self.training_params,
                                         'training', self.mode, images,
                                         label, folder) 
            
            if self.training_params['debug_mode']:
                save_path = (
                    self.config['project_folder'] + '{}/{}/{}/{}/'.format(
                    self.training_params['debug_folder'], self.mode,
                    self.epoch_nb, videoname
                ))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                for i in range(len(res['sequence'])):
                    cv2.imwrite(save_path + '{}_a.jpg'.format(i),
                                res['sequence'][i])
            
            if (
                self.mode == 'train' and
                self.training_params['use_data_augmentation']
            ): 
                if self.training_params['random_horizontal_flipping']: 
                    if flip_prob > 0.5:
                        for i in range(len(res['sequence'])):
                            res['sequence'][i] = np.fliplr(res['sequence'][i])

                if self.training_params['random_corner_cropping']:
                    crop_position = self.crop_positions[
                        np.random.randint(0, len(self.crop_positions))
                    ]
                    scale = self.crop_scales[
                        np.random.randint(0, len(self.crop_scales))
                    ]
                    min_length = min(self.config['input_shape'])
                    shape = self.config['image_shape']
                    crop_size = (int(shape[0]*scale), int(shape[1]*scale))
                    if crop_position == 'c':
                        center_x = shape[1] // 2
                        center_y = shape[0] // 2
                        box_half_x = crop_size[1] // 2
                        box_half_y = crop_size[0] // 2
                        x1 = center_x - box_half_x
                        y1 = center_y - box_half_y
                        x2 = center_x + box_half_x
                        y2 = center_y + box_half_y
                    elif crop_position == 'tl':
                        x1 = 0
                        y1 = 0
                        x2 = crop_size[1]
                        y2 = crop_size[0]
                    elif crop_position == 'tr':
                        x1 = shape[1] - crop_size[1]
                        y1 = 1
                        x2 = shape[1]
                        y2 = crop_size[0]
                    elif crop_position == 'bl':
                        x1 = 0
                        y1 = shape[0] - crop_size[0]
                        x2 = crop_size[1]
                        y2 = shape[0]
                    elif crop_position == 'br':
                        x1 = shape[1] - crop_size[1]
                        y1 = shape[0] - crop_size[0]
                        x2 = shape[1]
                        y2 = shape[0]

                    for i in range(len(res['sequence'])):
                        res['sequence'][i] = cv2.resize(
                            res['sequence'][i][y1:y2,x1:x2,:],
                            tuple(self.config['input_shape'])
                        )
                        
                        if self.training_params['debug_mode']:
                            cv2.imwrite(save_path + '{}_flipcrop_{}.jpg'.format(
                                i,crop_position),res['sequence'][i])
                        
            if (
                self.mode == 'val' and
                self.training_params['use_data_augmentation'] and
                self.training_params['random_corner_cropping']
            ):  
                shape = self.config['image_shape']
                scale = self.crop_scales[1]
                crop_size = (int(shape[0]*scale), int(shape[1]*scale))
                center_x = shape[1] // 2
                center_y = shape[0] // 2
                box_half_x = crop_size[1] // 2
                box_half_y = crop_size[0] // 2
                x1 = center_x - box_half_x
                y1 = center_y - box_half_y
                x2 = center_x + box_half_x
                y2 = center_y + box_half_y
                for i in range(len(res['sequence'])):
                    res['sequence'][i] = cv2.resize(
                        res['sequence'][i][y1:y2,x1:x2,:],
                        tuple(self.config['input_shape']))
                    
            batch.append(res['sequence'])
            batch_labels.append(res['labels'])
            del res
        
        if self.mode == 'train':
            assert len(batch) == self.training_params['batch_size']
        
        return (np.asarray(batch, dtype=np.float32),
            np.asarray(to_categorical(batch_labels, self.num_classes)))