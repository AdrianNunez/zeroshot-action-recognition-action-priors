import os
import sys
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from data import get_classes_ordered

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

split_nb = 1
variables_file = 'variables.json'
with open(variables_file) as f:
    config = json.load(f)

# ============================================================
# VARIABLES TO MODIFY
# ============================================================
split = 'name_of_split'
recipe_split = False
percentage_of_classes_for_test = 0.2
action_ind_file = 'original_action_idx.txt'
noun_ind_file = 'original_noun_idx.txt'
# ============================================================

root_path = config['split_path'] + split + '/'
if not os.path.exists(root_path):
    os.makedirs(root_path)

object_dict, object_names = dict(), []
verb_dict, verb_names = dict(), []
action_dict, action_names = dict(), []
with open(action_ind_file, 'r') as f:
    content = f.readlines()
    for c in content:
        action = c[:c.rfind(' ')].strip().lower()
        action_dict[action] = []
        action_names.append(action)

        verb, obj = action[:action.rfind(' ')], action[action.rfind(' ')+1:]

        if not verb in verb_dict:
            verb_dict[verb] = []
            verb_names.append(verb)

        if not obj in object_dict:
            object_dict[obj] = []
            object_names.append(obj)

non_compound_objects = []
with open(noun_ind_file, 'r') as f:
    content = f.readlines()
    for c in content:
        class_name, _ = c.strip().split(' ')
        non_compound_objects.append(class_name.lower())

# Join train and test sets (to get a collection of all the videos)
dataset = []
dataset += open('/data/anunez/EGTEA/train_split{}.txt'.format(
                split_nb), 'r').readlines()
dataset += open('/data/anunez/EGTEA/test_split{}.txt'.format(
                split_nb), 'r').readlines()

nb_samples_per_class = dict()

# Remove objects and verbs appearing only in one combination
for line in dataset:
    elems = line.strip().split(' ')
    action = action_names[int(elems[1])-1].lower()
    verb = verb_names[int(elems[2])-1].lower()
    obj = '{}'.format(
        ','.join([non_compound_objects[int(x)-1] for x in elems[3:]])).lower()
    if not obj in object_dict: object_dict[obj] = []
    if not verb in verb_dict: verb_dict[verb] = []
    if not action in object_dict[obj]: object_dict[obj].append(action)
    if not action in verb_dict[verb]: verb_dict[verb].append(action)
    if not action in nb_samples_per_class: nb_samples_per_class[action] = 0
    nb_samples_per_class[action] += 1

def remove_single_appearances(dic):
    selected = []
    for key in dic.keys():
        if len(dic[key]) > 1:
            selected.append(key)
    return selected

# Remove single appearances of objects/verbs
selected_objects = sorted(remove_single_appearances(object_dict))
selected_verbs = sorted(remove_single_appearances(verb_dict))

def save(path, selected):
    with open(path, 'w') as f:
        for count, elem in enumerate(selected):
            f.write('{} {}\n'.format(elem, count+1))

save(root_path + 'noun_idx.txt', selected_objects)
save(root_path + 'verb_idx.txt', selected_verbs)

selected_actions = []
for key in action_dict.keys():
    verb, noun = key[:key.rfind(' ')], key[key.rfind(' ')+1:]
    if not verb in selected_verbs:
        continue
    if not noun in selected_objects:
        continue
    selected_actions.append(key)

selected_actions = sorted(selected_actions)
save(root_path + 'action_idx.txt', selected_actions)

inv_selected_actions = dict(zip(selected_actions,range(len(selected_actions))))
inv_selected_verbs = dict(zip(selected_verbs,range(len(selected_verbs))))
inv_selected_objects = dict(zip(selected_objects,range(len(selected_objects))))

# Select videos of the actions selected
X, y = [], []
verbs, objects = [], []
for line in dataset:
    d = line.split(' ')
    folder, label = d[0], int(d[1])-1
    
    if action_names[label] in selected_actions:
        X.append(folder)
        # Include new label
        action = action_names[label]
        y.append(inv_selected_actions[action])
        #print(label, action, y[-1])
        
        """ v = inv_selected_verbs[action[:action.rfind(' ')]]
        verbs.append(v)
        o = inv_selected_objects[action[action.rfind(' ')+1:]]
        objects.append(o) """

        v = action[:action.rfind(' ')]
        verbs.append(v)
        o = action[action.rfind(' ')+1:]
        objects.append(o)

# Divide into train/test action
num_classes = len(selected_actions)
num_test = int(num_classes*percentage_of_classes_for_test)

# Choose class split for train/test
correct_split = False
times = 0

while not correct_split:
    times += 1
    correct_split = True
    indices = np.random.choice(range(len(selected_actions)),
                            size=num_test,
                            replace=False)
    train_actions, test_actions = [], []
    train_verbs, train_objects = set(), set()
    test_verbs, test_objects = set(), set()

    for ind in indices:
        test_actions.append(selected_actions[ind])
    
    for i in range(len(selected_actions)):
        if not i in indices:
            train_actions.append(selected_actions[i])
    
    assert len(list(set(train_actions) & set(test_actions))) == 0

    # Ban verbs, objects and actions
    if recipe_split:
        repeat = False
        for test_action in test_actions:
            if (   
                'sponge' in test_action or                
                'cabinet' in test_action or                
                'grocery_bag' in test_action or
                'fridge' in test_action or
                'drawer' in test_action or
                'eating_utensil' in test_action or
                'inspect' in test_action or
                'wash pan' == test_action
            ):
                repeat = True
        if repeat: 
            correct_split = False
            continue

    # Check if every verb and object of the test set appears in the
    # training set
    for test_action in test_actions:
        pos = test_action.rfind(' ')
        test_verb, test_obj = test_action[:pos], test_action[pos+1:]
        test_verbs.add(test_verb)
        test_objects.add(test_obj)
        found_verb, found_obj = False, False
        for train_action in train_actions:
            pos = train_action.rfind(' ')
            train_verb, train_obj = train_action[:pos], train_action[pos+1:]
            train_verbs.add(train_verb)
            train_objects.add(train_obj)
            if train_verb == test_verb:
                found_verb = True
            if train_obj == test_obj:
                found_obj = True
            
        if not found_verb or not found_obj:
            correct_split = False
            continue

def save_labels(path, elems, indices):
    with open(path, 'w') as f:
        for elem in elems:
            f.write('{} {}\n'.format(elem, indices[elem]))

inv_selected_actions = {k:v+1 for k,v in inv_selected_actions.items()}
save_labels(root_path + 'test_actions.txt', test_actions, inv_selected_actions)

def load_labels(path):
    dic = dict()
    with open(path, 'r') as f:
        content = f.readlines()
        for c in content:
            pos = c.rfind(' ')
            elem, num = c[:pos], int(c[pos+1:])
            dic[elem.lower()] = num
    return dic

verb_dict = load_labels(config['project_folder'] + 'original_verb_idx.txt')
obj_dict = load_labels(config['project_folder'] + 'original_noun_idx.txt')

_train_verbs = sorted(list(train_verbs))
_train_objects = sorted(list(train_objects))
inv_train_verbs = dict(zip(_train_verbs,range(1,len(_train_verbs)+1)))
inv_train_objects = dict(zip(_train_objects,range(1,len(_train_objects)+1)))

_test_verbs = sorted(list(test_verbs))
_test_objects = sorted(list(test_objects))

save_labels(root_path + 'test_verbs.txt', _test_verbs, inv_train_verbs)
save_labels(root_path + 'test_objects.txt', _test_objects, inv_train_objects)

train_X, train_y, train_verbs, train_objects = [], [], [], []
test_X, test_y, test_verbs, test_objects = [], [], [], []

num_samples_test = 0
for i in range(len(X)):
    if selected_actions[y[i]] in train_actions:
        train_X.append(X[i])
        train_y.append(y[i])
        obj = inv_train_objects[objects[i]]
        verb = inv_train_verbs[verbs[i]]
        train_verbs.append(verb)
        train_objects.append(obj)
    if selected_actions[y[i]] in test_actions:
        num_samples_test += 1
        test_X.append(X[i])
        test_y.append(y[i])
        obj = inv_train_objects[objects[i]]
        verb = inv_train_verbs[verbs[i]]
        test_verbs.append(verb)
        test_objects.append(obj)

with open('{}test.txt'.format(root_path), 'w') as f:
    for i in range(len(test_X)):
        f.write('{} {} {} {}\n'.format(test_X[i],
                                        test_y[i]+1,
                                        test_verbs[i],
                                        test_objects[i]))

save_labels(root_path + 'train_verbs.txt', _train_verbs, inv_train_verbs)
save_labels(root_path + 'train_objects.txt', _train_objects, inv_train_objects)
save_labels(root_path + 'train_actions.txt', train_actions, inv_selected_actions)
save_labels(root_path + 'val_verbs.txt', _train_verbs, inv_train_verbs)
save_labels(root_path + 'val_objects.txt', _train_objects, inv_train_objects)
save_labels(root_path + 'val_actions.txt', train_actions, inv_selected_actions)
sss = StratifiedShuffleSplit(n_splits=2, test_size=0.10, random_state=0)
train_index, val_index = sss.split(train_X, train_y).next()
        
num_samples_train, num_samples_val = 0, 0
with open('{}train.txt'.format(root_path), 'w') as f:
    for i in train_index:
        num_samples_train += 1
        f.write('{} {} {} {}\n'.format(train_X[i],
                                        train_y[i]+1,
                                        train_verbs[i],
                                        train_objects[i]))

with open('{}val.txt'.format(root_path), 'w') as f:
    for i in val_index:
        num_samples_val += 1
        f.write('{} {} {} {}\n'.format(train_X[i],
                                        train_y[i]+1,
                                        train_verbs[i],
                                        train_objects[i]))

s = 'Num samples train: {}, val: {}, test: {}'.format(
    num_samples_train, num_samples_val, num_samples_test)
print(s)
with open(root_path + 'nb_samples.txt', 'w') as f:
    f.write(s)

def plot_class_distribution(folder, labels_by_video, classes, mode):
    dist = Counter(labels_by_video)
    indices = sorted(range(len(dist.keys())), key=dist.keys().__getitem__)
    plt.bar(range(len(indices)), np.asarray(dist.values())[indices])
    tick_marks_x = np.arange(len(dist.values()))
    plt.title('Distribution of classes in {} by video'.format(mode))
    plt.xticks(tick_marks_x, sorted(classes), fontsize=4, rotation=90)
    plt.tight_layout()
    plt.ylabel('Number of videos')
    plt.xlabel('Classes')
    plt.savefig(folder + '{}_video_distribution.pdf'.format(mode), bbox_inches='tight')
    plt.gcf().clear()

plot_class_distribution(root_path,
                        np.asarray(train_y)[train_index],
                        train_actions,
                        'train')

plot_class_distribution(root_path,
                        np.asarray(train_y)[val_index],
                        train_actions,
                        'val')

plot_class_distribution(root_path,
                        test_y,
                        test_actions,
                        'test')

# ===================================================================
# CREATE PERFECT PRIOR
# ===================================================================

folder = root_path
#output_path = root_path + 'perfect_prior.json'

action_names = dict()
with open('{}train_actions.txt'.format(folder), 'r') as f:
    content = f.readlines()
    for c in content:
        action = c[:c.rfind(' ')].strip().lower()
        num = int(c[c.rfind(' ')+1:])
        action_names[num] = action
with open('{}test_actions.txt'.format(folder), 'r') as f:
    content = f.readlines()
    for c in content:
        action = c[:c.rfind(' ')].strip().lower()
        num = int(c[c.rfind(' ')+1:])
        action_names[num] = action

path = root_path + config['train_objects_file']
int_to_objects,_ = get_classes_ordered(path)
path = root_path + config['train_verbs_file']
int_to_verbs,_ = get_classes_ordered(path)

action_priors = dict()
for verb in int_to_verbs:
    for obj in int_to_objects:
        action_priors[verb + ' ' + obj] = []

dataset = open('{}test.txt'.format(folder), 'r').readlines()
for line in dataset:
    elems = line.strip().split(' ')
    action = action_names[int(elems[1])].lower()
    action_priors[action].append(1)

size = 0.
for action in action_priors.keys():
    if len(action_priors[action]) > 0:
        size += float(np.sum(action_priors[action]))

for action in action_priors.keys():
    if len(action_priors[action]) == 0:
        action_priors[action] = 0.
    else:
        total = float(np.sum(action_priors[action]))
        action_priors[action] = total/float(size)

with open(root_path + 'perfect_prior.json', 'w') as f:
    json.dump(action_priors, f, ensure_ascii=False, indent=4)

# ===================================================================

with open(root_path + 'perfect_prior.json', 'r') as json_file:
    perfect_prior = json.load(json_file) 

with open(root_path + 'test_actions_statistics.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Class', 'Nb samples', 'Perfect prior'])
    p = [perfect_prior[action] for action in test_actions]
    indices = np.argsort(np.asarray(p))[::-1]
    for i in indices:
        action = test_actions[i]
        writer.writerow([action, int(nb_samples_per_class[action]), 
                         '{:.2}'.format(perfect_prior[action]*100.)])
       