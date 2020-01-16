import os
import csv
import json
import operator
import numpy as np
from data import get_classes_ordered
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.dummy import DummyClassifier

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

variables_file = 'variables.json'
with open(variables_file) as f:
    config = json.load(f)

# ============================================================
# VARIABLES TO BE MODIFIED
# ============================================================
split = 'split_name'
mode = 'test'
nb_averaged_runs = 3
# ============================================================

root_path = config['prior_base_path'] + '{}/'.format(split)
experiments_path = config['project_folder'] + config['plots_folder']

with open(root_path + config['perfect_prior'], 'r') as json_file:
    perfect_prior = json.load(json_file) 

with open(config['action_frequencies_4gram'], 'r') as json_file:
    action_frequencies_4gram = json.load(json_file) 

with open(config['action_prior_4gram'], 'r') as json_file:
    action_prior_4gram = json.load(json_file) 

with open(config['google_prior'], 'r') as json_file:
    google_prior = json.load(json_file)  

with open(config['phrasefinder_prior'], 'r') as json_file:
    phrasefinder_prior = json.load(json_file)  

with open(root_path + config['{}_actions_file'.format(mode)], 'r') as f:
    content = f.readlines()
    test_actions = [x[:x.rfind(' ')] for x in content]
    test_indices = [int(x[x.rfind(' ')+1:]) for x in content]
    test_dict = dict(zip(test_indices, test_actions))

lines = open(root_path + 'test.txt', 'r').readlines()
nb_samples_per_class = dict()
for line in lines:
    elems = line.strip().split(' ')
    action = test_dict[int(elems[1])].lower()
    if not action in nb_samples_per_class: nb_samples_per_class[action] = 0
    nb_samples_per_class[action] += 1

def load_results(root_path):
    results = dict()
    for i in range(nb_averaged_runs):
        subpath = '/run_{}/results_by_class_top1.txt'.format(i)
        path = experiments_path + root_path + subpath
        with open(path, 'r') as f: content = f.readlines()
        for c in content:
            action = c[:c.find('-')-1]
            accuracy = float(c[c.find('-')+2:c.rfind('-')-1])
	    if not action in results: results[action] = []
            results[action].append(accuracy)
    for key in results.keys():
	results[key] = [np.mean(results[key])*100., np.std(results[key])*100.]
    return results

# Load results
subpath = '{}/split{}_{}_'.format(split, split, mode)
baseline_results = load_results(subpath + 'baseline_no_text')
perfect_results = load_results(subpath + 'perfect_prior')
wikicook_results_4gram = load_results(subpath + 'wikicook_prior_4gram')
google_results = load_results(subpath + 'google_prior')
phrasefinder_results = load_results(subpath + 'phrasefinder_prior')


most_common_action = sorted(nb_samples_per_class.items(),
                            key=operator.itemgetter(1))[-1]

total_videos = sum(nb_samples_per_class.values())
objects,_ = get_classes_ordered(root_path + config['train_objects_file'])
verbs,_ = get_classes_ordered(root_path + config['train_verbs_file'])
total_actions = len(objects)*len(verbs)
print('Random Classifier: Accuracy {:.2f}, Macro-F1 {:.2f}'.format(
    total_videos*(1./total_actions),
    most_common_action[1]/float(total_videos),
))
int_to_actions,action_indices = get_classes_ordered(root_path + config['actions_file'])
action_dict = dict(zip(int_to_actions, action_indices))

total_classes = len(verbs)*len(objects)
prob = 1./float(total_classes)
acc = np.sum([prob*nb_samples_per_class[action] for action in test_actions])
print(acc)

with open(experiments_path + '{}/results_statistics_split{}_{}.csv'.format(split,split, mode), 'w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Class', 'Nb samples', 'Baseline result', 
                     'Prior in wikicook (4-gram)', 'Result with wikicook (4-gram)',
                     'Prior in google', 'Result with google prior',
                     'Prior in phrasefinder', 'Result with phrasefinder prior'
                     'Perfect prior', 'Result with perfect prior',
    ])

    test_actions = sorted(test_actions)
    for action in test_actions:
        row = [
            action, int(nb_samples_per_class[action]),
            '{:.2f} (-+{:.2f})'.format(baseline_results[action][0], baseline_results[action][1]),
            '{:.6}'.format(action_prior_4gram[action]),
            '{:.2f} (-+{:.2f})'.format(wikicook_results_4gram[action][0], wikicook_results_4gram[action][1]),
            # Google prior
            '{:.6}'.format(google_prior[action]),
            '{:.2f} (-+{:.2f})'.format(google_results[action][0], google_results[action][1]),
            # Phrasefinder prior
            '{:.6}'.format(phrasefinder_prior[action]),
            '{:.2f} (-+{:.2f})'.format(phrasefinder_results[action][0], phrasefinder_results[action][1]),
            # Perfect prior
            '{:.6}'.format(perfect_prior[action]),
            '{:.2f} (-+{:.2f})'.format(perfect_results[action][0], perfect_results[action][1]),
        ]
        writer.writerow(row)