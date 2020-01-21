import os
import sys
import json
import urllib
import requests
import numpy as np
sys.path.insert(0, '..')
from data import get_classes_ordered
    
variables_file = 'variables.json'
with open(variables_file) as f:
    config = json.load(f)

# ============================================================
# VARIABLES TO MODIFY
# ============================================================
path = config['project_folder'] + 'phrasefinder_prior/'
# ============================================================

def transform_obj(obj):
    tweakedObj = [obj]
    if obj == 'bell_pepper':
        tweakedObj = ['bell pepper', 'green pepper', 'red pepper']
    elif obj == 'cup':
        tweakedObj = ['cup', 'mug']
    elif obj == 'pot':
        tweakedObj = ['pot', 'saucepan', 'pan']
    elif obj == 'pan':
        tweakedObj = ['pan', 'frying pan']
    elif obj == 'eating_utensil':
        tweakedObj = ['eating utensil', 'knife', 'spoon', 'fork']
    elif obj == 'cooking_utensil':
        tweakedObj = ['cooking utensil', 'knife', 'scissors', 'peeler',
                      'scale', 'jug', 'colander', 'strainer', 'blender']
    elif obj == 'fridge_drawer':
        tweakedObj = ['fridge drawer', 'refrigerator drawer']
    elif obj == 'cutting_board':
        tweakedObj = ['cutting board', 'cut board', 'chopping board',
                      'chop board']
    elif obj == 'cheese_container':
        tweakedObj = ['cheese container', 'cheese recipient', 'cheese package']
    elif obj == 'oil_container':
        tweakedObj = ['oil container', 'oil recipient', 'oil bottle']
    elif obj == 'bread_container':
        tweakedObj = ['bread container', 'bread recipient', 'bread package',
                      'bread bag']
    elif obj == 'grocery_bag':
        tweakedObj = ['grocery bag', 'groceries']
    elif obj == 'seasoning_container':
        tweakedObj = ['seasoning container', 'seasoning recipient',
                      'seasoning bottle', 'seasoning package']
    elif obj == 'condiment_container':
        tweakedObj = ['condiment container', 'condiment recipient',
                      'condiment bottle']
    elif obj == 'tomato_container':
        tweakedObj = ['tomato container', 'tomato recipient', 'tomato bottle']
    elif obj == 'fridge':
        tweakedObj = ['fridge', 'refrigerator']

    elif obj == 'paper_towel':
        tweakedObj = ['paper towel', 'tissue', 'kitchen paper',
                      'kitchen towel']
    elif obj == 'cabinet':
        tweakedObj = ['cabinet', 'locker', 'cupboard']
    return tweakedObj

def transform_verb(verb):
    tweakedVerb = [verb]
    if verb == 'divide/pull apart':
        tweakedVerb = ['divide', 'pull apart', 'separate', 'split', 'shred']
    elif verb == 'move_around':
        tweakedVerb = ['move around', 'move', 'transfer']
    elif verb == 'take':
        tweakedVerb = ['take', 'pick', 'pick up', 'grab']
    elif verb == 'put':
        tweakedVerb = ['put', 'leave', 'place']
    elif verb == 'cut':
        tweakedVerb = ['cut', 'slice', 'mince']
    elif verb == 'wash':
        tweakedVerb = ['wash', 'clean']
    elif verb == 'mix':
        tweakedVerb = ['mix', 'mingle', 'blend']
    return tweakedVerb

if __name__ == '__main__':
    if not os.path.exists(path):
        os.makedirs(path)

    # Get the set of verbs and objects
    objects,_ = get_classes_ordered(config['objects_file'])
    verbs,_ = get_classes_ordered(config['verbs_file'])

    frequencies = dict()
    # For each verb and object (and their synonyms)
    for verb in verbs:
        v = transform_verb(verb)
        frequencies[verb] = dict()
        for v_option in v:
            for obj in objects:
                if not obj in frequencies[verb]:
                    frequencies[verb][obj] = []
                o = transform_obj(obj)
                for o_option in o:
                    # Create and do the query
                    query = '{} ? {}'.format(v_option, o_option)
                    encoded_query = urllib.parse.quote(query)
                    params = {'corpus': 'eng-us', 'query': encoded_query,
                              'format': 'tsv'}
                    params = '&'.join(
                        '{}={}'.format(name, value) 
                        for name, value in params.items()
                    )
                    response = requests.get(
                        'https://api.phrasefinder.io/search?' + params)
                    # Assert that the query was successful
                    assert response.status_code == 200
                
                    # Get the number of results for the query
                    if response.text != '':
                        text = response.text.split('\n')
                        acum = 0
                        for t in text:
                            os = o_option.split(' ')
                            count = 0
                            for _o in os:
                                if _o in t:
                                    count += 1
                            if count == len(os):
                                name, s = t[:t.find('\t')], t[t.find('\t')+1:]
                                mc = int(s[:s.find('\t')]) # number of results
                                acum += mc
                        if acum > 0:
                            frequencies[verb][obj].append(acum)

    # Save frequencies (number of instances) before averaging the results
    # per action
    with open(path + 'frequencies_raw.json', 'w') as f:
        json.dump(frequencies, f, ensure_ascii=False, indent=4, sort_keys=True)

    # Average the results of each action
    action_priors = dict()
    total = 0
    for verb in frequencies.keys():
        for obj in frequencies[verb].keys():
            action = verb + ' ' + obj
            # If no result
            if not len(frequencies[verb][obj]):
                action_priors[action] = 0.
            else:
                action_priors[action] = np.mean(frequencies[verb][obj])
                # Accumulate the total
                total += action_priors[action]

    # Save frequencies (number of instances)
    with open(path + 'frequencies.json', 'w') as f:
        json.dump(action_priors, f, ensure_ascii=False,
                  indent=4, sort_keys=True)

    # Normalies frequency to obtain a probability distribution
    for action in action_priors.keys():
        action_priors[action] = action_priors[action] / float(total)

    # Save prior
    with open(path + 'action_prior.json', 'w') as f:
        json.dump(action_priors, f, ensure_ascii=False,
                  indent=4, sort_keys=True)
