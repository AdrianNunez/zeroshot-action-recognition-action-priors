# -*- coding: UTF-8 -*-
import os
import json
import logging
from googleapiclient.discovery import build
from tqdm import tqdm
from data import get_classes_ordered
logging.getLogger('googleapicliet.discovery_cache').setLevel(logging.ERROR)

variables_file = 'variables.json'
with open(variables_file) as f:
    config = json.load(f)

# ============================================================
# VARIABLES TO MODIFY
# ============================================================
output_path = config['project_folder'] + 'google_prior/'
# ============================================================

# API keys --------------------------------------------
api_keys = [
    # add API keys
]

# Google Custom Search --------------------------------
cse_ids = [
    # add Google Custom Search IDs
]

# Function to perform a google search
def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res

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
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    objects,_ = get_classes_ordered(config['objects_file'])
    verbs,_ = get_classes_ordered(config['verbs_file'])
    total = 0
    results_dict, action_priors = dict(), dict()
    if os.path.exists(output_path + 'google_search.json'):
        with open(output_path + 'google_search.json', 'r') as json_file:
            results_dict = json.load(json_file)   

def check_queries_left(results_dict):
    queries_done, queries_left = 0, 0
    # Check how many queries are left
    for verb in verbs:
            v = transform_verb(verb)
            for v_option in v:
                for obj in objects:
                    o = transform_obj(obj)
                    for o_option in o:
                        if not verb + ' ' + obj in results_dict:
                            queries_left += 1
                        elif not v_option + ' ' + o_option in results_dict[verb + ' ' + obj]:
                            queries_left += 1
                        else:
                            queries_done += 1

    print('Queries done: {}, queries left: {}, total queries: {}'.format(
        queries_done, queries_left, queries_done + queries_left
    ))

# It should print the total queries that must be done
check_queries_left(results_dict) 

for my_api_key, my_cse_id in tqdm(zip(api_keys, cse_ids)):
    # For each verb and object (and their synonyms)
    for verb in verbs:
        v = transform_verb(verb)
        for v_option in v:
            for obj in objects:
                o = transform_obj(obj)
                for o_option in o:
                    try:
                        if not verb + ' ' + obj in results_dict:
                            results_dict[verb + ' ' + obj] = dict()
                        action = v_option + ' * ' + o_option
                        if not v_option + ' ' + o_option in results_dict[verb + ' ' + obj]:
                            #print(action)
                            result = google_search('"' + action + '"', my_api_key, my_cse_id)
                            results_dict[verb + ' ' + obj][v_option + ' ' + o_option] = result
                            with open(output_path + 'google_search.json', 'w') as f:
                                json.dump(results_dict, f, indent=4)
                    except:
                        pass

# It should print 0, otherwise it must be repeated
check_queries_left(results_dict) 

# Create the prior using the computed results
accum_total = 0.
for verb in verbs:
    for obj in objects:
        action = verb + ' ' + obj
        info = []
        for key in results_dict[action].keys():
            num = int(
                results_dict[action][key]['searchInformation']['totalResults']
            )  
            info.append(num)
        accum, nb_elems = 0., 0
        for i in range(len(info)):
            if info[i] > 0: 
                accum += float(info[i])
                nb_elems += 1
        total = float(accum) / max(1,float(nb_elems))
        accum_total += total
        action_priors[action] = total
            
with open(output_path + 'unnormalised_action_priors.json', 'w') as f:
    json.dump(action_priors, f, indent=4)

for key in action_priors.keys():
    action_priors[key] /= float(accum_total) 

with open(output_path + 'action_priors.json', 'w') as f:
    json.dump(action_priors, f, indent=4)