# coding: utf-8
import os
import re
import sys
import unwiki
import nltk
import json
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from nltk.wsd import lesk
from nltk.corpus import wordnet, stopwords
import gensim
from tqdm import tqdm
from scipy import spatial
from data import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

variables_file = 'variables.json'
with open(variables_file) as f:
    config = json.load(f)

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# ============================================================
# VARIABLES TO BE MODIFIED
# ============================================================
# Path to save Cookbook related files
path = 'cookbook/'
# Link to the json containing the 1MRecipes+ dataset
recipes_file = 'recipes_with_nutritional_info.json'
# Whether to add the 1MRecipes+
add_recipes_corpus = False
# Output path to save the prior
output_path = 'priors/4ngrams/'
ngram_size = 4
# Only select one of them: whether to use stemmer or lemmatiser
use_stemmer = False
use_lemmatiser = True
# ============================================================

stemmer, lemmatiser = None, None
if use_stemmer: stemmer = PorterStemmer()
if use_lemmatiser: lemmatiser = WordNetLemmatizer() 



def removeNonAscii(s): return "".join(filter(lambda x: ord(x)<128, s))

def clean_text(s, stemmer, lemmatiser):
    """
    Takes a string as input and cleans it by removing non-ascii characters,
    lowercasing it, removing stopwords and lemmatising/stemming it
    - Input:
        * s (string)
        * stemmer (object that stems a string)
        * lemmatiser (object that lemmatises a string)
    - Output:
        * text (string)
    """
    stop_words = set(stopwords.words('english')) 
    # Remove non ASCII characters
    text = removeNonAscii(s)
    text = text.lower()
    # Remove any undesired character
    for s in ['#', '|', '*', '.', ',', ';', '!', ':']:
        text = text.replace(s, '')
    # Remove digits
    for s in [str(x) for x in range(10)]:
        text = text.replace(s, '')
    text = text.replace('\n', ' ')
    text = re.sub(' +', ' ', text)

    # Apply stemmer/lemmatiser
    word_tokens = word_tokenize(text)
    s = []
    for w in word_tokens:
        if w in stop_words: continue
        if stemmer:
            s.append(stemmer.stem(w))
        if lemmatiser:
            s.append(lemmatiser.lemmatize(w))
    text = ' '.join(s)

    return text

if np.sum([use_stemmer, use_lemmatiser]) > 1:
    print('Choose only one option or none among: use_stemmer, use_lemmatiser')
    sys.exit()

if not os.path.exists(output_path):
    os.makedirs(output_path)

# If this is already created, skip it
if not os.path.exists(output_path + 'features/ocurrences_matrix_cookbook.npy'):
    # Get all the scrapped Cookbook files
    wiki_files = sorted([f for f in os.listdir(path) 
                    if os.path.isfile(os.path.join(path, f))])
    # Clean the corpus
    corpus = []
    for wiki_file in wiki_files:
        text = unwiki.loads(' '.join(open(path + wiki_file)))

        text = clean_text(text, stemmer, lemmatiser)

        with open(output_path + wiki_file, 'w') as f:
            f.write(text)
        corpus.extend(text.split('. '))

    # Compute the occurences matrix
    features_output_path = output_path + 'features/'
    if not os.path.exists(features_output_path):
        os.makedirs(features_output_path)

    if not os.path.exists(new_output_path + 'ocurrences_matrix_cookbook.npy'): 
        print('Computing occurrences')
        cv = CountVectorizer(ngram_range=(ngram_size,ngram_size))
        X = cv.fit_transform(corpus)
        ngrams = cv.get_feature_names()
        X = X.toarray()
        X = np.sum(X, axis=0)
        
        with open(features_output_path + 'ngrams_cookbook.txt',
                  'w') as f:
            for ngram in ngrams:
                f.write('{}\n'.format(ngram))
        np.save(features_output_path + 'ocurrences_matrix_cookbook.npy', X)

        # Compute the vocabulary and save it (optional)
        cv_vocab = CountVectorizer(ngram_range=(1,1))
        X_vocab = cv_vocab.fit_transform(corpus)
        vocabulary = cv_vocab.vocabulary_.keys()
        with open(features_output_path + 'vocabulary_cookbook.txt', 'w') as f:
            for v in vocabulary:
                f.write(v + '\n')
    else:
        # Read the occurrence matrix
        X = np.load(features_output_path + 'ocurrences_matrix_cookbook.npy')
        with open(features_output_path + 'ngrams_cookbook.txt', 'r') as f:
            ngrams = [x.strip() for x in f.readlines()]

        with open(features_output_path + 'vocabulary_cookbook.txt', 'r') as f:
            vocabulary = [x.strip() for x in f.readlines()]
    
    # Add the 1MRecipes+ corpus to the Cookbook
    if add_recipes_corpus:
        print('Adding extra recipes')
        cv = CountVectorizer(ngram_range=(ngram_size,ngram_size))
        X = list(X)
        ngrams_dict = dict(zip(ngrams,[True]*len(ngrams)))
        vocabulary = list(vocabulary)
        vocabulary_dict = dict(zip(vocabulary,[True]*len(vocabulary)))
        with open(recipes_file, 'r') as json_file:
            recipes = json.load(json_file)
            corpus = []
            for i in tqdm(range(len(recipes))):
                for j in range(len(recipes[i]['instructions'])):
                    text = recipes[i]['instructions'][j]['text']
                    text = clean_text(text, stemmer, lemmatiser)
                    if len(text.split(' ')) < 5: continue
                    corpus.append(text)
                # Compute and save the results by steps to hasten the process
                if len(corpus) > 1e4:
                    _X = cv.fit_transform(corpus)
                    _X = _X.toarray()
                    _X = np.sum(_X, axis=0)
                    X = list(X)
                    _ngrams = cv.get_feature_names()
                    for count, ngram in enumerate(_ngrams):
                        if not ngram in ngrams_dict:
                            feature_names.append(ngram)
                            X.append(_X[count])
                            ngrams_dict[ngram] = True
                        else:
                            pos = ngrams.index(ngram)
                            X[pos] += _X[count]
                    cv_vocab = CountVectorizer(ngram_range=(1,1))
                    _X_vocab = cv_vocab.fit_transform(corpus)
                    _vocabulary = cv_vocab.vocabulary_.keys()
                    for vocab in list(_vocabulary):
                        if not vocab in vocabulary_dict:
                            vocabulary.append(vocab)
                            vocabulary_dict[vocab] = True
                    corpus = []
            # Compute and save the results of the remaining lines
            # to hasten the process
            if len(corpus) > 0:
                _X = cv.fit_transform(corpus)
                _X = _X.toarray()
                _X = np.sum(_X, axis=0)
                X = list(X)
                _feature_names = cv.get_feature_names()
                for count, ngram in enumerate(_ngrams):
                    if not ngram in ngrams:
                        feature_names.append(ngram)
                        X.append(_X[count])
                        ngrams_dict[ngram] = True
                    else:
                        pos = ngrams.index(ngram)
                        X[pos] += _X[count]
                cv_vocab = CountVectorizer(ngram_range=(1,1))
                _X_vocab = cv_vocab.fit_transform(corpus)
                _vocabulary = cv_vocab.vocabulary_.keys()
                for vocab in list(_vocabulary):
                    if not vocab in vocabulary_dict:
                        vocabulary.append(vocab)
                        vocabulary_dict[vocab] = True
        X = np.asarray(X)

    # Load the n-grams
    with open(features_output_path + 'ngrams.txt', 'w') as f:
        for ngram in ngrams:
            f.write('{}\n'.format(ngram))
    X = np.asarray(X)
    np.save(features_output_path + 'ocurrences_matrix.npy', X)

    cv_vocab = CountVectorizer(ngram_range=(1,1))
    X = cv_vocab.fit_transform(corpus)
    vocab = cv_vocab.vocabulary_.keys()
    with open(features_output_path + 'vocabulary.txt', 'w') as f:
        for v in vocab:
            f.write(v + '\n')
    

# Load N-grams
output_path = output_path + 'features/'
with open(output_path + 'ngrams.txt', 'r') as f:
    feature_names = [x.strip() for x in f.readlines()]

with open(output_path + 'vocabulary.txt', 'r') as f:
    vocabulary = [x.strip() for x in f.readlines()]

# Load ocurrence matrix
X = np.load(output_path + 'ocurrences_matrix.npy')

ngrams_dict = dict(zip(feature_names, range(len(feature_names))))

def tweakObject(obj):
    """
    Takes as input an object and returns a list of synonyms
    - Input:
        * obj (string)
    - Output:
        * tweakedObj (list of strings)
    """
    tweakedObj = [obj]
    if obj == 'bell_pepper':
        tweakedObj = ['bell pepper']
    elif obj == 'cup':
        tweakedObj = ['cup', 'mug']
    elif obj == 'pot':
        tweakedObj = ['pot', 'saucepan']
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
        tweakedObj = ['bread container', 'bread recipient', 'bread package']
    elif obj == 'grocery_bag':
        tweakedObj = ['grocery bag', 'groceries']
    elif obj == 'seasoning_container':
        tweakedObj = ['seasoning container', 'seasoning recipient',
                      'seasoning bottle', 'seasoning package']
    elif obj == 'condiment_container':
        tweakedObj = ['condiment container', 'condiment recipient',
                      'condiment bottle']
    elif obj == 'tomato_container':
        tweakedObj = ['tomato container', 'tomato recipient']
    elif obj == 'fridge':
        tweakedObj = ['fridge', 'refrigerator']
    elif obj == 'paper_towel':
        tweakedObj = ['paper towel', 'tissue', 'kitchen paper',
                      'kitchen towel']
    elif obj == 'cabinet':
        tweakedObj = ['cabinet', 'locker', 'cupboard']
    return tweakedObj

def tweakVerb(verb):
    """
    Takes as input a verb and returns a list of synonyms
    - Input:
        * verb (string)
    - Output:
        * tweakedVerb (list of strings)
    """
    tweakedVerb = [verb]
    if verb == 'divide/pull apart':
        tweakedVerb = ['divide', 'pull apart', 'separate', 'split', 'shred']
    elif verb == 'move_around':
        tweakedVerb = ['move around', 'move', 'roam', 'transfer']
    elif verb == 'take':
        tweakedVerb = ['take', 'pick', 'pick up', 'grab', 'bring']
    elif verb == 'put':
        tweakedVerb = ['put', 'leave', 'place', 'set', 'lay']
    elif verb == 'cut':
        tweakedVerb = ['cut', 'slice', 'mince']
    elif verb == 'wash':
        tweakedVerb = ['wash', 'clean']
    return tweakedVerb

def lemmatise(elems, lemmatiser):
    """
    Takes as input list of tokens (strings) and applies a lemmatiser to each
    of them
    - Input:
        * elems (list of strings)
    - Output:
        * elems (list of strings)
    """
    for i in range(len(elems)):
        if ' ' in elems[i]:
            _elems = elems[i].split(' ')
            s = []
            for j in range(len(_elems)):
                s.append(lemmatiser.lemmatize(_elems[j]))
            elems[i] = ' '.join(s)
        else:
            elems[i] = lemmatiser.lemmatize(elems[i])
    return elems
    
def stem(elems, stemmer):
    """
    Takes as input list of tokens (strings) and applies a stemmer to each
    of them
    - Input:
        * elems (list of strings)
    - Output:
        * elems (list of strings)
    """
    for i in range(len(elems)):
        if ' ' in elems[i]:
            _elems = elems[i].split(' ')
            s = []
            for j in range(len(_elems)):
                s.append(stemmer.stem(_elems[j]))
            elems[i] = ' '.join(s)
        else:
            elems[i] = lemmatiser.lemmatize(elems[i])
    return elems

# Save a copy of the ngrams without stemming/lemmatising
original_ngrams = np.copy(ngrams)
# Lemmatise/stem feature names (n-grams)
for i in range(len(ngrams)):
    word_tokens = word_tokenize(ngrams[i])
    s = []
    for word in word_tokens:
        if use_lemmatiser:
            s.append(lemmatiser.lemmatize(word))
        if stemmer:
            s.append(stemmer.stem(word))
    ngrams[i] = ' '.join(s)

# (OPTIONAL) Check object occurrences in the corpus (from the N-grams)
objects,_ = get_classes_ordered(config['train_objects_file'])
object_correspondences = dict()
for orig_obj in objects:
    object_correspondences[orig_obj] = set()
    objects = tweakObject(orig_obj)
    if lemmatiser:
        objects = lemmatise(objects, lemmatiser)
    if stemmer:
        objects = stem(objects, stemmer)
    for obj in objects:
        for feature_name in feature_names:
            if ' ' in obj:
                objs = obj.split(' ')
                check = 0
                for _obj in objs:
                    if _obj in feature_name:
                        check += 1
                if check == len(objs):
                    object_correspondences[orig_obj].add(feature_name)
            else:
                if obj in feature_name:
                    object_correspondences[orig_obj].add(feature_name)
    object_correspondences[orig_obj] = list(object_correspondences[orig_obj])
with open(output_path + 'object_correspondences.json', 'w') as f:
    json.dump(object_correspondences, f, ensure_ascii=False, indent=4)

# (OPTIONAL) Check verb occurrences in the corpus (from the N-grams)
verbs,_ = get_classes_ordered(config['train_verbs_file'])
verb_correspondences = dict()
for orig_verb in verbs:
    verb_correspondences[orig_verb] = set()
    verbs = tweakVerb(orig_verb)
    if lemmatiser:
        verbs = lemmatise(verbs, lemmatiser)
    if stemmer:
        verbs = stem(verbs, stemmer)
    for verb in verbs:
        for feature_name in feature_names:
            if verb in feature_name:
                verb_correspondences[orig_verb].add(feature_name)
    verb_correspondences[orig_verb] = list(verb_correspondences[orig_verb])
with open(output_path + 'verb_correspondences.json', 'w') as f:
    json.dump(verb_correspondences, f, ensure_ascii=False, indent=4)

# Generate the set of actions available in the dataset
objects,_ = get_classes_ordered(config['train_objects_file'])
verbs,_ = get_classes_ordered(config['train_verbs_file'])
actions = []
for verb in verbs:
    for obj in objects:
        actions.append(verb + ' ' + obj)

# Check action occurrences in the corpus (from the N-grams)
action_correspondences = dict()
for action in actions:
    pos = action.rfind(' ')
    verb, obj = action[:pos], action[pos+1:]
    # Get synonyms of object and verb
    objects = tweakObject(obj)
    verbs = tweakObject(verb)
    # Lemmatise/stem them
    if lemmatiser:
        objects = lemmatise(objects, lemmatiser)
        verbs = lemmatise(verbs, lemmatiser)
    if stemmer:
        objects = stem(objects, stemmer)
        verbs = stem(verbs, stemmer)
    
    action_correspondences[action] = dict()
    action_correspondences[action]['ngrams'] = []
    action_correspondences[action]['frequency'] = []
    for count, ngram in enumerate(ngrams):      
        verb_found, obj_found = False, False
        for verb in verbs:
            if verb in ngram.split(' '):
                verb_found = True
        
        for obj in objects:
            if ' ' in obj:
                objs = obj.split(' ')
                check = 0
                for _obj in objs:
                    if _obj in feature_name.split(' '):
                        check += 1
                if check == len(objs):
                    obj_found = True
            else:
                if obj in feature_name.split(' '):
                    obj_found = True
        
        if verb_found and obj_found:
            action_correspondences[action]['ngrams'].append(
                original_ngrams[count]
            )
            action_correspondences[action]['frequency'].append(
                X[ngrams_dict[original_ngrams[count]]]
            )

    with open(output_path + 'action_correspondences.json', 'w') as f:
        json.dump(action_correspondences, f, ensure_ascii=False, indent=4)

# Compute the frequency of each action
frequencies = []
for i in range(len(actions)):
    frequencies.append(
        float(np.sum(action_correspondences[actions[i]]['frequency']))
    )

with open(output_path + 'action_frequencies.json', 'w') as f:
    json.dump(dict(zip(actions, frequencies)), 
                f, ensure_ascii=False, indent=4)

# Normalise the frequency (to obtain the prior)
for i in range(len(frequencies)): frequencies[i] /= float(len(ngrams))

with open(output_path + 'action_prior.json', 'w') as f:
    json.dump(dict(zip(actions, frequencies)), 
                f, ensure_ascii=False, indent=4)

# (OPTIONAL) Save the top16 action probabilities
indices = np.argsort(frequencies)[::-1]
with open(output_path + 'top16_action_probabilities.txt', 'w') as f:
    for i in indices[:16]:
        f.write('{} ({})\n'.format(actions[i], frequencies[i]))

# (OPTIONAL) Save a plot of the action prior
frequencies = np.asarray(frequencies)
plt.bar(range(len(actions)), frequencies)
tick_marks_x = np.arange(len(actions))
plt.title('Action priors')
plt.xticks(tick_marks_x, sorted(actions), fontsize=2, rotation=90)
plt.tight_layout()
plt.ylabel('Action probability distribution ({}-grams)'.format(
    ngram_size)
)
plt.xlabel('Action classes')
plt.savefig(output_path + 'action_prior.pdf', bbox_inches='tight')
plt.gcf().clear()

# (OPTIONAL) Save actions with at least one instance
action_with_atleast_one_instance, freq = [], []
for count, action in enumerate(actions):
        if frequencies[count] > 0:
            action_with_atleast_one_instance.append(action)
            freq.append(frequencies[count])
action_with_atleast_one_instance = sorted(action_with_atleast_one_instance)
with open(output_path + 'actions_with_atleast_one_instance.txt', 'w') as f:
    for i in range(len(action_with_atleast_one_instance)):
        f.write('{} {}\n'.format(
            action_with_atleast_one_instance[i], freq[i])
        )

# (OPTIONAL) Apply LaPlacian correction
frequencies = []
for i in range(len(actions)):
    frequencies.append(
        float(np.sum(action_correspondences[actions[i]]['frequency']))
    )
    # LaPlacian correction
    frequencies[i] = (frequencies[i]+1)/(len(feature_names)+len(vocabulary))

with open(output_path + 'action_prior_laplace_correction.json', 'w') as f:
    json.dump(dict(zip(actions, frequencies)), 
                f, ensure_ascii=False, indent=4)

# (OPTIONAL) Save a plot of the LaPlacian corrected action prior
frequencies = np.asarray(frequencies)
plt.bar(range(len(actions)), frequencies)
tick_marks_x = np.arange(len(actions))
plt.title('Action priors')
plt.xticks(tick_marks_x, sorted(actions), fontsize=2, rotation=90)
plt.tight_layout()
plt.ylabel('Probability of action appearing within a 5-gram')
plt.xlabel('Action classes')
plt.savefig(output_path + 'action_prior_laplace_correction.pdf',
            bbox_inches='tight')
plt.gcf().clear()

# (OPTIONAL) Save actions with at least one instance (with LaPlacian correction)
indices = np.argsort(frequencies)[::-1]
with open(output_path + 'top16_action_probabilities_laplace_correction.txt',
         'w') as f:
    for i in indices[:16]:
        f.write('{} ({})\n'.format(actions[i], frequencies[i]))
