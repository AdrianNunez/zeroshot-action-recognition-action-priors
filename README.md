## zeroshot-action-recognition-action-priors

If you find the code useful for your research, please, cite our paper.

This repository contains the following files:

* **action_inferece.py**: script to infer actions (with and without priors, all are used). Results are stored in csv format within the split path.
* **compute_action_results_per_class.py**: generates a csv file of the results per class per experiment. Results are stored in csv format within the split path.
* **create_cookbook_prior.py**: takes the scrapped Cookbook recipes and creates the Cookbook prior.
* **create_google_prior.py**: creates the google prior.
* **create_phrasefinder_prior.py**: creates the phrasefinder prior. Requires Python3, use requirements_py3.txt.
* **create_split.py**: creates a new split and its corresponding perfect prior.
* **data.py**: scripts related to data management (used by the training scripts, mainly).
* **model.py**: deploys the detectors.
* **objects.txt**: dictionary of the final considered objects.
* **original_action_idx.txt**: original action dictionary of EGTEA Gaze+. Used to create new splits.
* **original_noun_idx.txt**: original object dictionary of EGTEA Gaze+. Used to create new splits.
* **original_verb_idx.txt**: original verb dictionary of EGTEA Gaze+. Used to create new splits.
* **requirements_py2.xt**: to install the Python 2 packages.
* **requirements_py3.xt**: to install the Python 2 packages.
* **train_detector.py**: script to train the verb and object detectors.
* **utils.py**: utils of the script to train the detectors.
* **variables.json**: configuration json.
* **verbs.txt**: dictionary of the final considered verbs.
________

### Step by step

#### Configure the working space

The file *variables.json* must be edited with the configuration of the working space. Specifically:
* project_folder: should point to the main directory.
* images_folder: points to the path where the images of the dataset are stored.
* split_path: root path of the splits, e.g., */home/user/mypath/split*. The name of the split will be appended within the experiments' scripts, for example: */home/user/mypath/split_R*.

There are also paths pointing to the priors which should be modified if the name of the folders containing the priors are changed.

#### Create the split

Create a new split of training and test using the following script:

```
python create_split.py
```

The name of the split is customisable; for that, change the variable that appears on the top part of the script.

The splits used in the paper can be downloaded below.

#### Train the detectors

To train the detectors, the following script must be called twice:

```
python train_detector.py
```

There is a dictionary to configure the experiment, within it you must change:
* The name of the split used (the one that was given in the previous step).
* The label (verb/object), depending on which detector wants to be trained.

The saved models (used in the paper) can be downloaded below. The plots are stored in the *plots* folder of the main directory whilst the saved models in the *checkpoints* folder in the same directory.

#### Priors

You can create again the Cookbook, Google and Phrasefinder prior using the following scripts:

```
python create_cookbook_prior.py
python create_google_prior.py
python create_phrasefinder_prior.py
```

The Phrasefinder script requires Python 3, if you use Python 2 use the requirements_py3.txt file to install the Python 3 environment separately.

The priors used in the paper can be downloaded below and are stored in the main directory. Change the path to any of these is changed, modify it accordingly in the *variables.json* file.

#### Action inference

Having the previous steps done and all paths correctly set, launch this script to do action inference.

```
python action_inference.py
```

You can also get the results per class (after the previous step) using this script:

```
python compute_action_results_per_class.py
```

### Downloads

* Splits: [[R split](https://drive.google.com/file/d/1D1dc6GjHHnKYhls9qOWZq-OCaeYvybWm/view?usp=sharing)][[NR split](https://drive.google.com/file/d/1aECI87Hfdl8onTWewYQ8RuFBfZKDrVeH/view?usp=sharing)]
* Saved models: [[R split verb detector](https://drive.google.com/file/d/1gNvSBC599n3hoE1YtoHf8HrW27IhC5WL/view?usp=sharing)][[R split object detector](https://drive.google.com/file/d/1wKH0MXm3cw_aR5eoe7doclREIhMO7DnN/view?usp=sharing)][[NR split verb detector](https://drive.google.com/file/d/1lPR1d7ZIlmmXl3t5_izD_bN56RFm7kDa/view?usp=sharing)][[NR split object detector](https://drive.google.com/file/d/1pKBNI72oxNigH5AscOPdhnyOuKB-mFMC/view?usp=sharing)]
* [[Scrapped Cookbook dataset](https://drive.google.com/file/d/1sb5GgGz5gurlLBNc_CHrVGRa-jX_ynrG/view?usp=sharing)]
* Action priors: [[Cookbook](https://drive.google.com/file/d/162WrjuAggDiFyfwDVicqqp7bRRcGwLwn/view?usp=sharing)][[Google](https://drive.google.com/file/d/1K5GY5SSDTlFlRbqytMm05CE-YukLBKsh/view?usp=sharing)][[Phrasefinder](https://drive.google.com/file/d/1jHz8GpJt05IYriR3WmcxmMeccig_2BJz/view?usp=sharing)]