from nltk.corpus import wordnet as wn
from copy import deepcopy
import json
import sys
from my_datasets import build_dataset
import random
import re
import os
import shutil
from tqdm import tqdm
datasets = ['ImageNet','imagenet_a','imagenet-sketch','imagenet_v2','imagenet_r_origin',]
with open('./dataset_hub.json','r') as f:
    dataset_hub = json.load(f)
dataset_list = dataset_hub['target_dataset']
for dataset_name in datasets:
    with open(f"./gpt_gen/classnames/{dataset_name}.txt", "r") as f:
        lines = f.readlines()
        classnames = [line.strip() for line in lines]
    # tench,Tinca tinca
    # goldfish,Carassius auratus
    # great white shark,white
    synsets_all = dict()
    for i,classname in enumerate(classnames):
        classnames_split = classname.split(',')
        synsets = []
        for classname_split in classnames_split:
            classname_split = classname_split.replace(' ', '_')
            synset = wn.synsets(classname_split)
            # print(classname_split)
            # synsets.append(synset)
            if len(synset) != 0:
                for s in synset:
                    if s in synsets_all.keys():
                        synsets_all[s].update([i])
                    else:
                        synsets_all[s] = set([i])
    dataset = build_dataset(dataset_name,dataset_list[dataset_name]['root'],False)
    target_i_index = []
    dataset_path = './evaluation_dataset'
    targets = None
    imgs = None
    if hasattr(dataset,'targets'):
        targets = dataset.targets
    elif hasattr(dataset,'_labels'):
        targets = dataset._labels
    if hasattr(dataset,'imgs'):
        imgs = dataset.imgs
    elif hasattr(dataset,'_image_files'):
        imgs = dataset._image_files
    pass
    for i in range(len(classnames)):
        target_i_index.append([index for index, value in enumerate(targets) if value == i])
    for synset,indexs in tqdm(synsets_all.items()):
        pattern = r"Synset\('([\s \S]+)'\)"
        classnames = re.findall(pattern,str(synset))[0].replace('/','*')

        index_for_select = set()
        for index in indexs:
            index_for_select.update(target_i_index[index])
        index_for_select = list(index_for_select)
        if len(index_for_select) > 25:
            selected_targets = random.sample(index_for_select, 25)
        else:
            selected_targets = index_for_select
        target_path = os.path.join(dataset_path,classnames)
        os.makedirs(target_path, exist_ok=True)
        for target in selected_targets:
            if type(imgs[target]) == tuple:
                source_img_path = imgs[target][0]
            else:
                source_img_path = imgs[target]
            target_img_path = os.path.join(target_path,os.path.basename(source_img_path))
            shutil.copy(source_img_path,target_img_path)
        for root, dirs, files in os.walk(target_path):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                files_in_dir = os.listdir(dir_path)
                if len(files_in_dir) > 75:
                    files_to_delete = random.sample(files_in_dir, len(files_in_dir) - 75)
                    for file in files_to_delete:
                        os.remove(os.path.join(dir_path, file))

def dict_merge(dict1:dict,dict2:dict):
    dict_all = dict()
    for key in dict1.keys():
        dict_all[key] = dict1[key]
    for key in dict2.keys():
        if key in dict_all.keys():
            dict_all[key].extend(dict2[key])
        else:
            dict_all[key] = dict2[key]
    return dict_all
synsets = list()
for classname in os.listdir(dataset_path):
    classname = classname.replace('*','/')
    synset = wn.synset(classname)
    synsets.append(synset)
def get_hyper_hypo(synset,type):
    hyper_hypo_all = dict()
    for synset in synsets:
        if type == 'hypernyms':
            hyper_hypo_list = synset.hypernyms()
        elif type == 'hyponyms':
            hyper_hypo_list = synset.hyponyms()
        else:
            raise ValueError('type must be hypernyms or hyponyms')
        for hyper_hypo in hyper_hypo_list:
            if hyper_hypo in hyper_hypo_all.keys():
                hyper_hypo_all[hyper_hypo].append(synset)
            else:
                hyper_hypo_all[hyper_hypo] = [synset]
    return hyper_hypo_all
def get_all_hyper_hypo(synsets,type):
    hyper_hypo_all = get_hyper_hypo(synsets,type)
    count = 5
    while True:
        hyper_hypo_all_tmp = hyper_hypo_all
        hyper_hypo_all = dict_merge(get_hyper_hypo(hyper_hypo_all_tmp.keys(),type),hyper_hypo_all_tmp)
        if len(hyper_hypo_all_tmp) == len(hyper_hypo_all):
            count -= 1
        if count == 0:
            break
    for k,v in hyper_hypo_all.items():
        hyper_hypo_all[k] = list(set(v))
    return hyper_hypo_all

hypernyms = get_all_hyper_hypo(synsets,'hypernyms')
hyponyms = get_all_hyper_hypo(synsets,'hyponyms')
for hyponym, categories in hyponyms.items():
    hyponym_name = re.findall(r"Synset\('([\s \S]+)'\)", str(hyponym))[0].replace('/', '*')
    hyponym_path = os.path.join(dataset_path, hyponym_name)
    os.makedirs(hyponym_path, exist_ok=True)

    for category in categories:
        category_name = re.findall(r"Synset\('([\s \S]+)'\)", str(category))[0].replace('/', '*')
        category_path = os.path.join(dataset_path, category_name)

        if os.path.exists(category_path):
            files = os.listdir(category_path)
            if len(files) > 75 // len(categories):
                files = random.sample(files, 75 // len(categories))

            for file in files:
                shutil.copy(os.path.join(category_path, file), os.path.join(hyponym_path, file))
