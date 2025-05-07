import torch
import open_clip
import os
import json
import pickle
from tqdm import tqdm
from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd
from my_utils import MetricLogger
from my_datasets import build_dataset
from timm.utils import accuracy

root_path = './'
with open(f'{root_path}/pretrained.json','r') as f:
    pre_list = json.load(f)


label_path = f'{root_path}/label'
os.makedirs(label_path,exist_ok=True)
device = 'cuda:0'

true_all = []
true_positive_all = []
false_positive_all = []
for model_name, pretrained in pre_list[:int(len(pre_list)/3)]:
    if os.path.exists(f'{label_path}/{model_name}_{pretrained}.pkl'):
        print(f'{model_name}_{pretrained} already exists')
        continue
    with open(f'{label_path}/{model_name}_{pretrained}.pkl','wb') as f:
        pickle.dump({1:1},f)

    model, _ , preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,pretrained=pretrained,device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    dataset_name = 'evaluation_dataset'
    dataset = build_dataset(dataset_name,f'{root_path}/evaluation_dataset',is_train=False,transform=preprocess)
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=32,
        num_workers=10,
        drop_last=False
    )
    if hasattr(dataset,'classes'):
        node_names = dataset.classes
    elif hasattr(dataset,'categories'):
        node_names = dataset.categories
    else:
        continue
    model_label = [[] for _ in range(len(node_names))]
    text = [str(wn.synset(node).definition()) for node in node_names]
    text_token = tokenizer(text).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_embeddings = model.encode_text(text_token)
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    metric_logger = MetricLogger(delimiter="  ")
    header = f'{dataset_name}-{model_name}-{pretrained}:'
    for batch in metric_logger.log_every(data_loader, 15, header):
        images, targets = batch
        targets = targets.to(device)
        images = images.to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            img_embeddings = model.encode_image(images)
            img_embeddings /= img_embeddings.norm(dim=-1, keepdim=True)
            similarity = (100.0 * img_embeddings @ text_embeddings.T)
            for i, target in enumerate(targets):
                j = target.item()
                model_label[j].append(similarity[i].cpu().numpy())
    with open(f'{label_path}/{model_name}_{pretrained}.pkl','wb') as f:
        pickle.dump(model_label,f)


