import os
import json,pickle
import numpy as np
import pandas as pd
import argparse
import torch

from PIL import Image
from my_datasets import build_all_dataset,build_dataset
from my_utils import MetricLogger
from timm.utils import accuracy
import open_clip
from string import Template


device = 'cuda:0'

t = Template('a photo of $class_name')
select_path = f'./res/reuse_metrics'
save_path = f'./res/reuse'
if not os.path.exists(save_path):
    os.makedirs(save_path)
with open('./datasets.json', 'r') as dataset_name:
    dataset_hub = json.load(dataset_name)
dataset_list = dataset_hub['target_dataset']

with open('./pretrained.json','r') as dataset_name:
    pretrained_models = json.load(dataset_name)

pretrained_models_combined = {f'{model[0]}_{model[1]}':model for model in pretrained_models}



for dataset_name in os.listdir(select_path):

    if not os.path.isdir(os.path.join(select_path, dataset_name)) or dataset_name not in dataset_list:
        print(f'{dataset_name} not in dataset_hub')
        continue

    for setting in os.listdir(os.path.join(select_path, dataset_name)):
        import re
        if not re.match(r'^selected_\d+\.\d+_\d+\.json$',setting):
            continue
        alpha = setting.split('_')[1]
        k = setting.split('_')[2].split('.')[0]
        with open(os.path.join(select_path, dataset_name, f'selected_{alpha}_{k}.json'), 'r') as f:
            select_models_with_classnames = json.load(f)
        save_name = f'{dataset_name}_{alpha}_{k}'
        if os.path.exists(os.path.join(save_path,f'{save_name}.pkl')):
            print(f'{save_name}.pkl already exists')
            continue

        classnames = list(select_models_with_classnames.keys())
        models = set()
        for classname in classnames:
            models.update(select_models_with_classnames[classname])

        df = pd.DataFrame(0, index=list(models), columns=classnames)
        for classname, model_names in select_models_with_classnames.items():
            df.loc[model_names, classname] = 1

        all_output = {}
        target = None
        com_output = None
        for model_name in models:
            type = None
            model, _ , preprocess = open_clip.create_model_and_transforms(
                    model_name=pretrained_models_combined[model_name][0],pretrained=pretrained_models_combined[model_name][1],device=device)
            dataset = build_dataset(dataset_name, dataset_list[dataset_name]['root'], is_train=False, transform=preprocess)
            sampler = torch.utils.data.SequentialSampler(dataset)
            data_loader = torch.utils.data.DataLoader(
                dataset,
                sampler=sampler,
                batch_size= 32,
                num_workers=10,
                pin_memory=True,
                drop_last=False
            )

            def eval():
                combined_target = None
                combined_output = None
                if hasattr(data_loader.dataset,'classes'):
                    classnames = data_loader.dataset.classes
                elif hasattr(data_loader.dataset,'categories'):
                    classnames = data_loader.dataset.categories
                else:
                    raise Exception('No labels')
                if dataset_name == "ImageNet":
                    classnames = [",".join(label) for label in classnames]

                metric_logger = MetricLogger(delimiter="  ")
                header = f'{dataset_name}-{model_name}:'

                tokenizer = open_clip.get_tokenizer(pretrained_models_combined[model_name][0])
                text = [t.substitute(class_name=classname) for classname in classnames]
                text = tokenizer(classnames).to(device)
                for batch in metric_logger.log_every(data_loader, 15, header):
                    images, targets = batch
                    targets = targets.to(device)
                    images = images.to(device)
                    combined_target = targets if combined_target is None else torch.cat((combined_target,targets),dim=0)
                    text_special = text if model.context_length >= len(text[0]) else text[::, :model.context_length]
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        image_features = model.encode_image(images)
                        text_features = model.encode_text(text_special)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        outputs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                        combined_output = outputs if combined_output is None else torch.cat((combined_output,outputs),dim=0)
                return combined_target,combined_output

            target_tmp, output = eval()
            if target is None:
                target = target_tmp

            all_output.update({model_name:output})
            torch.cuda.empty_cache()
        com_output = None
        for model_name in df.index:
            output = np.array(all_output[model_name])
            probability_entropy = - 1 / np.sum(output * np.log2(output), axis = 1, keepdims=True)
            tmp = output * probability_entropy * df.loc[model_name].values
            com_output = tmp if com_output is None else com_output + tmp
        com_output = torch.tensor(com_output).to(device)
        pred = com_output.argmax(dim=1)
        acc1, acc5 = accuracy(com_output, target, topk=(1, 5))
        all_output = {key: value.detach().cpu() for key, value in all_output.items()}
        com_output = com_output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path,f'{save_name}.pkl'),'wb') as f:
            pickle.dump({
                'pred':pred,
                'target':target,
                'acc1':acc1,
                'acc5':acc5
                },f)




