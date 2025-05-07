import pickle
import os
import json
import pandas as pd
import torch.utils
from tqdm import tqdm
import torch
from my_datasets import build_dataset
from my_utils import get_embedding
import argparse
def get_arg():
    parser = argparse.ArgumentParser(description="Evaluate and select models based on their performance.")
    parser.add_argument('--alpha', type=float, default=0.7, help='Weighting factor for F1 score adjustment.')
    parser.add_argument('--k', type=int, default=1, help='Number of top models to select per class.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the evaluation on.')
    parser.add_argument('--root_path', type=str, default='./MLL', help='Root path for datasets and results.')
    args = parser.parse_args()
    return args

def main(args):
    alpha = args.alpha
    k = args.k
    device = args.device
    root_path = args.root_path
    with open(f'{root_path}/dataset_hub.json','r') as f:
        dataset_hub = json.load(f)
    dataset_list = dataset_hub['target_dataset']
    # pre_list
    with open(f'{root_path}/pretrained.json','r') as f:
        pre_list = json.load(f)


    metric_path = f'{root_path}/res/reuse_metrics'
    label_path = f'{root_path}/label'

    os.makedirs(metric_path,exist_ok=True)
    evaluation_dataset = build_dataset('evaluation_dataset',f'{root_path}/evaluation_dataset')

    classes = evaluation_dataset.classes
    def eval_clip(model_name,pretrained,captions):
        if not os.path.exists(f'{label_path}/{model_name}_{pretrained}.pkl'):
            raise Exception(f'{model_name}_{pretrained} not exists')

        with open(f'{label_path}/{model_name}_{pretrained}.pkl','rb') as f:
            model_label = pickle.load(f)

        with open(f'{root_path}/caption_gen/embedding/evaluation_embedding.pkl','rb') as f:
            evaluation_caption_embeddings = pickle.load(f)

        num_classes = len(captions)
        num_embeddings = evaluation_caption_embeddings.shape[0]

        similarity_matrix = torch.zeros((num_classes, num_embeddings))
        classnames = list(captions.keys())  # 提取所有 classname
        idx_class_names = list(range(num_embeddings))  # 生成每个 idx_class 的列名

        for i, (classname, caption) in enumerate(captions.items()):
            caption_embedding = get_embedding(caption)
            similarity = (100.0 * caption_embedding @ evaluation_caption_embeddings.T)

            top5_similarity, top5_idx = similarity.topk(5)
            for sim, idx in zip(top5_similarity, top5_idx):
                similarity_matrix[i, idx] = sim.item()  # 将相似度值写入矩阵
        # 将相似度矩阵转换为 pandas DataFrame，行是 classname，列是 idx_class
        df_similarity = pd.DataFrame(similarity_matrix.cpu().numpy(), index=classnames, columns=idx_class_names)
        df_similarity_for_eval = df_similarity.applymap(lambda x: 1 if x != 0 else 0)

        true_positive = torch.zeros(num_embeddings, device=device,dtype=torch.int64)
        false_positive = torch.zeros(num_embeddings, device=device,dtype=torch.int64)
        true = torch.zeros(num_embeddings, device=device,dtype=torch.int64)

        index = f'{model_name}_{pretrained}'
        for idx, outputs in enumerate(model_label):
            outputs = outputs * df_similarity_for_eval[idx].values
            preds = outputs.argmax(dim=1)
            targets = torch.tensor([idx] * len(preds), device=device)
            true_positive += (preds == targets).int()
            false_positive += (preds != targets).int()
            true += outputs.sum(dim=0)
        for classname in classes:
            class_similarity = df_similarity[classname].values
            true_positive_for_classes = true_positive * torch.tensor(class_similarity, device=true_positive.device)
            false_positive_for_classes = false_positive * torch.tensor(class_similarity, device=false_positive.device)
            true_for_classes = true * torch.tensor(class_similarity, device=true.device)

        df_true_positive = pd.DataFrame(
            data=torch.stack(true_positive_for_classes).numpy().T,
            columns=classes
        )
        df_false_positive = pd.DataFrame(
            data=torch.stack(false_positive_for_classes).numpy().T,
            columns=classes
        )
        df_true = pd.DataFrame(
            data=torch.stack(true_for_classes).numpy().T,
            columns=classes
        )

        return df_true_positive,df_false_positive,df_true

    for dataset_name in dataset_list:
        if not os.path.exists(os.path.join(metric_path,dataset_name)):
            os.makedirs(os.path.join(metric_path,dataset_name))
        else:
            continue
        def dict_reverse(dict_origin):
            dict_target = dict()
            for key,values in dict_origin.items():
                for value in values:
                    if value not in dict_target:
                        dict_target.update({value:[key]})
                    else:
                        dict_target[value].append(key)
            return dict_target
        def eval(pre_list):
            dataset = build_dataset(dataset_name, dataset_list[dataset_name]['root'], is_train=False)
            if hasattr(dataset,'classes'):
                labels = dataset.classes
            elif hasattr(dataset,'categories'):
                labels = dataset.categories
            else:
                return
            if dataset_name == "ImageNet":
                labels = [",".join(label) for label in labels]
            with open(f'{root_path}/caption_gen/captions/{dataset_name}.json','r') as f:
                captions = json.load(f)
            df_TP = pd.DataFrame(columns=labels)
            df_FP = pd.DataFrame(columns=labels)
            df_T = pd.DataFrame(columns=labels)
            for model_name, pretrained in tqdm(pre_list):
                df_true_positive,df_false_positive,df_true = eval_clip(model_name,pretrained,captions)
                torch.cuda.empty_cache()
                df_TP = pd.concat([df_TP,df_true_positive[labels]],axis=0)
                df_FP = pd.concat([df_FP,df_false_positive[labels]],axis=0)
                df_T = pd.concat([df_T,df_true[labels]],axis=0)
            e = 1e-7
            df_recall = df_TP/(df_T+e)
            df_precision = df_TP/(df_TP+df_FP+e)
            df_f1 = 2*df_TP/(2*df_TP+df_FP+df_T+e)

            df_recall.to_csv(os.path.join(metric_path,dataset_name,f'recall.csv'))
            df_precision.to_csv(os.path.join(metric_path,dataset_name,f'precision.csv'))
            df_f1.to_csv(os.path.join(metric_path,dataset_name,f'f1.csv'))
            df_TP.to_csv(os.path.join(metric_path,dataset_name,f'TP.csv'))
            df_FP.to_csv(os.path.join(metric_path,dataset_name,f'FP.csv'))
            df_T.to_csv(os.path.join(metric_path,dataset_name,f'P.csv'))
            return
        eval(pre_list)

    select_path = "./MLL/res/selection"
    for dataset_name in dataset_list:
        if not os.path.exists(os.path.join(metric_path,dataset_name,f'f1.csv')):
            continue
        df_f1 = pd.read_csv(os.path.join(metric_path,dataset_name,f'f1.csv'),index_col=0)

        select_by_class = dict()
        for key, value in df_f1.sum(axis=1).items():
            df_f1.loc[key] = df_f1.loc[key] * alpha + (1 - alpha) * (value - df_f1.loc[key]) / (df_f1.shape[1] - 1)
        for column in df_f1.columns:
            select_by_class.update({column: df_f1[column].nlargest(k).index.tolist()})
        if not os.path.exists(os.path.join(select_path,dataset_name)):
            os.makedirs(os.path.join(select_path,dataset_name))
        with open(os.path.join(select_path,dataset_name,f'selected_{alpha}_{k}.json'),'w') as f:
            json.dump(select_by_class,f)

if __name__ == '__main__':
    args = get_arg()
    main(args)