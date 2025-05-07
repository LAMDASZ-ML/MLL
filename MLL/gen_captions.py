import json
from tqdm import tqdm
from openai import OpenAI
import os
from my_datasets import build_all_dataset


def openai_request(client,query):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages= [{"role": "user", "content":query}],

        )
    except Exception as e:
        print("############## fail #1, pausing and trying again")
        import time
        time.sleep(150)
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages= [{"role": "user", "content":query}],
            )
        except:
            print([{"role": "user", "content":query}])

    res = dict(dict(dict(response)['choices'][0])['message'])['content'].split('\n\n')
    return res

def main():
    def class_gen():
        with open('./dataset_hub.json','r') as f:
            dataset_hub = json.load(f)
        dataset_list = dataset_hub['target_dataset']
        all_datasets = build_all_dataset(dataset_list,is_train=False,transform=None)
        for dataset_name in dataset_list:
            dataset = all_datasets[dataset_name]
            if hasattr(dataset,'classes'):
                labels = dataset.classes
            elif hasattr(dataset,'categories'):
                labels = dataset.categories
            else:
                continue
            if dataset_name == "ImageNet":
                labels = [",".join(label) for label in labels]
            with open(f'{root_path}/classnames/{dataset_name}.txt','w') as f:
                for label in labels:
                    f.write(label+'\n')

    client = OpenAI()
    root_path = './caption_gen'
    caption_length = 50
    caption_dir = f'{root_path}/captions'

    with open('./dataset_hub.json','r') as f:
        dataset_hub = json.load(f)
    dataset_list = dataset_hub['target_dataset']

    text_prompt = lambda classname, domain, task, length: f"Generate long detailed caption for the {domain} of {classname} in the {task}.
    e.g., “ The {domain} of {classname}, which is ... ”. Generate long caption for cat within {length} words"

    class_gen()
    for dataset_name in tqdm(dataset_list):
        res = {}
        dataset = dataset_list[dataset_name]
        with open(f'{root_path}/classnames/{dataset_name}.txt', 'r') as file:
            lines = file.readlines()
        classnames = [line.rstrip() for line in lines]
        for classname in tqdm(classnames, desc=dataset_name+', Class', leave=False):
            try:
                response = openai_request(client,text_prompt(classname, dataset['domain'], dataset['task'], caption_length))
                res[classname] = response
            except Exception as e:
                print(f"Failed for {classname} in {dataset_name} with error: {e}")
        if not os.path.exists(caption_dir):
            os.makedirs(caption_dir)
        with open(f"{caption_dir}/{dataset_name}.json", "w") as f:
            json.dump(res, f)

if __name__ == "__main__":
    main()


