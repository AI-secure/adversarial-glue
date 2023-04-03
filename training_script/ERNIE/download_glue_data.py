import os
from datasets import load_dataset

one_sentence_tasks = ['cola', 'sst2']
two_sentence_tasks = ['mrpc', 'stsb', 'qqp', 'mnli', 'mnli-mm', 'qnli', 'rte', 'wnli']

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

folder_to_split = {'train': 'train', 'dev': 'validation', 'test': 'test'}
dataset_root = './data/glue'
os.makedirs(dataset_root, exist_ok=True)

if __name__ == '__main__':
    for task in one_sentence_tasks + two_sentence_tasks:
        dataset = load_dataset('glue', task if task != 'mnli-mm' else 'mnli')
        for folder in ['train', 'dev', 'test']:
            if folder == 'dev':
                split = 'validation'
                if task == 'mnli':
                    split = 'validation_matched'
                elif task == 'mnli-mm':
                    split = 'validation_mismatched'
            elif folder == 'test':
                split = 'test'
                if task == 'mnli':
                    split = 'test_matched'
                elif task == 'mnli-mm':
                    split = 'test_mismatched'
            else:
                split = folder
            data_path = os.path.join(dataset_root, task, folder)
            os.makedirs(data_path, exist_ok=True)
            data_list = dataset[split]
            lines = []
            sentence1_key, sentence2_key = task_to_keys[task if task != 'mnli-mm' else 'mnli']
            for item in data_list:
                if sentence2_key is None:
                    line = item[sentence1_key] + '\t' + str(item['label']) + '\n'
                else:
                    line = item[sentence1_key] + '\t' + item[sentence2_key] + '\t' + str(item['label']) + '\n'
                lines.append(line)
            file_name = os.path.join(data_path, 'data')
            with open(file_name, 'w') as f:
                f.writelines(lines)
