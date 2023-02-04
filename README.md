# Adversarial GLUE dataset

This is the official code base for our NeurIPS 2021 paper (Dataset and benchmark track, Oral presentation, 3.3% accepted rate)

[Adversarial GLUE: A Multi-Task Benchmark for Robustness Evaluation of Language Models](https://arxiv.org/abs/2111.02840)


Boxin Wang*, Chejian Xu*, Shuohang Wang, Zhe Gan, Yu Cheng, Jianfeng Gao, Ahmed Hassan Awadallah, Bo Li.

## Citation

```
@inproceedings{wang2021adversarial,
title={Adversarial GLUE: A Multi-Task Benchmark for Robustness Evaluation of Language Models},
author={Wang, Boxin and Xu, Chejian and Wang, Shuohang and Gan, Zhe and Cheng, Yu and Gao, Jianfeng and Awadallah, Ahmed Hassan and Li, Bo},
booktitle={Advances in Neural Information Processing Systems},
year={2021}
}
```

## Getting Started

We have built a few resources to help you get started with the dataset.

Download a copy of the dataset in `dev.json` (distributed under the CC BY-SA 4.0 license).

To evaluate your models, we have also made available the evaluation script we will use for official evaluation, along with a sample prediction file that the script will take as input. To run the evaluation, use 

```
python evaluate.py <path_to_dev> <path_to_predictions>
```

Once you have a built a model that works to your expectations on the dev set, you submit it to get official scores on the dev and a hidden test set. To preserve the integrity of test results, we do not release the test set to the public. Instead, we require you to submit your model so that we can run it on the test set for you. Here's a [tutorial](https://worksheets.codalab.org/worksheets/0x023aaebc1cd74f3fb8eccc57643687dd/) walking you through official evaluation of your model.

## Data Format

File ```dev.json``` contains the dev data of AdvGLUE dataset. Each task forms a key-value pair inside the json object. The structure of the file should look like:

```
{
  "sst2": sst2_item_list,
  "qqp": qqp_item_list,
  "mnli": mnli_item_list,
  "mnli-mm": mnli-mm_item_list,
  "qnli": qnli_item_list,
  "rte": rte_item_list
}
```

Items in different tasks have different formats. The format of each task is listed below:

  - **SST-2:** ```{'idx': index, 'label': label, 'sentence': text}```
  - **QQP:** ```{'idx': index, 'label': label, 'question1': text, 'question2': text}```
  - **MNLI:** ```{'idx': index, 'label': label, 'premise': text, 'hypothesis': text}```
  - **QNLI:** ```{'idx': index, 'label': label, 'question': text, 'sentence': text}```
  - **RTE:** ```{'idx': index, 'label': label, 'sentence1': text, 'sentence2': text}```


## Human Evaluation

We also provide the human evaluation scripts in the folder `human_eval/`.

## Contact

Ask us questions by creating GitHub issues or send emails at boxinw2@illinois.edu and xuchejian@zju.edu.cn.