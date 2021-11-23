# IsoBN
Code for AAAI 2021 paper [IsoBN: Fine-Tuning BERT with Isotropic Batch Normalization](https://arxiv.org/abs/2005.02178).

## Requirements
* [PyTorch](http://pytorch.org/) (tested on 1.7.0)
* [Transformers](https://github.com/huggingface/transformers) (tested on 3.4.0)
* wandb
* tqdm
* datasets

## Training and Evaluation
Train the BERT-base model on GLUE with the following command:

```bash
>> python train.py --task_name TASK
```

The training loss and evaluation results on the dev set are synced to the wandb dashboard.
