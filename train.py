import argparse
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model import ClsModel
from utils import set_seed, collate_fn
from datasets import load_dataset, load_metric
import wandb
import warnings

warnings.filterwarnings("ignore")


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
}


def train(args, model, train_dataset, dev_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)
    total_steps = int(len(train_dataloader) * args.num_train_epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    no_decay = ["LayerNorm.weight", "bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    print('Total steps: {}'.format(total_steps))
    print('Warmup steps: {}'.format(warmup_steps))

    num_steps = 0
    for epoch in range(int(args.num_train_epochs)):
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            batch = {key: value.to(args.device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs[0]
            loss.backward()
            num_steps += 1
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            wandb.log({'loss': loss.item()}, step=num_steps)

        print(num_steps)
        results = evaluate(args, model, dev_dataset, tag="dev")
        print(results)
        wandb.log(results, step=num_steps)


def evaluate(args, model, eval_dataset, tag="train"):
    metric = load_metric("glue", args.task_name)

    def compute_metrics(preds, labels):
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=labels)
        if len(result) > 1:
            result["score"] = np.mean(list(result.values())).item()
        return result
    dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    label_list, logit_list = [], []
    for step, batch in enumerate(tqdm(dataloader)):
        model.eval()
        labels = batch["labels"].detach().cpu().numpy()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        batch["labels"] = None
        outputs = model(**batch)
        logits = outputs[0].detach().cpu().numpy()
        label_list.append(labels)
        logit_list.append(logits)
    preds = np.concatenate(logit_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    results = compute_metrics(preds, labels)
    results = {"{}_{}".format(tag, key): value for key, value in results.items()}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--task_name", default="rte", type=str)

    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--num_train_epochs", default=10.0, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project_name", type=str, default="IsoBN")
    parser.add_argument("--isobn", action="store_true")
    args = parser.parse_args()

    name = "{}-{}-{}".format(args.model_name_or_path, args.task_name, args.seed)
    if args.isobn:
        name += "-isobn"
    wandb.init(project=args.project_name, name=name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args.seed)

    datasets = load_dataset("glue", args.task_name)
    label_list = datasets["train"].features["label"].names
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    config.gradient_checkpointing = True
    config.isobn = args.isobn
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    model = ClsModel(args, config)
    model.to(0)

    sentence1_key, sentence2_key = task_to_keys[args.task_name]
    datasets = load_dataset("glue", args.task_name)

    def preprocess_function(examples):
        inputs = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key],)
        )
        result = tokenizer(*inputs, max_length=args.max_seq_length, truncation=True)
        result["labels"] = examples["label"] if 'label' in examples else 0
        return result

    train_dataset = list(map(preprocess_function, datasets["train"]))
    if args.task_name == 'mnli':
        dev_dataset = list(map(preprocess_function, datasets["validation_matched"]))
    else:
        dev_dataset = list(map(preprocess_function, datasets["validation"]))
    train(args, model, train_dataset, dev_dataset)


if __name__ == "__main__":
    main()
