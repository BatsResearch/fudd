import json
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import acc_tracker
import argument_parser
import clip
import numpy as np
import torch
from modeling import fup_model, fup_task, main_task
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import get_contrastive_prompt_factory, get_dataset, get_naive_prompt_factory


def get_model(config):

    clip_model, _ = clip.load(config["backbone"], device="cpu")
    main_prompt_factory = get_naive_prompt_factory(config)

    main_task_instance = main_task.MainFupTask(
        prompt_factory=main_prompt_factory, tokenizer=clip.tokenize, cache="result"
    )

    fup_prompt_factory, nc_controls = get_contrastive_prompt_factory(config)
    if not config["non_contrastive"]:
        nc_controls = []

    fup_task_instance = fup_task.FupFupTask(
        topk=10,
        prompt_factory=fup_prompt_factory,
        tokenizer=clip.tokenize,
        cache="raw",
        non_contrastive=config["non_contrastive"],
        non_contrastive_controls=nc_controls,
    )

    fup_model_instance = fup_model.FupModel(
        model=clip_model,
        main_task=main_task_instance,
        fup_task=fup_task_instance,
        do_fup=True,
        single_return=False,
    )
    return fup_model_instance


def evaluate(config):
    acc_topks = [1, 5]
    acc_topks.append(config.get("model", {}).get("fup_task", {}).get("topk", 1))
    acc_topks = sorted(list(set(acc_topks)))

    main_acc_metric = acc_tracker.CustomAccMetric(topk_list=acc_topks)
    fup_acc_metric = acc_tracker.CustomAccMetric(topk_list=acc_topks)

    dataset = get_dataset(config)
    dataloader = DataLoader(dataset=dataset, batch_size=32, num_workers=8)

    model = get_model(config)

    if torch.cuda.is_available():
        if torch.cuda.device_count() != 1:
            raise NotImplementedError
        model = model.cuda()

    with torch.no_grad():
        for images_, targets_ in tqdm(dataloader):
            targets = targets_.cuda()
            images = images_.cuda()
            output = model(images)

            main_acc_metric.add_batch(
                targets=targets,
                preds=output["main"]["preds"],
                logits=output["main"]["logits"],
                order_pred_by_logit=True,
            )
            if output.get("fup"):
                fup_acc_metric.add_batch(
                    targets=targets,
                    preds=output["fup"]["preds"],
                    logits=output["fup"]["logits"],
                    order_pred_by_logit=True,
                )

        perf = {"main": main_acc_metric.compute(percent=True)}
        if fup_acc_metric.total_num != 0:
            perf["fup"] = fup_acc_metric.compute(percent=True)
        else:
            perf["fup"] = {k: 0 for k in perf["main"].keys()}

    print(json.dumps(config, indent=2))
    print("\n\n")
    accuracy = perf["fup"][1]
    print(f"Accuracy: {accuracy}\n")
    print("\n\n")

    if config.get("log_root") is not None:
        formatted_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        log_filename = Path(
            config["log_root"], formatted_timestamp + f"_{uuid4().hex}.json"
        )
        log_filename.parent.mkdir(exist_ok=True, parents=True)
        with open(log_filename, "w") as f:
            json.dump(
                {"config": config, "perf": perf, "accuracy": accuracy}, f, indent=2
            )

    return perf


def main():
    config = argument_parser.parse_args()
    np.random.seed(0)
    evaluate(config)


if __name__ == "__main__":
    main()
