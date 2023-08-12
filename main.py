import json
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import argument_parser
import fudd_model
import prompt_factory
import torch
from tqdm import tqdm

import utils


def evaluate(config):
    dataloader = utils.get_dataloader(config=config)

    prompt_factory_instance = prompt_factory.PromptFactory(
        prompt_root=config["prompt_root"],
        dataset_name=config["dataset"],
    )
    model = fudd_model.FuDD(
        top_k=config["topk"],
        prompt_factory=prompt_factory_instance,
        backbone=config["backbone"],
        step_one_source=config["source"],
        follow_up=config["follow_up"],
    )
    if torch.cuda.is_available():
        if torch.cuda.device_count() != 1:
            raise NotImplementedError
        model = model.cuda()

    num_correct = 0
    total_num = 0
    with torch.no_grad():
        for images_, targets_ in tqdm(dataloader):
            targets = targets_.cuda()
            images = images_.cuda()
            logits, preds = model(images)

            sorted_indices = torch.argsort(logits, descending=True, dim=-1)
            preds = preds[torch.arange(preds.shape[0])[:, None], sorted_indices]
            num_correct += (targets.view([-1, 1]) == preds[:, :1]).sum()
            total_num += targets.shape[0]

        acc = num_correct / total_num
        if isinstance(acc, torch.Tensor):
            acc = acc.item()

    print("Config:")
    print(json.dumps(config, indent=2))
    print("\n\n")
    print(f"Accuracy: {acc * 100}\n")

    run_summary = {"config": config, "accuracy": acc * 100}

    return run_summary


def main():
    start_time = time.time()
    config = argument_parser.parse_args()
    run_summary = evaluate(config)
    elapsed_time = time.time() - start_time

    if config.get("log_root") is not None:
        formatted_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        log_filename = Path(
            config["log_root"], formatted_timestamp + f"_{uuid4().hex}.json"
        )
        log_filename.parent.mkdir(exist_ok=True, parents=True)
        with open(log_filename, "w") as f:
            json.dump(
                {"run_summary": run_summary, "elapsed_time": elapsed_time}, f, indent=2
            )


if __name__ == "__main__":
    main()
