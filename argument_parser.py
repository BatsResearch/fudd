import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        help="Data root.",
        required=True,
    )
    parser.add_argument(
        "--prompt_root",
        type=str,
        default="./differential_descriptions",
        help="Path to class description files.",
        required=False,
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default=None,
        help="If and where to save the json log file.",
        required=False,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Target dataset.",
        required=True,
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="ViT-B/32",
        help="CLIP backbone.",
        required=False,
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        required=False,
        help="Number of ambiguous classes.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel dataloader workers.",
        required=False,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        required=False,
        help="Dataloader batch size.",
    )
    parser.add_argument(
        "--follow_up",
        action=argparse.BooleanOptionalAction,
        default=True,
        required=False,
        help="Whether to solve the follow-up classification or not.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="gpt",
        choices=["gpt", "classname", "single_template", "template_set"],
        help="How to create the descriptions.",
        required=False,
    )

    args = vars(parser.parse_args())

    if args["follow_up"] and args["source"] != "gpt":
        raise ValueError(
            "You can only solve the follow-up classification problem with GPT prompts."
        )

    return args
