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
        default="./descriptions",
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
        "--non_contrastive",
        action=argparse.BooleanOptionalAction,
        default=False,
        required=False,
        help="Use differential or non-differential prompts.",
    )

    args = vars(parser.parse_args())
    return args
