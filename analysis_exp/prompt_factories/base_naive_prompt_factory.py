import json
from pathlib import Path

from . import prompt_factory_utils


def load_json(p):
    with open(p, "r") as f:
        obj = json.load(f)
    return obj


def generic_get_classnames(path, replace_chars=[]):
    cls_names = load_json(path)
    for c1, c2 in replace_chars:
        cls_names = [cn.replace(c1, c2) for cn in cls_names]
    return cls_names


def generic_get_iclr_prompts(prompt_path, classname_path=None):
    prompts = load_json(prompt_path)
    if classname_path is not None:
        cls_names = load_json(classname_path)
    else:
        cls_names = list(prompts.keys())

    prompts = [prompts[k] for k in cls_names]
    return prompts


class BaseNaivePromptFactory(object):
    def __init__(
        self,
        prompt_root,
        source,  # single_template, template_set, classname, iclr
        augment=False,
        extend=False,
    ) -> None:
        self.augment = augment
        self.extend = extend
        self.prompt_root = prompt_root
        self.source = source
        # self.class_names = load_json(
        #     Path(prompt_root, "imagenet/imagenet_classes.json")
        # )
        self.all_templates = load_json(Path(prompt_root, "clip_templates.json"))
        self.templates = [
            t.rstrip(".")
            for t in self.all_templates
            if t.endswith("{}") or t.endswith("{}.")
        ]

        self.possible_text_prefixes = [
            "a photo of",
            "a photograph of",
            "a video of",
            "a close-up of",
            "a screenshot of",
            "a close-up photo of",
            "a microscopic image of",
        ]
        self.default_template = "a photo of a {}."

        self.class_names = self.get_classnames()

        if source == "single_template":
            prompt_text_list = prompt_factory_utils.create_from_single_template(
                class_names=self.class_names, template=self.default_template
            )
        elif source == "template_set":
            prompt_text_list = prompt_factory_utils.create_from_template_set(
                class_names=self.class_names, templates=self.all_templates
            )
        elif source == "classname":
            prompt_text_list = prompt_factory_utils.create_from_classname(
                class_names=self.class_names
            )
        elif source == "iclr":
            # iclr_prompts = load_json(
            #     Path(
            #         prompt_root,
            #         "imagenet/descriptor_paper_iclr_prompts/imagenet_gpt.json",
            #     )
            # )
            # prompt_text_list = [v for v in iclr_prompts.values()]
            prompt_text_list = self.get_iclr_prompts()
        else:
            raise ValueError

        self.all_classes_prompts = [
            self.extend_structure_augment_texts(p) for p in prompt_text_list
        ]

    def get_classnames(self):
        raise NotImplementedError

    def get_iclr_prompts(self):
        raise NotImplementedError

    def extend_structure_augment_texts(self, text_list, capitalize=True):
        prompts = prompt_factory_utils.extend_structure_augment_texts(
            text_list=text_list,
            templates=self.templates,
            prefixes=self.possible_text_prefixes,
            augment=self.augment,
            extend=self.extend,
            capitalize=capitalize,
        )
        return prompts

    def get_prompts(self, class_ids=None, all_classes=False):
        if all_classes:
            return self.all_classes_prompts
        class_prompts = [self.all_classes_prompts[i] for i in class_ids]
        return class_prompts
