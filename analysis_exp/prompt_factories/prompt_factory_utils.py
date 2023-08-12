import copy
import re


def extend_structure_augment_texts(
    text_list,
    templates,
    prefixes,
    augment,
    extend,
    capitalize,
):
    if extend:
        text_list = extend_text_list(
            text_list=text_list,
            templates=templates,
            prefixes=prefixes,
        )
    prompts = structure_texts(text_list)

    if augment:
        prompts = augment_prompt_texts(
            prompts=prompts,
            templates=templates,
            prefixes=prefixes,
        )

    if capitalize:
        prompts = {
            "content": [c.capitalize() for c in prompts["content"]],
            "text": {
                k.capitalize(): [i.capitalize() for i in v]
                for k, v in prompts["text"].items()
            },
        }
    return prompts


def text_variants(text, templates, prefixes):
    original_text = text.lower()
    text = original_text
    for ii in prefixes:
        text = text.replace(ii, "")
    text = text.strip()
    text = re.sub(r"^an? \s*", "", text, count=1)
    variants = [t.format(text) for t in templates] + [original_text]
    variants = list(set(variants))
    return variants


def extend_text_list(text_list, templates, prefixes):
    extended_list = list()
    for text in text_list:
        extended_list.extend(text_variants(text, templates, prefixes))
    extended_list = list(set(extended_list))
    return extended_list


def structure_texts(text_list):
    prompts = {"content": text_list, "text": {t: [t] for t in text_list}}
    return prompts


def augment_prompt_texts(prompts, templates, prefixes):
    prompts_ = copy.deepcopy(prompts)
    for c in prompts_["content"]:
        prompts_["text"][c] = extend_text_list(
            text_list=prompts_["text"][c], templates=templates, prefixes=prefixes
        )
    return prompts_


def create_from_single_template(class_names, template):
    return [[template.format(c)] for c in class_names]


def create_from_template_set(class_names, templates):
    return [[t.format(c) for t in templates] for c in class_names]


def create_from_classname(class_names):
    return [[f"{c}."] for c in class_names]
