from pathlib import Path

import clip
import cub_dataset
import places365_dataset
import stanford_dogs_dataset
import torchvision
from prompt_factories import (
    contrastive_prompt_factory_collection,
    naive_prompt_factory_collection,
)


def get_naive_prompt_factory(config):
    if config["dataset"] == "cub":
        npf = naive_prompt_factory_collection.CubNaivePromptFactory(
            prompt_root=config["prompt_root"], source="single_template"
        )
    elif config["dataset"] == "dtd":
        npf = naive_prompt_factory_collection.DTDNaivePromptFactory(
            prompt_root=config["prompt_root"], source="single_template"
        )
    elif config["dataset"] == "fgvc_aircraft":
        npf = naive_prompt_factory_collection.FGVCAircraftNaivePromptFactory(
            prompt_root=config["prompt_root"], source="single_template"
        )
    elif config["dataset"] == "flowers":
        npf = naive_prompt_factory_collection.Flowers102NaivePromptFactory(
            prompt_root=config["prompt_root"], source="single_template"
        )
    elif config["dataset"] == "food101":
        npf = naive_prompt_factory_collection.Food101NaivePromptFactory(
            prompt_root=config["prompt_root"], source="single_template"
        )
    elif config["dataset"] == "pets":
        npf = naive_prompt_factory_collection.OxfordIIITPetNaivePromptFactory(
            prompt_root=config["prompt_root"], source="single_template"
        )
    elif config["dataset"] == "places365":
        npf = naive_prompt_factory_collection.Places365NaivePromptFactory(
            prompt_root=config["prompt_root"], source="single_template"
        )
    elif config["dataset"] == "stanford_cars":
        npf = naive_prompt_factory_collection.StanfordCarsNaivePromptFactory(
            prompt_root=config["prompt_root"], source="single_template"
        )
    elif config["dataset"] == "stanford_dogs":
        npf = naive_prompt_factory_collection.StanfordDogsNaivePromptFactory(
            prompt_root=config["prompt_root"], source="single_template"
        )
    else:
        raise ValueError
    return npf


def get_contrastive_prompt_factory(config):
    if config["dataset"] == "cub":
        cpf = contrastive_prompt_factory_collection.CubContrastivePromptFactory(
            prompt_root=config["prompt_root"], source="contrastive_gpt", augment=True
        )
        nc_controls = ["prompts", "attr_types", "attr_words"]
    elif config["dataset"] == "dtd":
        cpf = contrastive_prompt_factory_collection.DTDContrastivePromptFactory(
            prompt_root=config["prompt_root"], source="contrastive_gpt", augment=True
        )
        nc_controls = ["prompts", "attr_types"]
    elif config["dataset"] == "fgvc_aircraft":
        cpf = (
            contrastive_prompt_factory_collection.FGVCAircraftContrastivePromptFactory(
                prompt_root=config["prompt_root"],
                source="contrastive_gpt",
                augment=True,
            )
        )
        nc_controls = ["prompts", "attr_types", "attr_words"]
    elif config["dataset"] == "flowers":
        cpf = contrastive_prompt_factory_collection.Flowers102ContrastivePromptFactory(
            prompt_root=config["prompt_root"], source="contrastive_gpt", augment=True
        )
        nc_controls = ["prompts", "attr_types", "attr_words"]
    elif config["dataset"] == "food101":
        cpf = contrastive_prompt_factory_collection.Food101ContrastivePromptFactory(
            prompt_root=config["prompt_root"], source="contrastive_gpt", augment=True
        )
        nc_controls = ["prompts", "attr_types", "attr_words"]
    elif config["dataset"] == "pets":
        cpf = (
            contrastive_prompt_factory_collection.OxfordIIITPetContrastivePromptFactory(
                prompt_root=config["prompt_root"],
                source="contrastive_gpt",
                augment=True,
            )
        )
        nc_controls = ["prompts", "attr_types"]
    elif config["dataset"] == "places365":
        cpf = contrastive_prompt_factory_collection.Places365ContrastivePromptFactory(
            prompt_root=config["prompt_root"], source="contrastive_gpt", augment=True
        )
        nc_controls = ["prompts", "attr_types", "attr_words"]
    elif config["dataset"] == "stanford_cars":
        cpf = (
            contrastive_prompt_factory_collection.StanfordCarsContrastivePromptFactory(
                prompt_root=config["prompt_root"],
                source="contrastive_gpt",
                augment=True,
            )
        )
        nc_controls = ["prompts", "attr_types", "attr_words"]
    elif config["dataset"] == "stanford_dogs":
        cpf = (
            contrastive_prompt_factory_collection.StanfordDogsContrastivePromptFactory(
                prompt_root=config["prompt_root"],
                source="contrastive_gpt",
                augment=True,
            )
        )
        nc_controls = ["prompts", "attr_types"]
    else:
        raise ValueError

    return cpf, nc_controls


def get_dataset(config):
    _, preprocess = clip.load(config["backbone"], device="cpu")
    if config["dataset"] == "cub":
        dataset = cub_dataset.Cub2011(
            root=config["root"],
            train=False,
            transform=preprocess,
        )
    elif config["dataset"] == "dtd":
        dataset = torchvision.datasets.DTD(
            root=config["root"], split="test", transform=preprocess
        )
    elif config["dataset"] == "fgvc_aircraft":
        dataset = torchvision.datasets.FGVCAircraft(
            root=config["root"], transform=preprocess, split="test"
        )
    elif config["dataset"] == "flowers":
        dataset = torchvision.datasets.Flowers102(
            root=config["root"], transform=preprocess, split="test"
        )
    elif config["dataset"] == "food101":
        dataset = torchvision.datasets.Food101(
            root=config["root"], transform=preprocess, split="test"
        )
    elif config["dataset"] == "pets":
        dataset = torchvision.datasets.OxfordIIITPet(
            root=config["root"], transform=preprocess, split="test"
        )
    elif config["dataset"] == "places365":
        dataset = places365_dataset.Places365(
            root=config["root"], transform=preprocess, split="val"
        )
    elif config["dataset"] == "stanford_cars":
        dataset = torchvision.datasets.StanfordCars(
            root=config["root"], transform=preprocess, split="test"
        )
    elif config["dataset"] == "stanford_dogs":
        dataset = stanford_dogs_dataset.StanfordDogs(
            root=config["root"], transform=preprocess, split="test"
        )
    else:
        raise ValueError
    return dataset
