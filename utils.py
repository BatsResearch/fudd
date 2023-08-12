from pathlib import Path

import clip
import cub_dataset
import imagenetv2_pytorch
import places365_dataset
import stanford_dogs_dataset
import torchvision
from torch.utils.data import DataLoader


def get_dataloader(config):
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
    elif config["dataset"] == "eurosat":
        dataset = torchvision.datasets.EuroSAT(
            root=config["root"], transform=preprocess
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
    elif config["dataset"] == "imagenet":
        dataset = torchvision.datasets.ImageNet(
            root=config["root"], transform=preprocess, split="val"
        )
    elif config["dataset"] == "imagenet_v2":
        dataset = imagenetv2_pytorch.ImageNetV2Dataset(
            location=config["root"],
            transform=preprocess,
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

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )
    return dataloader
