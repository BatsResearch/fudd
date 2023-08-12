# Datasets

For [DTD](https://pytorch.org/vision/main/generated/torchvision.datasets.DTD.html), [EuroSAT](https://pytorch.org/vision/main/generated/torchvision.datasets.EuroSAT.html#torchvision.datasets.EuroSAT), [FGVCAircraft](https://pytorch.org/vision/main/generated/torchvision.datasets.FGVCAircraft.html#torchvision.datasets.FGVCAircraft), [Flowers102](https://pytorch.org/vision/main/generated/torchvision.datasets.Flowers102.html#torchvision.datasets.Flowers102), [Food101](https://pytorch.org/vision/main/generated/torchvision.datasets.Food101.html#torchvision.datasets.Food101), [ImageNet](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageNet.html#torchvision.datasets.ImageNet), [Oxford Pets](https://pytorch.org/vision/main/generated/torchvision.datasets.OxfordIIITPet.html#torchvision.datasets.OxfordIIITPet), and [Stanford Cars](https://pytorch.org/vision/main/generated/torchvision.datasets.StanfordCars.html#torchvision.datasets.StanfordCars) datasets, we use the standard dataset classes that come with `torchvision`.
Creating the dataset directory structures according to official instructions should work with this code base. The value of the `--root` argument of the main script is passed to the `root` argument of the `torchvision` dataset classes directly.

For ImageNetV2, we use [this implementation](https://github.com/modestyachts/ImageNetV2_pytorch), which should download the dataset itself in the given directory.

## Cub

Download the images and annotations from [here](https://www.vision.caltech.edu/datasets/cub_200_2011/). Extract the files in the `DATA_ROOT` directory and pass `DATA_ROOT` as the `--root` argument of the main script.

## Places365

Download the Places365 dataset from [here](http://places2.csail.mit.edu/download.html).
In the `DATA_ROOT` directory, create another directory named `places365`.
In this directory, place the `categories_places365.txt` and `val.txt` files as well as the `val` folder containing the images. Pass `DATA_ROOT` as the `--root` argument of the main script.

## Stanford Dogs

Set the `DATA_ROOT` environment variable and run the following script to setup the stanford dogs dataset.

```bash
cd $DATA_ROOT
mkdir stanford_dogs
cd stanford_dogs
curl -OLJ http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
curl -OLJ http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar
curl -OLJ http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar
curl -OLJ http://vision.stanford.edu/aditya86/ImageNetDogs/README.txt
tar -xvf images.tar
tar -xvf lists.tar
```

Pass `DATA_ROOT` as the `--root` argument of the main script.
