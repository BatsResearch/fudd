from pathlib import Path

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class Places365(Dataset):
    def __init__(
        self,
        root,
        split="val",
        download=False,
        transform=None,
        target_transform=None,
        loader=default_loader,
    ) -> None:
        super().__init__()

        self.root = Path(root).joinpath("places365")
        self.split = split
        self.download = download
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        assert self.split in ["train", "val"]

        self.cls_folders = self.get_classnames()
        self.class_to_idx = {
            self.cls_folders[i]: i for i in range(len(self.cls_folders))
        }

        self.samples = list()
        for target_ in range(len(self.cls_folders)):
            cls = self.cls_folders[target_]
            cls_root = self.root.joinpath(self.split, cls)
            file_names = list(sorted([p.name for p in cls_root.iterdir()]))
            self.samples.extend(
                [(cls_root.joinpath(fn).as_posix(), target_) for fn in file_names]
            )

    def get_classnames(self):
        with self.root.joinpath("categories_places365.txt").open("r") as f:
            lines = f.readlines()
        cls_id_to_folders = {
            int(l.strip().split(" ")[-1]): l.strip()
            .split(" ")[0]
            .strip()
            .split("/", maxsplit=2)[-1]
            .replace("/", "-")
            for l in lines
        }
        classes = [cls_id_to_folders[i] for i in range(len(cls_id_to_folders))]
        return classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
