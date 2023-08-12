from pathlib import Path

from . import base_contrastive_prompt_factory, base_naive_prompt_factory


class CubContrastivePromptFactory(
    base_contrastive_prompt_factory.BaseContrastivePromptFactory
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_classnames(self):
        classnames_path = Path(self.prompt_root, "cub_class_names.json")
        return base_naive_prompt_factory.generic_get_classnames(
            classnames_path, replace_chars=[("_", " ")]
        )

    def get_prompt_pairs(self):
        if self.source == "contrastive_gpt":
            return base_naive_prompt_factory.load_json(
                Path(
                    self.prompt_root,
                    f"cub_prompt_pairs.json",
                )
            )
        else:
            raise ValueError


class DTDContrastivePromptFactory(
    base_contrastive_prompt_factory.BaseContrastivePromptFactory
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_classnames(self):
        classnames_path = Path(self.prompt_root, "dtd_class_names.json")
        return base_naive_prompt_factory.generic_get_classnames(
            classnames_path, replace_chars=[("_", " ")]
        )

    def get_prompt_pairs(self):
        if self.source == "contrastive_gpt":
            return base_naive_prompt_factory.load_json(
                Path(
                    self.prompt_root,
                    f"dtd_prompt_pairs.json",
                )
            )
        else:
            raise ValueError


class Food101ContrastivePromptFactory(
    base_contrastive_prompt_factory.BaseContrastivePromptFactory
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_classnames(self):
        classnames_path = Path(self.prompt_root, "food101_class_names.json")
        return base_naive_prompt_factory.generic_get_classnames(
            classnames_path, replace_chars=[("_", " ")]
        )

    def get_prompt_pairs(self):
        if self.source == "contrastive_gpt":
            return base_naive_prompt_factory.load_json(
                Path(
                    self.prompt_root,
                    f"food101_prompt_pairs.json",
                )
            )
        else:
            raise ValueError


class OxfordIIITPetContrastivePromptFactory(
    base_contrastive_prompt_factory.BaseContrastivePromptFactory
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_classnames(self):
        classnames_path = Path(self.prompt_root, "pets_class_names.json")
        return base_naive_prompt_factory.generic_get_classnames(
            classnames_path, replace_chars=[("_", " ")]
        )

    def get_prompt_pairs(self):
        if self.source == "contrastive_gpt":
            return base_naive_prompt_factory.load_json(
                Path(
                    self.prompt_root,
                    f"pets_prompt_pairs.json",
                )
            )
        else:
            raise ValueError


class Places365ContrastivePromptFactory(
    base_contrastive_prompt_factory.BaseContrastivePromptFactory
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_classnames(self):
        classnames_path = Path(self.prompt_root, "places365_class_names.json")
        return base_naive_prompt_factory.generic_get_classnames(
            classnames_path, replace_chars=[("_", " ")]
        )

    def get_prompt_pairs(self):
        if self.source == "contrastive_gpt":
            return base_naive_prompt_factory.load_json(
                Path(
                    self.prompt_root,
                    f"places365_prompt_pairs.json",
                )
            )
        else:
            raise ValueError


class FGVCAircraftContrastivePromptFactory(
    base_contrastive_prompt_factory.BaseContrastivePromptFactory
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_classnames(self):
        classnames_path = Path(self.prompt_root, "fgvc_aircraft_class_names.json")
        return base_naive_prompt_factory.generic_get_classnames(
            classnames_path, replace_chars=[("_", " ")]
        )

    def get_prompt_pairs(self):
        if self.source == "contrastive_gpt":
            return base_naive_prompt_factory.load_json(
                Path(
                    self.prompt_root,
                    f"fgvc_aircraft_prompt_pairs.json",
                )
            )
        else:
            raise ValueError


class Flowers102ContrastivePromptFactory(
    base_contrastive_prompt_factory.BaseContrastivePromptFactory
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_classnames(self):
        classnames_path = Path(self.prompt_root, "flowers_class_names.json")
        return base_naive_prompt_factory.generic_get_classnames(
            classnames_path, replace_chars=[("_", " ")]
        )

    def get_prompt_pairs(self):
        if self.source == "contrastive_gpt":
            return base_naive_prompt_factory.load_json(
                Path(
                    self.prompt_root,
                    f"flowers_prompt_pairs.json",
                )
            )
        else:
            raise ValueError


class StanfordCarsContrastivePromptFactory(
    base_contrastive_prompt_factory.BaseContrastivePromptFactory
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_classnames(self):
        classnames_path = Path(self.prompt_root, "stanford_cars_class_names.json")
        return base_naive_prompt_factory.generic_get_classnames(
            classnames_path, replace_chars=[("_", " ")]
        )

    def get_prompt_pairs(self):
        if self.source == "contrastive_gpt":
            return base_naive_prompt_factory.load_json(
                Path(
                    self.prompt_root,
                    f"stanford_cars_prompt_pairs.json",
                )
            )
        else:
            raise ValueError


class StanfordDogsContrastivePromptFactory(
    base_contrastive_prompt_factory.BaseContrastivePromptFactory
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_classnames(self):
        classnames_path = Path(self.prompt_root, "stanford_dogs_class_names.json")
        return base_naive_prompt_factory.generic_get_classnames(
            classnames_path, replace_chars=[("_", " ")]
        )

    def get_prompt_pairs(self):
        if self.source == "contrastive_gpt":
            return base_naive_prompt_factory.load_json(
                Path(
                    self.prompt_root,
                    f"stanford_dogs_prompt_pairs.json",
                )
            )
        else:
            raise ValueError
