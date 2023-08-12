from pathlib import Path

from . import base_naive_prompt_factory


class CubNaivePromptFactory(base_naive_prompt_factory.BaseNaivePromptFactory):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_classnames(self):
        classnames_path = Path(self.prompt_root, "cub_class_names.json")
        return base_naive_prompt_factory.generic_get_classnames(
            classnames_path, replace_chars=[("_", " ")]
        )

    def get_iclr_prompts(self):
        raise RuntimeError


class DTDNaivePromptFactory(base_naive_prompt_factory.BaseNaivePromptFactory):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_classnames(self):
        classnames_path = Path(self.prompt_root, "dtd_class_names.json")
        return base_naive_prompt_factory.generic_get_classnames(
            classnames_path, replace_chars=[("_", " ")]
        )

    def get_iclr_prompts(self):
        raise RuntimeError


class Food101NaivePromptFactory(base_naive_prompt_factory.BaseNaivePromptFactory):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_classnames(self):
        classnames_path = Path(self.prompt_root, "food101_class_names.json")
        return base_naive_prompt_factory.generic_get_classnames(
            classnames_path, replace_chars=[("_", " ")]
        )

    def get_iclr_prompts(self):
        raise RuntimeError


class OxfordIIITPetNaivePromptFactory(base_naive_prompt_factory.BaseNaivePromptFactory):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_classnames(self):
        classnames_path = Path(self.prompt_root, "pets_class_names.json")
        return base_naive_prompt_factory.generic_get_classnames(
            classnames_path, replace_chars=[("_", " ")]
        )

    def get_iclr_prompts(self):
        raise RuntimeError


class Places365NaivePromptFactory(base_naive_prompt_factory.BaseNaivePromptFactory):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_classnames(self):
        classnames_path = Path(self.prompt_root, "places365_class_names.json")
        return base_naive_prompt_factory.generic_get_classnames(
            classnames_path, replace_chars=[("_", " ")]
        )

    def get_iclr_prompts(self):
        raise RuntimeError


class FGVCAircraftNaivePromptFactory(base_naive_prompt_factory.BaseNaivePromptFactory):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_classnames(self):
        classnames_path = Path(self.prompt_root, "fgvc_aircraft_class_names.json")
        return base_naive_prompt_factory.generic_get_classnames(
            classnames_path, replace_chars=[("_", " ")]
        )

    def get_iclr_prompts(self):
        raise RuntimeError


class Flowers102NaivePromptFactory(base_naive_prompt_factory.BaseNaivePromptFactory):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_classnames(self):
        classnames_path = Path(self.prompt_root, "flowers_class_names.json")
        return base_naive_prompt_factory.generic_get_classnames(
            classnames_path, replace_chars=[("_", " ")]
        )

    def get_iclr_prompts(self):
        raise RuntimeError


class StanfordCarsNaivePromptFactory(base_naive_prompt_factory.BaseNaivePromptFactory):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_classnames(self):
        classnames_path = Path(self.prompt_root, "stanford_cars_class_names.json")
        return base_naive_prompt_factory.generic_get_classnames(
            classnames_path, replace_chars=[("_", " ")]
        )

    def get_iclr_prompts(self):
        raise RuntimeError


class StanfordDogsNaivePromptFactory(base_naive_prompt_factory.BaseNaivePromptFactory):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_classnames(self):
        classnames_path = Path(self.prompt_root, "stanford_dogs_class_names.json")
        return base_naive_prompt_factory.generic_get_classnames(
            classnames_path, replace_chars=[("_", " ")]
        )

    def get_iclr_prompts(self):
        raise RuntimeError
