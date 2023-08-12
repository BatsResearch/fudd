import itertools
import json
from pathlib import Path

import numpy as np

from . import prompt_factory_utils


def load_json(p):
    with open(p, "r") as f:
        obj = json.load(f)
    return obj


class BaseContrastivePromptFactory(object):
    def __init__(
        self,
        prompt_root,
        source,
        augment=False,
        extend=False,
        first_n_pairs=None,
    ) -> None:
        self.augment = augment
        self.extend = extend
        self.first_n_pairs = first_n_pairs
        self.source = source
        self.prompt_root = prompt_root
        # if source == "contrastive_gpt":
        #     self.prompt_pairs = load_json(
        #         Path(prompt_root, "imagenet/contrastive_prompts_gpt/prompt_pairs.json")
        #     )
        # else:
        #     raise ValueError
        # self.class_names = load_json(
        #     Path(prompt_root, "imagenet/imagenet_classes.json")
        # )
        templates = load_json(Path(self.prompt_root, "clip_templates.json"))
        self.templates = [
            t.rstrip(".") for t in templates if t.endswith("{}") or t.endswith("{}.")
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
        self.all_classes_prompts = None
        self.all_classes_prompts_raw = None
        self.all_classes_prompts_by_attributes = None
        self.all_classes_prompts_by_attributes_raw = None
        self.contrastive_metadata = None
        self.prompt_to_attr_mapping = None

        self.prompt_pairs = self.get_prompt_pairs()
        self.class_names = self.get_classnames()

    def get_classnames(self):
        raise NotImplementedError

    def get_prompt_pairs(self):
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

    def get_cls_pair_prompts(self, classes, return_attrs=False):
        pair_key = f"{min(classes)}_{max(classes)}"
        if pair_key not in self.prompt_pairs:
            prompts = [
                [self.default_template.format(self.class_names[c])] for c in classes
            ]
            if return_attrs:
                return prompts, ["no_attr"]
            else:
                return prompts
        else:
            prompts = [[] for _ in classes]
            attrs = []

            cls_pair_data = self.prompt_pairs[pair_key]
            cls_ids = cls_pair_data["classes"]
            prompt_pairs_list = cls_pair_data["prompt_pairs"]
            if self.first_n_pairs is not None:
                prompt_pairs_list = prompt_pairs_list[: self.first_n_pairs]
            for cls_prompt_pairs in prompt_pairs_list:
                prompts[classes.index(cls_ids[0])].append(
                    cls_prompt_pairs["prompt_pair"][0]
                )
                prompts[classes.index(cls_ids[1])].append(
                    cls_prompt_pairs["prompt_pair"][1]
                )
                attrs.append(cls_prompt_pairs["attr_type"])
            if return_attrs:
                return prompts, attrs
            else:
                return prompts

    def get_all_classes_prompts(self, raw=False):
        if self.all_classes_prompts is not None:
            if raw:
                return self.all_classes_prompts_raw
            else:
                return self.all_classes_prompts
        prompts = [[] for _ in self.class_names]
        for cls_pair_data in self.prompt_pairs.values():
            cls_ids = cls_pair_data["classes"]
            cls_pair_prompts = self.get_cls_pair_prompts(cls_ids)
            for pi_ in range(2):
                prompts[cls_ids[pi_]].extend(cls_pair_prompts[pi_])
        for i in range(len(prompts)):
            if len(prompts[i]) == 0:
                prompts[i].append(self.default_template.format(self.class_names[i]))
        self.all_classes_prompts_raw = [list(set(p)) for p in prompts]
        prompts = [self.extend_structure_augment_texts(list(set(p))) for p in prompts]
        self.all_classes_prompts = prompts
        if raw:
            return self.all_classes_prompts_raw
        else:
            return self.all_classes_prompts

    def get_all_contrastive_metadata(self):
        if self.contrastive_metadata is None:
            self.get_all_classes_prompts_by_attributes()
        return self.contrastive_metadata

    def get_cls_prompt_metadata(self, cls_id, prompt):
        contrastive_metadata = self.get_all_contrastive_metadata()
        metadata = contrastive_metadata[cls_id][prompt.lower()]
        return metadata

    def get_all_classes_prompts_by_attributes(self, raw=False):
        if self.all_classes_prompts_by_attributes is not None:
            if raw:
                return self.all_classes_prompts_by_attributes_raw
            else:
                return self.all_classes_prompts_by_attributes

        prompts = [{} for _ in self.class_names]
        contrastive_metadata = [{} for _ in self.class_names]
        for cls_pair_data in self.prompt_pairs.values():
            cls_ids = cls_pair_data["classes"]
            cls_pair_prompts, cls_pair_attrs = self.get_cls_pair_prompts(
                cls_ids, return_attrs=True
            )
            for pi_ in range(2):
                for cls_prompt, cls_attr_type in zip(
                    cls_pair_prompts[pi_], cls_pair_attrs
                ):
                    if cls_attr_type in prompts[cls_ids[pi_]]:
                        prompts[cls_ids[pi_]][cls_attr_type].append(cls_prompt)
                    else:
                        prompts[cls_ids[pi_]][cls_attr_type] = [cls_prompt]

                    if cls_prompt not in contrastive_metadata[cls_ids[pi_]]:
                        contrastive_metadata[cls_ids[pi_]][cls_prompt] = {}
                    contrastive_metadata[cls_ids[pi_]][cls_prompt][
                        cls_ids[1 - pi_]
                    ] = cls_attr_type

        for icm in range(len(contrastive_metadata)):
            contrastive_metadata[icm][
                self.default_template.format(self.class_names[icm])
            ] = {icm: "no attr"}
        self.contrastive_metadata = contrastive_metadata

        prompts = [
            {
                attr_value: list(set(attr_prompts))
                for attr_value, attr_prompts in cls_prompts.items()
            }
            for cls_prompts in prompts
        ]

        self.all_classes_prompts_by_attributes_raw = prompts
        prompts = [
            {
                attr_value: self.extend_structure_augment_texts(attr_prompts)
                for attr_value, attr_prompts in cls_prompts.items()
            }
            for cls_prompts in prompts
        ]
        self.all_classes_prompts_by_attributes = prompts

        if raw:
            return self.all_classes_prompts_by_attributes_raw
        else:
            return self.all_classes_prompts_by_attributes

    def prompt_for_logging(self, class_ids):
        prompts = [[] for _ in class_ids]

        for cls_pair_idx in itertools.combinations(range(len(class_ids)), 2):
            cls_pair = [class_ids[i] for i in cls_pair_idx]
            cls_pair_prompts, cls_pair_attrs = self.get_cls_pair_prompts(
                cls_pair, return_attrs=True
            )
            prompts[cls_pair_idx[0]].append(
                {
                    "pairwise_prompts": cls_pair_prompts[0],
                    "attrs": cls_pair_attrs,
                    "other": cls_pair[1],
                }
            )
            prompts[cls_pair_idx[1]].append(
                {
                    "pairwise_prompts": cls_pair_prompts[1],
                    "attrs": cls_pair_attrs,
                    "other": cls_pair[0],
                }
            )
        return prompts

    def get_prompts(
        self,
        class_ids=None,
        all_classes=False,
        raw=False,
        return_attrs=False,
        non_contrastive=False,
        controls=[],
        log_prep=False,
        non_contrastive_true_attrs=False,
    ):
        assert not (non_contrastive and (all_classes or len(class_ids) == 1))
        assert not (non_contrastive and log_prep)

        if log_prep:
            return self.prompt_for_logging(class_ids=class_ids)

        if all_classes:
            return self.get_all_classes_prompts()
        if len(class_ids) == 1:
            return [self.get_all_classes_prompts()[class_ids[0]]]

        if non_contrastive:
            prompts, used_attrs = self.get_prompts_non_contrastive(
                class_ids=class_ids,
                controls=controls,
                non_contrastive_true_attrs=non_contrastive_true_attrs,
            )
        elif len(class_ids) == 2:
            prompts, used_attrs = self.get_cls_pair_prompts(
                class_ids, return_attrs=True
            )
            used_attrs = [used_attrs, used_attrs]
        else:
            prompts = [list() for _ in class_ids]
            used_attrs = [[] for _ in class_ids]

            for cls_pair_idx in itertools.combinations(range(len(class_ids)), 2):
                cls_pair = [class_ids[i] for i in cls_pair_idx]
                cls_pair_prompts, cls_pair_attrs = self.get_cls_pair_prompts(
                    cls_pair, return_attrs=True
                )
                for pi_ in range(2):
                    used_attrs[cls_pair_idx[pi_]].extend(cls_pair_attrs)
                    prompts[cls_pair_idx[pi_]].extend(cls_pair_prompts[pi_])
                # used_attrs[cls_pair_idx[1]].extend(cls_pair_attrs)
                # prompts[cls_pair_idx[0]].extend(cls_pair_prompts[0])
                # prompts[cls_pair_idx[1]].extend(cls_pair_prompts[1])
            prompts = [list(set(p)) for p in prompts]

        prompts = [list(set(p)) for p in prompts]
        used_attrs = [list(set(ua)) for ua in used_attrs]
        if not raw:
            prompts = [self.extend_structure_augment_texts(p) for p in prompts]
        if return_attrs:
            return prompts, used_attrs
        else:
            return prompts

    def get_prompts_non_contrastive(
        self, class_ids=None, controls=[], non_contrastive_true_attrs=False
    ):
        if len(controls) == 0:
            controls = ["prompts"]
        assert all([c in ["prompts", "attr_types", "attr_words"] for c in controls])

        # all_cls_prompts = self.get_all_classes_prompts(raw=True)
        all_cls_prompts_by_attrs = self.get_all_classes_prompts_by_attributes(raw=True)

        if non_contrastive_true_attrs and self.prompt_to_attr_mapping is None:
            prompt_to_attr_mapping = [{} for _ in all_cls_prompts_by_attrs]
            for ci_ in range(len(all_cls_prompts_by_attrs)):
                prompt_to_attr_mapping[ci_][
                    self.default_template.format(self.class_names[ci_])
                ] = ["template"]
                for attr__, plist__ in all_cls_prompts_by_attrs[ci_].items():
                    for pitem__ in plist__:
                        if pitem__ in prompt_to_attr_mapping[ci_]:
                            prompt_to_attr_mapping[ci_][pitem__].append(attr__)
                        else:
                            prompt_to_attr_mapping[ci_][pitem__] = [attr__]
            self.prompt_to_attr_mapping = prompt_to_attr_mapping

        contrastive_prompts, used_attrs = self.get_prompts(
            class_ids=class_ids, non_contrastive=False, raw=True, return_attrs=True
        )
        used_attr_words = [
            list(itertools.chain(*[a.split(" ") for a in cls_used_attrs]))
            for cls_used_attrs in used_attrs
        ]
        used_attr_words = [list(set(uaw)) for uaw in used_attr_words]
        available_prompts = [[] for _ in class_ids]

        for class_id_idx, class_id in enumerate(class_ids):
            for class_attr, class_attr_prompts in all_cls_prompts_by_attrs[
                class_id
            ].items():
                if "attr_types" in controls and class_attr in used_attrs[class_id_idx]:
                    continue
                if "attr_words" in controls and any(
                    [w in class_attr for w in used_attr_words[class_id_idx]]
                ):
                    continue
                available_prompts[class_id_idx].extend(class_attr_prompts)

        if "prompts" in controls:
            available_prompts = [
                list(set(avail).difference(set(used)))
                for avail, used in zip(available_prompts, contrastive_prompts)
            ]
        available_prompts = [list(set(p)) for p in available_prompts]

        selected_prompts = list()
        for cls_id, avail, contrastive in zip(
            class_ids, available_prompts, contrastive_prompts
        ):
            if len(avail) == 0:
                selected_prompts.append(
                    [self.default_template.format(self.class_names[cls_id])]
                )
            else:
                num_prompts = min(len(avail), len(contrastive))
                selected_prompts.append(
                    np.random.choice(avail, [num_prompts], replace=False).tolist()
                )
        if non_contrastive_true_attrs:
            used_attrs = [[] for _ in class_ids]
            for cls_id_idx in range(len(class_ids)):
                for sp in selected_prompts[cls_id_idx]:
                    used_attrs[cls_id_idx].extend(
                        self.prompt_to_attr_mapping[class_ids[cls_id_idx]][sp]
                    )
        return selected_prompts, used_attrs
