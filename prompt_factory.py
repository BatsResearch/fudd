import itertools
import json
from pathlib import Path


def load_json(p):
    with open(p, "r") as f:
        obj = json.load(f)
    return obj


class PromptFactory(object):
    def __init__(self, prompt_root, dataset_name) -> None:
        self.prompt_root = Path(prompt_root)
        if dataset_name == "imagenet_v2":
            dataset_name = "imagenet"
        self.dataset_name = dataset_name

        self.all_templates = load_json(Path(self.prompt_root, "clip_templates.json"))
        self.templates = [
            t.rstrip(".")
            for t in self.all_templates
            if t.endswith("{}") or t.endswith("{}.")
        ]

        self.default_template = "a photo of a {}."

        self.prompt_pairs = load_json(
            self.prompt_root.joinpath(f"{dataset_name}_prompt_pairs.json")
        )
        self.class_names = load_json(
            self.prompt_root.joinpath(f"{dataset_name}_class_names.json")
        )
        self.class_names = [c.replace("_", " ") for c in self.class_names]

        self.all_classes_prompts_classname = None
        self.all_classes_prompts_single_template = None
        self.all_classes_prompts_template_set = None
        self.all_classes_prompts_gpt = None

    def get_all_classes_prompts(self, prompt_type):
        if prompt_type == "classname":
            if self.all_classes_prompts_classname is None:
                prompt_list = [[f"{c}."] for c in self.class_names]
                prompt_list = [
                    [p_.capitalize() for p_ in prompts_] for prompts_ in prompt_list
                ]
                self.all_classes_prompts_classname = prompt_list
            return self.all_classes_prompts_classname
        elif prompt_type == "single_template":
            if self.all_classes_prompts_single_template is None:
                prompt_list = [
                    [self.default_template.format(c)] for c in self.class_names
                ]
                prompt_list = [
                    [p_.capitalize() for p_ in prompts_] for prompts_ in prompt_list
                ]
                self.all_classes_prompts_single_template = prompt_list
            return self.all_classes_prompts_single_template
        elif prompt_type == "template_set":
            if self.all_classes_prompts_template_set is None:
                prompt_list = [
                    [t.format(c) for t in self.all_templates] for c in self.class_names
                ]
                prompt_list = [
                    [p_.capitalize() for p_ in prompts_] for prompts_ in prompt_list
                ]
                self.all_classes_prompts_template_set = prompt_list
            return self.all_classes_prompts_template_set
        elif prompt_type == "gpt":
            if self.all_classes_prompts_gpt is None:
                prompts = [[] for _ in self.class_names]
                for cls_pair_data in self.prompt_pairs.values():
                    cls_ids = cls_pair_data["classes"]
                    cls_pair_prompts = self.get_cls_pair_prompts(cls_ids)

                    prompts[cls_ids[0]].extend(cls_pair_prompts[0])
                    prompts[cls_ids[1]].extend(cls_pair_prompts[1])
                for i in range(len(prompts)):
                    if len(prompts[i]) == 0:
                        prompts[i].append(
                            self.default_template.format(self.class_names[i])
                        )
                prompts = [list(set(p)) for p in prompts]
                prompts = [
                    [p_t.capitalize() for p_t in prompts_t] for prompts_t in prompts
                ]
                self.all_classes_prompts_gpt = prompts
            return self.all_classes_prompts_gpt
        else:
            raise ValueError

    def get_cls_pair_prompts(self, classes):
        pair_key = f"{min(classes)}_{max(classes)}"
        if pair_key not in self.prompt_pairs:
            prompts = [
                [self.default_template.format(self.class_names[c])] for c in classes
            ]
            return prompts
        else:
            prompts = [[] for _ in classes]

            cls_pair_data = self.prompt_pairs[pair_key]
            cls_ids = cls_pair_data["classes"]
            prompt_pairs_list = cls_pair_data["prompt_pairs"]
            for cls_prompt_pairs in prompt_pairs_list:
                prompts[classes.index(cls_ids[0])].append(
                    cls_prompt_pairs["prompt_pair"][0]
                )
                prompts[classes.index(cls_ids[1])].append(
                    cls_prompt_pairs["prompt_pair"][1]
                )
            return prompts

    def get_prompts(
        self,
        class_ids,
        prompt_type,
    ):

        if class_ids is None or len(class_ids) == 1:
            all_classes_prompts = self.get_all_classes_prompts(prompt_type=prompt_type)
            if class_ids is None:
                return all_classes_prompts
            if len(class_ids) == 1:
                return all_classes_prompts[0]

        if prompt_type != "gpt":
            all_classes_prompts = self.get_all_classes_prompts(prompt_type=prompt_type)
            return [all_classes_prompts[ci] for ci in class_ids]

        prompts = [[] for _ in class_ids]
        for cls_pair_idx in itertools.combinations(range(len(class_ids)), 2):
            cls_pair = [class_ids[i] for i in cls_pair_idx]
            cls_pair_prompts = self.get_cls_pair_prompts(cls_pair)
            prompts[cls_pair_idx[0]].extend(cls_pair_prompts[0])
            prompts[cls_pair_idx[1]].extend(cls_pair_prompts[1])
        prompts = [list(set(p)) for p in prompts]
        prompts = [[p_t.capitalize() for p_t in prompts_t] for prompts_t in prompts]
        return prompts


# if __name__ == "__main__":
# pf = PromptFactory(prompt_root='/media/master/sdisk/data_root_pd/prompts/public_prompts', dataset_name='pets')
# pf.get_prompts(class_ids=None, prompt_type='gpt')
# from IPython import embed; embed()
