from typing import Union

import torch

from . import fup_task, main_task


class FupModel(torch.nn.Module):
    def __init__(
        self,
        model,
        main_task: main_task.MainFupTask,
        fup_task: Union[None, fup_task.FupFupTask] = None,
        do_fup: bool = True,
        single_return: bool = False,
    ) -> None:
        super().__init__()
        self.single_return = single_return
        self.model = model
        self.main_task = main_task
        self.fup_task = fup_task
        self.do_fup = do_fup

    def forward(self, images):
        ret = dict()

        image_features = self.model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        main_logits, main_preds = self.main_task.get_logits_preds(
            images=image_features, model=self.model, features=True
        )

        ret["main"] = {
            "logits": main_logits,
            "preds": main_preds,
            "prompts": self.main_task.prompts,
        }

        if self.fup_task is not None and self.do_fup:
            fup_logits, fup_preds, prompt_logs_ = self.fup_task.batch_logit_pred(
                images=image_features,
                curr_logits=main_logits,
                curr_preds=main_preds,
                model=self.model,
                features=True,
                prompt_log=True,
            )
            ret["fup"] = {
                "logits": fup_logits,
                "preds": fup_preds,
                "prompts": prompt_logs_,
                "all_prompts": self.fup_task.prompt_factory.get_prompts(
                    all_classes=True
                ),
            }

        return ret
