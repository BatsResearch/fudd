import itertools

import torch
from torch.nn import functional as F

from . import embedding_helper


class FupFupTask(torch.nn.Module):
    def __init__(
        self,
        topk,
        prompt_factory,
        tokenizer,
        aggregation="sim_of_mean",
        method=None,
        ovo_metric=None,
        ovo_winner_only=None,
        cache=None,
        non_contrastive=False,
        non_contrastive_controls=[],
    ) -> None:
        super().__init__()
        self.topk = topk
        self.aggregation = aggregation
        self.prompt_factory = prompt_factory
        self.method = method
        self.ovo_metric = ovo_metric
        self.ovo_winner_only = ovo_winner_only
        self.non_contrastive = non_contrastive
        self.non_contrastive_controls = non_contrastive_controls

        assert cache in [None, "none", False, "raw"]
        do_cache = False
        if cache == "raw":
            do_cache = True
        self.embedder = embedding_helper.EmbeddingWithCache(
            tokenizer=tokenizer, cache=do_cache
        )

    def get_prompt_embeddings(self, prompts, model):
        emb_list = list()
        for cls_prompts in prompts:
            cls_embs = self.embedder.embed_prompts(prompts=cls_prompts, model=model)
            if self.aggregation == "sim_of_mean":
                cls_embs = cls_embs.mean(dim=0)
                cls_embs /= cls_embs.norm()
            emb_list.append(cls_embs)
        embs = emb_list
        if self.aggregation == "sim_of_mean":
            embs = torch.stack(embs)
        return embs

    def image_prompt_ensemble_logit_pred(
        self,
        image_features,
        img_logits,
        img_preds,
        model,
        keep_preds_order=True,
        prompt_log=False,
    ):
        prompts = self.prompt_factory.get_prompts(
            class_ids=img_preds.tolist(),
            non_contrastive=self.non_contrastive,
            controls=self.non_contrastive_controls,
        )
        log_contrastive_prompts = self.prompt_factory.get_prompts(
            class_ids=img_preds.tolist(), non_contrastive=False
        )
        p_log = {"log_contrastive": log_contrastive_prompts, "used_prompts": prompts}

        embeddings = self.get_prompt_embeddings(prompts=prompts, model=model)

        image_features = torch.unsqueeze(image_features, dim=0)

        if self.aggregation == "sim_of_mean":
            logits = image_features @ embeddings.t()
        elif self.aggregation == "mean_of_sims":
            logits_list = [(image_features @ emb.t()).mean(dim=1) for emb in embeddings]
            logits = torch.stack(logits_list, dim=1)
        elif self.aggregation == "mean_of_probs":
            image_prompt_sims = [
                torch.squeeze(image_features @ emb.t(), dim=0) for emb in embeddings
            ]
            image_prompt_sims = torch.stack(image_prompt_sims)
            image_prompt_probs = F.softmax(image_prompt_sims, dim=0)
            logits = image_prompt_probs.mean(dim=1)
        else:
            raise ValueError
        logits = torch.squeeze(logits, dim=0)
        if prompt_log:
            return logits, img_preds, p_log
        else:
            return logits, img_preds

    def image_ovo_logit_pred(
        self,
        image_features,
        img_logits,
        img_preds,
        model,
    ):
        ovo_counts = [0 for _ in range(img_preds.shape[0])]
        ovo_logits = [0 for _ in range(img_preds.shape[0])]
        ovo_probs = [0 for _ in range(img_preds.shape[0])]

        for cls_pair_idx_ in itertools.combinations(range(img_preds.shape[0]), 2):
            cls_pair_idx = list(cls_pair_idx_)
            pair_logits, _ = self.image_prompt_ensemble_logit_pred(
                image_features=image_features,
                img_logits=img_logits[cls_pair_idx],
                img_preds=img_preds[cls_pair_idx],
                model=model,
                keep_preds_order=True,
            )
            pair_probs = F.softmax(pair_logits, dim=0)

            winner_idx, loser_idx = torch.argsort(pair_logits, descending=True)
            ovo_counts[cls_pair_idx[winner_idx]] += 1
            ovo_logits[cls_pair_idx[winner_idx]] += pair_logits[winner_idx]
            ovo_probs[cls_pair_idx[winner_idx]] += pair_probs[winner_idx]
            if not self.ovo_winner_only:
                ovo_logits[cls_pair_idx[loser_idx]] += pair_logits[loser_idx]
                ovo_probs[cls_pair_idx[loser_idx]] += pair_probs[loser_idx]

        if self.ovo_metric == "count":
            new_logits = torch.tensor(ovo_counts)
        elif self.ovo_metric == "logit":
            new_logits = torch.tensor(ovo_logits)
        elif self.ovo_metric == "prob":
            new_logits = torch.tensor(ovo_probs)
        else:
            raise ValueError
        return new_logits, img_preds

    def image_logit_pred(
        self,
        image_features,
        img_logits,
        img_preds,
        model,
        prompt_log=False,
    ):
        top_indices = torch.argsort(img_logits, descending=True, dim=-1)[: self.topk]
        top_logits = img_logits[top_indices]
        top_preds = img_preds[top_indices]
        input_kwargs = dict(
            model=model,
            image_features=image_features,
            img_logits=top_logits,
            img_preds=top_preds,
            prompt_log=prompt_log,
        )
        if self.method == "ovo":
            raise NotImplementedError
            new_logits, new_preds = self.image_ovo_logit_pred(**input_kwargs)
        else:
            output_ = self.image_prompt_ensemble_logit_pred(**input_kwargs)
            if prompt_log:
                new_logits, new_preds, p_log = output_
            else:
                new_logits, new_preds = output_
                p_log = None

        if prompt_log:
            return new_logits, new_preds, p_log
        else:
            return new_logits, new_preds

    def batch_logit_pred(
        self,
        images,
        curr_logits,
        curr_preds,
        model,
        features=True,
        prompt_log=False,
    ):
        if not features:
            images = self.embedder.embed_images(images, model)

        logit_list = list()
        pred_list = list()
        prompt_log_list = list()
        for i_feat, i_logit, i_pred in zip(images, curr_logits, curr_preds):
            i_output_ = self.image_logit_pred(
                image_features=i_feat,
                img_logits=i_logit,
                img_preds=i_pred,
                model=model,
                prompt_log=prompt_log,
            )
            if prompt_log:
                img_logits, img_preds, img_prompt_logs = i_output_
                prompt_log_list.append(img_prompt_logs)
            else:
                img_logits, img_preds = i_output_

            logit_list.append(img_logits)
            pred_list.append(img_preds)
        fup_logits = torch.stack(logit_list)
        fup_preds = torch.stack(pred_list)

        if prompt_log:
            return fup_logits, fup_preds, prompt_log_list
        else:
            return fup_logits, fup_preds
