import torch
from tqdm import tqdm

from . import embedding_helper


class MainFupTask(torch.nn.Module):
    def __init__(
        self,
        prompt_factory,
        tokenizer,
        aggregation="sim_of_mean",
        cache=None,
    ) -> None:
        super().__init__()
        assert cache in [None, "result", "none", False]
        self.aggregation = aggregation

        self.cache_embs = False
        if cache == "result":
            self.cache_embs = True

        self.prompts = prompt_factory.get_prompts(all_classes=True)
        self.prompt_embeddings = None
        self.embedder = embedding_helper.EmbeddingWithCache(
            tokenizer=tokenizer, cache=False
        )

    def embed_cls_prompts(self, model, cls_prompts):
        cls_embs = self.embedder.embed_prompts(prompts=cls_prompts, model=model)
        if self.aggregation == "sim_of_mean":
            cls_embs = cls_embs.mean(dim=0)
            cls_embs /= cls_embs.norm()
        return cls_embs

    def get_prompt_embeddings(self, model):
        if self.cache_embs and self.prompt_embeddings is not None:
            return self.prompt_embeddings
        emb_list = list()
        for cls_prompts in tqdm(self.prompts, desc="Main task embeddings"):
            emb_list.append(
                self.embed_cls_prompts(model=model, cls_prompts=cls_prompts)
            )
        embs = emb_list
        if self.aggregation == "sim_of_mean":
            embs = torch.stack(embs)

        if self.cache_embs:
            self.prompt_embeddings = embs
        return embs

    def calc_logits(self, images, cls_embs):
        if self.aggregation == "sim_of_mean":
            logits = images @ cls_embs.t()
            return logits
        elif self.aggregation == "mean_of_sims":
            logits_list = [(images @ emb.t()).mean(dim=1) for emb in cls_embs]
            logits = torch.stack(logits_list, dim=1)
            return logits
        else:
            raise ValueError

    def get_logits(self, images, model, features=False):
        if not features:
            images = self.embedder.embed_images(images=images, model=model)

        cls_embs = self.get_prompt_embeddings(model=model)
        return self.calc_logits(images=images, cls_embs=cls_embs)

    def get_logits_preds(self, *args, **kwargs):
        logits = self.get_logits(*args, **kwargs)
        preds = torch.arange(logits.shape[1]).expand(logits.shape)
        preds = preds.to(logits.device)
        return logits, preds
