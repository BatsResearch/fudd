import clip
import torch

# class DummyCLIIPModel(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.layer = torch.nn.Linear(in_features=10, out_features=10)
#
#     def encode_text(self, tokens):
#         emb = torch.rand(tokens.shape[0], 512).cuda()
#         return emb
#
#     def encode_image(self, images):
#         emb = torch.rand(images.shape[0], 512).cuda()
#         return emb


class FuDD(torch.nn.Module):
    def __init__(
        self,
        top_k,
        prompt_factory,
        backbone,
        follow_up,
        step_one_source,
    ) -> None:
        super().__init__()

        self.top_k = top_k
        self.follow_up = follow_up
        if follow_up:
            self.step_one_source = "single_template"
        else:
            self.step_one_source = step_one_source
        self.prompt_factory = prompt_factory
        self.clip_model, _ = clip.load(backbone, device="cpu")
        # self.clip_model = DummyCLIIPModel()

        self.step_one_embeddings = None

        self.embedding_cache = list()
        self.cache_emb_mapping = dict()

    def ensure_emb_cache_exists(self, texts):
        missing_texts = list()
        for t in texts:
            if t not in self.cache_emb_mapping:
                missing_texts.append(t)
        if len(missing_texts) == 0:
            return

        tokens = clip.tokenize(missing_texts)
        tokens = tokens.to(next(iter(self.clip_model.parameters())).device)
        embs = self.clip_model.encode_text(tokens)
        embs /= embs.norm(dim=-1, keepdim=True)

        self.embedding_cache.append(embs)
        for i, c in enumerate(missing_texts):
            self.cache_emb_mapping[c] = [len(self.embedding_cache) - 1, i]

    def get_text_embeddings(self, texts):
        emb_list = list()
        for text_list in texts:
            self.ensure_emb_cache_exists(text_list)

            text_emb_list = list()
            for c in text_list:
                ce_idx, tensor_idx = self.cache_emb_mapping[c]
                text_emb_list.append(self.embedding_cache[ce_idx][tensor_idx])
            embs = torch.stack(text_emb_list)
            emb = embs.mean(dim=0)
            emb /= emb.norm()
            emb_list.append(emb)
        embs = torch.stack(emb_list)
        return embs

    def get_step_one_embeddings(self):
        if self.step_one_embeddings is not None:
            return self.step_one_embeddings

        descs = self.prompt_factory.get_prompts(
            class_ids=None, prompt_type=self.step_one_source
        )
        step_one_embeddings = self.get_text_embeddings(descs)
        self.step_one_embeddings = step_one_embeddings
        return self.step_one_embeddings

    def forward(self, images):
        image_features = self.clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        step_one_cls_embeddings = self.get_step_one_embeddings()
        step_one_logits = image_features @ step_one_cls_embeddings.t()

        step_one_preds_unsorted = torch.arange(step_one_logits.shape[1]).expand(
            step_one_logits.shape
        )
        step_one_preds_unsorted = step_one_preds_unsorted.to(step_one_logits.device)

        image_logit_list = list()
        image_pred_list = list()

        if not self.follow_up:
            return step_one_logits, step_one_preds_unsorted

        for i_feat, i_logit in zip(image_features, step_one_logits):
            img_logits, img_preds = self.image_logit_pred(
                image_features=i_feat,
                img_logits=i_logit,
                img_preds=step_one_preds_unsorted[0],
            )

            image_logit_list.append(img_logits)
            image_pred_list.append(img_preds)

        fudd_logits = torch.stack(image_logit_list)
        fudd_preds_unsorted = torch.stack(image_pred_list)

        return fudd_logits, fudd_preds_unsorted

    def image_logit_pred(
        self,
        image_features,
        img_logits,
        img_preds,
    ):
        top_indices = torch.argsort(img_logits, descending=True, dim=-1)[: self.top_k]
        top_preds = img_preds[top_indices]

        descs = self.prompt_factory.get_prompts(
            class_ids=top_preds.tolist(),
            prompt_type="gpt",
        )

        embeddings = self.get_text_embeddings(texts=descs)
        image_features = torch.unsqueeze(image_features, dim=0)
        new_logits = image_features @ embeddings.t()
        new_logits = torch.squeeze(new_logits, dim=0)
        return new_logits, top_preds
