import torch


class EmbeddingWithCache(torch.nn.Module):
    def __init__(self, tokenizer, cache=False) -> None:
        super().__init__()
        self.cache = cache
        self.embedding_cache = list()
        self.cache_emb_mapping = dict()
        self.tokenizer = tokenizer

    def tokenize_text_list(self, text_list):
        return self.tokenizer(text_list)

    def embed_text_list(self, text_list, model):
        tokens = self.tokenize_text_list(text_list)
        tokens = tokens.to(next(iter(model.parameters())).device)
        embs = model.encode_text(tokens)
        embs /= embs.norm(dim=-1, keepdim=True)
        return embs

    def calc_prompt_embeddings_chunks(self, prompts, model, contents=None):
        if contents is None:
            contents = prompts["content"]

        all_texts = list()
        n_text_per_prompt = list()
        for c in contents:
            n_text_per_prompt.append(len(prompts["text"][c]))
            all_texts.extend(prompts["text"][c])
        all_text_embs = self.embed_text_list(text_list=all_texts, model=model)
        emb_set_per_prompt = torch.split(all_text_embs, n_text_per_prompt)
        prompt_embs = [e.mean(dim=0) for e in emb_set_per_prompt]
        embs = torch.stack(prompt_embs)
        embs /= embs.norm(dim=-1, keepdim=True)
        return embs

    def calc_prompt_embeddings_(self, prompts, model, contents=None):
        if contents is None:
            contents = prompts["content"]

        cumsum = 0
        curr_contents = list()
        emb_list = list()
        for c in contents:
            curr_contents.append(c)
            cumsum += len(prompts["text"][c])
            if cumsum >= 500:
                emb_list.append(
                    self.calc_prompt_embeddings_chunks(
                        prompts=prompts, contents=curr_contents, model=model
                    )
                )
                cumsum = 0
                curr_contents = list()
        if len(curr_contents) != 0:
            emb_list.append(
                self.calc_prompt_embeddings_chunks(
                    prompts=prompts, contents=curr_contents, model=model
                )
            )
        embs = torch.concat(emb_list, dim=0)
        return embs

        #     n_text_per_prompt.append(len(prompts["text"][c]))
        #     all_texts.extend(prompts["text"][c])
        # all_text_embs = self.embed_text_list(text_list=all_texts, model=model)
        # emb_set_per_prompt = torch.split(all_text_embs, n_text_per_prompt)
        # prompt_embs = [e.mean(dim=0) for e in emb_set_per_prompt]
        # embs = torch.stack(prompt_embs)
        # embs /= embs.norm(dim=-1, keepdim=True)
        # return embs

    def embed_prompts(self, prompts, model):
        if not self.cache:
            return self.calc_prompt_embeddings_(prompts=prompts, model=model)

        missing_ = [t for t in prompts["content"] if t not in self.cache_emb_mapping]
        missing_ = list(set(missing_))

        if len(missing_) != 0:
            missing_embs = self.calc_prompt_embeddings_(
                prompts=prompts, contents=missing_, model=model
            )
            self.embedding_cache.append(missing_embs)
            for i, c in enumerate(missing_):
                self.cache_emb_mapping[c] = [len(self.embedding_cache) - 1, i]

        emb_list = list()
        for c in prompts["content"]:
            ce_idx, tensor_idx = self.cache_emb_mapping[c]
            emb_list.append(self.embedding_cache[ce_idx][tensor_idx])

        embs = torch.stack(emb_list)
        return embs

    def embed_images(self, images, model):
        embs = model.encode_image(images)
        embs /= embs.norm(dim=-1, keepdim=True)
        return embs
