import torch


class CustomAccMetric:
    def __init__(self, topk_list):
        self.topk_list = topk_list
        self.sums = {i: 0 for i in self.topk_list}
        self.total_num = 0

    def add_batch(self, targets, logits=None, preds=None, order_pred_by_logit=False):
        if preds is None:
            preds = torch.argsort(logits, descending=True, dim=-1)
        elif order_pred_by_logit:
            sorted_indices = torch.argsort(logits, descending=True, dim=-1)
            preds = preds[torch.arange(preds.shape[0])[:, None], sorted_indices]
        for tk in self.topk_list:
            self.sums[tk] += (targets.view([-1, 1]) == preds[:, :tk]).sum()
        self.total_num += targets.shape[0]

    def compute(self, do_print=False, percent=False, return_accs=True):
        coef = 100 if percent else 1
        tk_accs = {k: coef * (v / self.total_num) for k, v in self.sums.items()}
        for k in tk_accs:
            if isinstance(tk_accs[k], torch.Tensor):
                tk_accs[k] = tk_accs[k].item()
        if do_print:
            tks = sorted(list(tk_accs.keys()))
            log_str = []
            for t in tks:
                log_str.append(f"Top-{t}: {tk_accs[t]:.3f}")
            print(" ** ".join(log_str))
        if return_accs:
            return tk_accs
