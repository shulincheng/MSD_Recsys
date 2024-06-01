import time
import torch as th
from torch import nn, optim
from tqdm import tqdm


def fix_weight_decay(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(map(lambda x: x in name, ['bias', 'batch_norm', 'activation'])):
            no_decay.append(param)
        else:
            decay.append(param)
    params = [{'params': decay}, {'params': no_decay, 'weight_decay': 0}]
    return params


def prepare_batch(batch, device):
    seqs_train, labels = batch
    seqs_train_gpu = [x.to(device) for x in seqs_train]
    labels_gpu = labels.to(device)

    return seqs_train_gpu, labels_gpu


def evaluate(model, data_loader, device, cutoff=20):
    model.eval()
    mrr = 0
    hit = 0
    num_samples = 0

    with th.no_grad():
        for batch in data_loader:
            seqs_train, labels = prepare_batch(batch, device)
            logits = model(*seqs_train)

            batch_size = logits.size(0)
            num_samples += batch_size
            topk = logits.topk(k=cutoff)[1]
            labels = labels.unsqueeze(-1)
            hit_ranks = th.where(topk == labels)[1] + 1
            hit += hit_ranks.numel()
            mrr += hit_ranks.float().reciprocal().sum().item()

    return mrr / num_samples, hit / num_samples


class TrainRunner:
    def __init__(
            self,
            model,
            train_loader,
            test_loader,
            device,
            lr=1e-3,
            weight_decay=0,
            patience=3,
    ):
        self.model = model
        if weight_decay > 0:
            params = fix_weight_decay(model)
        else:
            params = model.parameters()
        self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epoch = 0
        self.batch = 0
        self.patience = patience

    def train(self, epochs, log_interval=100):
        max_mrr = 0
        max_hit = 0
        bad_counter = 0
        t = time.time()
        mean_loss = 0

        for epoch in tqdm(range(epochs)):
            self.model.train()

            for batch in self.train_loader:
                seqs_train, labels = prepare_batch(batch, self.device)
                self.optimizer.zero_grad()
                scores = self.model(*seqs_train)
                loss = nn.functional.nll_loss(scores, labels)
                loss.backward()
                self.optimizer.step()

                mean_loss += loss.item() / log_interval

                if self.batch > 0 and self.batch % log_interval == 0:
                    print(f'Batch {self.batch}: Loss = {mean_loss:.4f}, Time Elapsed = {time.time() - t:.2f}s')
                    t = time.time()
                    mean_loss = 0

                self.batch += 1
            self.scheduler.step()
            mrr, hit = evaluate(self.model, self.test_loader, self.device)
            log = f'Epoch {self.epoch}: MRR@20 = {mrr * 100:.3f}%, Hit@20 = {hit * 100:.3f}%'
            print(log)

            if mrr < max_mrr and hit < max_hit:
                bad_counter += 1
                if bad_counter == self.patience:
                    break
            else:
                bad_counter = 0
            max_mrr = max(max_mrr, mrr)
            max_hit = max(max_hit, hit)
            self.epoch += 1
        return max_mrr, max_hit
