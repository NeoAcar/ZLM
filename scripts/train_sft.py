"""Supervised fine-tuning of the pretrained base checkpoint on chat data (the SFT v2 recipe).

Every message is encoded as turn_start + role + text + turn_end after one bos token. Only assistant
text and its turn_end enter the loss. turn_end starts from the pretrained eos embedding plus noise.
A seed-137 random 1,499 dialogues of the dataset are held out for validation.
"""

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import pyarrow.parquet as pq
from tokenizers import Tokenizer

if __package__:
    from .model import ModelConfig, ZLM
else:
    from model import ModelConfig, ZLM

BOS, EOS, PAD = 0, 1, 2
ROLE = {'system': 3, 'user': 4, 'assistant': 5}
TURN_END, TURN_START = 8, 13  # turn_start reuses reserved_0
VALIDATION_DIALOGUES = 1499


@dataclass(frozen=True)
class SFTConfig:
    peak_lr: float = 5e-5
    final_lr: float = 1e-6
    warmup_steps: int = 138
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    batch_dialogues: int = 64
    microbatch_tokens: int = 12_288
    epochs: int = 2
    seed: int = 137
    validation_interval: int = 250
    checkpoints: int = 4


@dataclass(frozen=True)
class Dialogue:
    inputs: np.ndarray
    targets: np.ndarray
    source: str
    loss_tokens: int


def encode_turn(role, body):
    return [TURN_START, ROLE[role], *body, TURN_END]


def load_dialogues(rows, tokenizer):
    encoded = iter(tokenizer.encode_batch([m['content'] for row in rows for m in row['messages']],
                                          add_special_tokens=False))
    dialogues = []
    for row in rows:
        ids, labels = [BOS], [-100]
        for message in row['messages']:
            body = next(encoded).ids
            ids += encode_turn(message['role'], body)
            labels += [-100, -100] + ([*body, TURN_END] if message['role'] == 'assistant' else [-100] * (len(body) + 1))
        count = sum(label != -100 for label in labels)
        dialogues.append(Dialogue(np.array(ids[:-1], dtype=np.int64), np.array(labels[1:], dtype=np.int64),
                                  row['source'], count))
    return dialogues


def microbatches(dialogues, limit):
    """Sort by length and bound every padded forward pass by its token area."""
    group = []
    for dialogue in sorted(dialogues, key=lambda item: len(item.inputs)):
        if group and (len(group) + 1) * len(dialogue.inputs) > limit:
            yield group
            group = []
        group.append(dialogue)
    if group:
        yield group


def collate(group, device):
    width = max(len(item.inputs) for item in group)
    inputs = np.full((len(group), width), PAD, dtype=np.int64)
    targets = np.full((len(group), width), -100, dtype=np.int64)
    for row, item in enumerate(group):
        inputs[row, :len(item.inputs)] = item.inputs
        targets[row, :len(item.targets)] = item.targets
    return tuple(torch.from_numpy(x).pin_memory().to(device, non_blocking=True) for x in (inputs, targets))


def labelled_logits(model, inputs, targets):
    """Project only assistant positions; right padding and causal attention need no mask."""
    mask = targets != -100
    hidden = model.hidden_states(inputs)[mask]
    return F.linear(hidden, model.embed_tokens.weight), targets[mask]


def training_step(model, batch, config, device):
    count = sum(item.loss_tokens for item in batch)
    total = torch.zeros((), device=device)
    padded = forwards = 0
    for group in microbatches(batch, config.microbatch_tokens):
        inputs, targets = collate(group, device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, labels = labelled_logits(model, inputs, targets)
            loss = F.cross_entropy(logits.float(), labels, reduction='sum', label_smoothing=0.0)
        # Equal weight per assistant token in the whole optimizer batch.
        (loss / count).backward()
        total += loss.detach()
        padded += inputs.numel()
        forwards += 1
    return total.item() / count, count, padded, forwards


def learning_rate(step, total, config):
    if step < config.warmup_steps:
        return config.peak_lr * (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / (total - 1 - config.warmup_steps)
    return config.final_lr + (config.peak_lr - config.final_lr) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, dialogues, config, device):
    model.eval()
    totals, counts = {}, {}
    stops = recalled = other = early = 0
    for group in microbatches(dialogues, config.microbatch_tokens):
        inputs, targets = collate(group, device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, labels = labelled_logits(model, inputs, targets)
            losses = F.cross_entropy(logits.float(), labels, reduction='none')
        predicted, is_stop = logits.argmax(-1) == TURN_END, labels == TURN_END
        stops += is_stop.sum().item()
        recalled += (predicted & is_stop).sum().item()
        other += (~is_stop).sum().item()
        early += (predicted & ~is_stop).sum().item()
        for item, part in zip(group, losses.double().split([item.loss_tokens for item in group])):
            totals[item.source] = totals.get(item.source, 0.0) + part.sum().item()
            counts[item.source] = counts.get(item.source, 0) + item.loss_tokens
    weight = model.embed_tokens.weight
    cosine = lambda a, b: F.cosine_similarity(weight[a].float(), weight[b].float(), dim=0).item()
    norm = lambda index: weight[index].float().norm().item()
    metrics = {'validation/loss': sum(totals.values()) / sum(counts.values()),
               'validation/eos_recall': recalled / stops,
               'validation/early_eos_rate': early / other,
               'roles/cosine_user_assistant': cosine(4, 5),
               'roles/cosine_system_user': cosine(3, 4),
               'roles/cosine_system_assistant': cosine(3, 5),
               'roles/median_text_norm': weight[128:].float().norm(dim=1).median().item(),
               'roles/cosine_turn_start_end': cosine(TURN_START, TURN_END),
               'roles/cosine_turn_end_eos': cosine(TURN_END, EOS),
               'roles/norm_turn_start': norm(TURN_START), 'roles/norm_turn_end': norm(TURN_END),
               'roles/norm_eos': norm(EOS)}
    metrics.update({f'roles/norm_{name}': norm(index) for name, index in ROLE.items()})
    metrics.update({f'validation/loss_{name}': totals[name] / counts[name] for name in sorted(totals)})
    model.train()
    return metrics


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    return value


def train(base, data, tokenizer_path, output, resume=None):
    import wandb

    config, device = SFTConfig(), torch.device('cuda')
    if torch.cuda.get_device_properties(0).total_memory < 60_000_000_000:
        config = replace(config, microbatch_tokens=8_192)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.encode_special_tokens = True
    rows = pq.read_table(data).to_pylist()
    held = set(np.random.Generator(np.random.PCG64(config.seed)).permutation(len(rows))[:VALIDATION_DIALOGUES].tolist())
    training = load_dialogues([row for i, row in enumerate(rows) if i not in held], tokenizer)
    validation = load_dialogues([row for i, row in enumerate(rows) if i in held], tokenizer)
    print(json.dumps({'event': 'data_ready', 'train': len(training), 'validation': len(validation)}), flush=True)

    torch.manual_seed(config.seed)
    state = torch.load(base, map_location='cpu', weights_only=True, mmap=True)
    model_config = ModelConfig(**state['model_config'])
    model = ZLM(model_config, device=device)
    model.load_state_dict(state['model'])
    del state
    if resume is None:
        # turn_end starts from the pretrained eos row plus noise with 10% of its norm (embeddings are tied).
        with torch.no_grad():
            row = model.embed_tokens.weight[EOS].float()
            noise = torch.randn(row.shape, generator=torch.Generator().manual_seed(config.seed)).to(row.device)
            model.embed_tokens.weight[TURN_END] = (row + 0.1 * row.norm() * noise / noise.norm()).to(
                model.embed_tokens.weight.dtype)
    model.rope.set_phase(True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.peak_lr, betas=config.betas, eps=config.eps,
                                  weight_decay=config.weight_decay, foreach=False, fused=True)
    settings = json.loads(json.dumps(asdict(config)))
    step, run_id = 0, uuid.uuid4().hex[:8]
    if resume is not None:
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        step, run_id = saved['step'], saved['wandb_run_id']
        del saved
    per_epoch = math.ceil(len(training) / config.batch_dialogues)
    total = per_epoch * config.epochs
    generator = np.random.Generator(np.random.PCG64(config.seed))
    orders = [generator.permutation(len(training)) for _ in range(config.epochs)]
    checkpoint_steps = {round(total * part / config.checkpoints) for part in range(1, config.checkpoints + 1)}

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {'sft': settings, 'model': asdict(model_config), 'steps': total}
    run = wandb.init(project='ZLM', entity='eminacar-itu', id=run_id, name='trinity-0.5b-sft-v2',
                     resume='must' if resume is not None else 'never', config=summary,
                     settings=wandb.Settings(disable_code=True))
    stream = (output / 'metrics.jsonl').open('a')

    def record(event, values):
        line = json.dumps({'event': event, 'step': step, **values}, ensure_ascii=False)
        stream.write(line + '\n')
        stream.flush()
        print(line, flush=True)

    def inspect():
        values = evaluate(model, validation, config, device)
        run.log(values, step=step)
        record('validation', values)

    record('started', {'total_steps': total, 'wandb_url': run.url, 'run_id': run_id})
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        if step == 0:
            inspect()
        while step < total:
            started = time.monotonic()
            epoch, index = divmod(step, per_epoch)
            chosen = orders[epoch][index * config.batch_dialogues:(index + 1) * config.batch_dialogues]
            batch = [training[i] for i in chosen]
            lr = learning_rate(step, total, config)
            loss, count, padded, forwards = training_step(model, batch, config, device)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm,
                                                 error_if_nonfinite=True, foreach=True).item()
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step += 1
            seconds = time.monotonic() - started
            values = {'train/loss': loss, 'train/grad_norm': norm,
                      'train/clipped': int(norm > config.max_grad_norm), 'train/lr': lr,
                      'train/epoch': step / per_epoch, 'train/loss_tokens': count,
                      'train/real_tokens': sum(len(item.inputs) for item in batch),
                      'train/padded_tokens': padded, 'train/forwards': forwards, 'train/seconds': seconds,
                      'train/forward_tokens_per_second': padded / seconds,
                      'gpu/allocated_bytes': torch.cuda.memory_allocated(),
                      'gpu/peak_allocated_bytes': torch.cuda.max_memory_allocated()}
            run.log(values, step=step)
            record('train', values)
            if step in checkpoint_steps:
                path = output / f'sft-{step:08d}.pt'
                torch.save({'kind': 'sft', 'step': step, 'total_steps': total, 'sft_config': settings,
                            'wandb_run_id': run_id, 'model_config': asdict(model_config),
                            'active_context_length': model.rope.active_context,
                            'model': cpu_tree(model.state_dict()), 'optimizer': cpu_tree(optimizer.state_dict()),
                            'torch_rng': torch.get_rng_state()}, path)
                record('checkpoint', {'file': path.name})
            if step % config.validation_interval == 0 or step == total:
                inspect()
    stream.close()
    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True, help='Pretrained base checkpoint.')
    parser.add_argument('--data', type=Path, required=True, help='data/train.parquet of 137N/trinity-turkish-sft.')
    parser.add_argument('--tokenizer-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    train(**vars(parser.parse_args()))
