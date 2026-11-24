"""Pretrain ZLM from scratch with the recipe that produced the released base checkpoint.

Steps 0-20697 train on the main corpus at the 4,096-token context: linear warmup over 500 steps,
then a constant LR. The global batch is 262,144 tokens and doubles to 524,288 after step 3249.
The peak LRs drop by 2/3 at step 9419 and by a further 1/2 at step 14366. From step 20698 the
academic decay corpus trains at the 16,384-token YaRN context while both LRs fall linearly to zero.
Validation uses the last blocks of the shuffled main corpus, which training never reaches.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import uuid

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

if __package__:
    from .model import ModelConfig, ZLM, build_optimizers
    from .training_data import TokenBlocks
else:
    from model import ModelConfig, ZLM, build_optimizers
    from training_data import TokenBlocks

SEED = 137
WARMUP_STEPS = 500
MUON_LR, ADAMW_LR = 0.001, 0.0006
MAX_GRAD_NORM = 1.0
BATCH_TOKENS = 262_144
DOUBLE_BATCH_AFTER_STEP = 3249
LR_DROPS = ((9419, 2 / 3), (14366, 0.5))
DECAY_START_STEP = 20698
CHECKPOINT_SECONDS = 1800
# Held out from the end of the shuffled main corpus: the size of the original 18.8M-token validation set.
HELD_OUT_BLOCKS = 4598
VALIDATION_BLOCKS = 64  # evaluated per validation pass


def source_step(step):
    """Position in 262,144-token units; each step after the doubling consumes two."""
    return step if step <= DOUBLE_BATCH_AFTER_STEP else 2 * step - DOUBLE_BATCH_AFTER_STEP


def peak_lrs(step):
    muon, adamw = MUON_LR, ADAMW_LR
    for start, scale in LR_DROPS:
        if step >= start:
            muon, adamw = muon * scale, adamw * scale
    return muon, adamw


class PackedOrder:
    """One seed-137 permutation of the blocks; longer contexts concatenate consecutive picks."""

    def __init__(self, blocks):
        self.blocks = blocks
        self.order = np.random.Generator(np.random.PCG64(SEED)).permutation(len(blocks))

    def batch(self, offset, sequences, length):
        base = self.blocks.sequence_length
        group = length // base
        selected = self.order[offset:offset + sequences * group]
        values = np.empty((sequences, length + 1), dtype=np.int64)
        for row in range(sequences):
            for column in range(group):
                inputs, targets = self.blocks[int(selected[row * group + column])]
                values[row, column * base:(column + 1) * base] = inputs
            values[row, -1] = targets[-1]
        # Targets come from the concatenation, including every block boundary.
        return torch.from_numpy(values[:, :-1]), torch.from_numpy(values[:, 1:])


class LanguageModelLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, targets):
        return F.cross_entropy(self.model(inputs).float().flatten(0, 1), targets.flatten())


def to_device(inputs, targets):
    return inputs.pin_memory().cuda(non_blocking=True), targets.pin_memory().cuda(non_blocking=True)


def accumulate_gradients(loss_fn, order, *, offset, sequences, length, microbatch):
    base = order.blocks.sequence_length
    accumulation = sequences // microbatch
    total = torch.zeros((), device='cuda')
    for index in range(accumulation):
        inputs, targets = to_device(*order.batch(offset + index * microbatch * (length // base), microbatch, length))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = loss_fn(inputs, targets)
        (loss / accumulation).backward()
        total += loss.detach() / accumulation
    return total


@torch.no_grad()
def evaluate(loss_fn, model, order, *, length, microbatch):
    group = length // order.blocks.sequence_length
    count = VALIDATION_BLOCKS - VALIDATION_BLOCKS % (microbatch * group)
    model.eval()
    total = torch.zeros((), device='cuda')
    for offset in range(0, count, microbatch * group):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            total += loss_fn(*to_device(*order.batch(offset, microbatch, length)))
    model.train()
    return {'validation/loss': total.item() / (count // (microbatch * group)),
            'validation/tokens': count * order.blocks.sequence_length, 'validation/context_length': length}


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def train(data, decay_data, output, resume=None):
    import wandb

    torch.manual_seed(SEED)
    config = ModelConfig()
    base, extended = config.context_length, config.extended_context_length
    order = PackedOrder(TokenBlocks(data, base))
    validation_order = PackedOrder(order.blocks)
    validation_order.order = order.order[-HELD_OUT_BLOCKS:]
    decay_order = PackedOrder(TokenBlocks(decay_data, extended))
    main_tokens = source_step(DECAY_START_STEP) * BATCH_TOKENS
    decay_tokens = len(decay_order.blocks) * extended
    decay_batch = 2 * BATCH_TOKENS
    end_step = DECAY_START_STEP + -(-decay_tokens // decay_batch)
    consumed = lambda step: min(max(0, step - DECAY_START_STEP) * decay_batch, decay_tokens)
    microbatch_4k = 8 if torch.cuda.get_device_properties(0).total_memory >= 60_000_000_000 else 4

    model = ZLM(config, device='cuda')
    optimizers = build_optimizers(model, muon_lr=MUON_LR, adamw_lr=ADAMW_LR)
    step, run_id = 0, uuid.uuid4().hex[:8]
    if resume is not None:
        state = torch.load(resume, map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'])
        for optimizer, saved in zip(optimizers, state['optimizers']):
            optimizer.load_state_dict(saved)
        step, run_id = state['step'], state['wandb_run_id']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    run = wandb.init(project='ZLM', entity='eminacar-itu', id=run_id, name='zlm-0.5b-pretrain',
                     resume='must' if resume is not None else 'never', config={'model': asdict(config)},
                     settings=wandb.Settings(disable_code=True))

    def save():
        torch.save({'step': step, 'model_config': asdict(config), 'active_context_length': model.rope.active_context,
                    'wandb_run_id': run_id, 'model': cpu_tree(model.state_dict()),
                    'optimizers': [cpu_tree(optimizer.state_dict()) for optimizer in optimizers]},
                   output / f'checkpoint-{step:08d}.pt')
        for old in sorted(output.glob('checkpoint-*.pt'))[:-2]:
            old.unlink()

    last_phase = loss_fn = None
    last_checkpoint = last_log = time.monotonic()
    first_step = step
    last_tokens = source_step(step) * BATCH_TOKENS if step <= DECAY_START_STEP else main_tokens + consumed(step)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        while step < end_step:
            decaying = step >= DECAY_START_STEP
            muon_peak, adamw_peak = peak_lrs(step)
            if decaying:
                length, batch_order = extended, decay_order
                batch_tokens = consumed(step + 1) - consumed(step)
                offset = consumed(step) // length
                tokens = main_tokens + consumed(step + 1)
                factor = (decay_tokens - consumed(step)) / decay_tokens
            else:
                length, batch_order = base, order
                batch_tokens = (source_step(step + 1) - source_step(step)) * BATCH_TOKENS
                offset = source_step(step) * BATCH_TOKENS // base
                tokens = source_step(step + 1) * BATCH_TOKENS
                factor = min(1.0, (step + 1) / WARMUP_STEPS)
            if decaying != last_phase:
                model.rope.set_phase(decaying)
                model.activation_checkpointing = decaying
                loss_fn = torch.compile(LanguageModelLoss(model), fullgraph=True, dynamic=False)
                last_phase = decaying
            microbatch = 1 if decaying else microbatch_4k
            loss = accumulate_gradients(loss_fn, batch_order, offset=offset, sequences=batch_tokens // length,
                                        length=length, microbatch=microbatch)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM, error_if_nonfinite=True,
                                                  foreach=True)
            for optimizer, peak in zip(optimizers, (muon_peak, adamw_peak)):
                for group in optimizer.param_groups:
                    group['lr'] = peak * factor
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 5 == 0 or step == first_step + 1:
                torch.cuda.synchronize()
                elapsed = time.monotonic() - last_log
                metrics = {'train/loss': loss.item(), 'train/grad_norm': norm.item(),
                           'train/muon_lr': muon_peak * factor, 'train/adamw_lr': adamw_peak * factor,
                           'train/tokens': tokens, 'train/tokens_per_second': (tokens - last_tokens) / elapsed,
                           'train/global_batch_tokens': batch_tokens, 'train/context_length': length,
                           'gpu/allocated_bytes': torch.cuda.memory_allocated(),
                           'gpu/peak_allocated_bytes': torch.cuda.max_memory_allocated()}
                run.log(metrics, step=step)
                print(json.dumps({'step': step, 'total_steps': end_step, **metrics}), flush=True)
                last_log, last_tokens = time.monotonic(), tokens
            boundary = step in (DECAY_START_STEP, end_step)
            if step % 1000 == 0 or boundary:
                # Eager evaluation avoids another compilation per phase.
                values = evaluate(LanguageModelLoss(model), model, validation_order, length=length,
                                  microbatch=microbatch)
                run.log(values, step=step)
                print(json.dumps({'step': step, **values}), flush=True)
            if boundary or time.monotonic() - last_checkpoint >= CHECKPOINT_SECONDS:
                save()
                last_checkpoint = time.monotonic()
    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Main corpus token shards.')
    parser.add_argument('--decay-data', type=Path, required=True, help='Academic decay token shards.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    train(**vars(parser.parse_args()))
