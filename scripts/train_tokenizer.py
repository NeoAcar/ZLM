"""Train the 49,152-token byte-level BPE tokenizer from Parquet text shards.

A 65,536-token vocabulary is trained first; the released tokenizer keeps its first 49,152 IDs and
the matching prefix of its merge list.
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


SPECIAL_TOKENS = [
    '<|bos|>', '<|eos|>', '<|pad|>', '<|system|>', '<|user|>', '<|assistant|>',
    '<|tool|>', '<|message_end|>', '<|turn_end|>', '<|tool_call|>',
    '<|tool_call_end|>', '<|think|>', '<|/think|>',
] + [f'<|reserved_{i}|>' for i in range(115)]
REPEATED_LETTER_PATTERN = r" ?['’]?(\p{L})\1{4,}"
PRETOKEN_PATTERN = (
    r" ?['’]?(\p{L})\1{3}(?=\1|\z)|"
    r'(?:\r\n|[\r\n\v\f\x{0085}\x{2028}\x{2029}])+'
    r'|\t+'
    r"| ?['’]?[\p{L}\p{M}]+"
    r'|\p{N}'
    r'| +(?= )| +'
    r'|[^\S \t\r\n\v\f\x{0085}\x{2028}\x{2029}]+'
    r"|[^\p{L}\p{M}\p{N}\s'’]{1,4}|['’]"
)


def make_tokenizer():
    tokenizer = Tokenizer(models.BPE(
        dropout=None, unk_token=None,
        fuse_unk=False, byte_fallback=False, ignore_merges=False,
    ))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(REPEATED_LETTER_PATTERN), behavior='isolated', invert=False),
        pre_tokenizers.Split(Regex(PRETOKEN_PATTERN), behavior='isolated', invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False, trim_offsets=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.encode_special_tokens = True
    return tokenizer


def text_batches(directory):
    for path in sorted(Path(directory).glob('*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=['text']):
            yield batch.column(0).to_pylist()


def train(directory, vocab_size):
    tokenizer = make_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2, max_token_length=32, show_progress=True,
        special_tokens=[AddedToken(token, single_word=False, lstrip=False, rstrip=False,
                                   normalized=False, special=True) for token in SPECIAL_TOKENS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(text_batches(directory), trainer=trainer)
    return tokenizer


def merge_prefix(tokenizer, size):
    """Keep the first size IDs and the merge-chain prefix whose results stay in that vocabulary."""
    data = json.loads(tokenizer.to_str())
    vocab = {text: index for text, index in data['model']['vocab'].items() if index < size}
    prefix = []
    for left, right in data['model']['merges']:
        if left + right not in vocab:
            break
        prefix.append([left, right])
    data['model']['vocab'] = vocab
    data['model']['merges'] = prefix
    return Tokenizer.from_str(json.dumps(data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True, help='Directory of Parquet shards with a text column.')
    parser.add_argument('--output', type=Path, required=True, help='Path of the tokenizer JSON to write.')
    args = parser.parse_args()
    merge_prefix(train(args.train, 65_536), 49_152).save(str(args.output))
