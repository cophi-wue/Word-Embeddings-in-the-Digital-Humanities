import argparse
import collections
import os
import random
import sys
from dataclasses import dataclass
from typing import Union

from tqdm import tqdm
from transformers import BertTokenizer


def line_iter(corpus_file):
    with tqdm(unit='B', unit_scale=True, total=os.path.getsize(corpus_file.name)) as pbar:
        for line in corpus_file:
            pbar.update(len(line))
            yield line.decode('utf-8').strip().split()


@dataclass
class TrieNode:
    value: Union[str, None]
    children: dict


BERT_TOKENIZER = BertTokenizer.from_pretrained("deepset/gbert-base")
SUFFIXES = set(p for p in BERT_TOKENIZER.wordpiece_tokenizer.vocab if p.startswith('##'))


def build_query_word_trie(query_words):
    _trie = lambda: TrieNode(value=None, children=collections.defaultdict(_trie))
    trie = _trie()
    for word in query_words:
        cur = trie
        for piece in BERT_TOKENIZER.wordpiece_tokenizer.tokenize(word):
            cur = cur.children[piece]
        cur.value = word

    return trie


def search_occurrences(trie, tokenized_line):
    l = 0
    index = 0
    cur = trie
    for i, piece in enumerate(tokenized_line):
        if piece not in SUFFIXES:
            if cur.value is not None:
                yield cur.value, index, l

            cur = trie.children[piece]
            index, l = i, 1
        else:
            if piece in cur.children.keys():
                cur = cur.children[piece]
                l += 1
            else:
                cur = trie

    if cur.value is not None:
        yield cur.value, index, l


def generate_contexts(sample_probs, corpus_file):
    query_words_token_trie = build_query_word_trie(sample_probs.keys())

    for line in line_iter(corpus_file):
        for token, focus_start, focus_len in search_occurrences(query_words_token_trie, line):
            if sample_probs[token] < random.random():
                continue

            assert len(line) <= 510
            yield token, line, focus_start, focus_len


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query-words', dest='query_words', type=argparse.FileType('r'), required=True,
                        help='File of query words to sample contexts for')
    parser.add_argument('--output', type=argparse.FileType('w'), required=True, help='File to write samples to')
    parser.add_argument('--vocab', type=argparse.FileType('r'), required=True,
                        help='Vocabulary file corresponding to the corpus file, as generated from corpus_tokenizer.py')
    parser.add_argument('--corpus', type=argparse.FileType('rb'), required=True,
                        help='Processed corpus file, as generated from corpus_tokenizer.py')
    parser.add_argument('--count', type=int,
                        help='(Approximate) number of contexts to sample from the corpus, per query word',
                        default=100)
    args = parser.parse_args()
    print(args, file=sys.stderr)

    print("loading vocabulary", file=sys.stderr, flush=True)
    vocab = {line.strip().split(' ')[0]: int(line.strip().split(' ')[1]) for line in args.vocab}
    args.vocab.close()

    print("loading search words", file=sys.stderr, flush=True)
    target_num = args.count
    sample_probs = {}
    counter = {}
    with args.query_words as f:
        for line in f:
            line = line.strip().split(' ')
            tok = line[0]
            if tok not in vocab.keys():
                pass
                # print(tok + " not in vocab", file=sys.stderr, flush=True)
            else:
                sample_probs[tok] = target_num / vocab[tok]
                counter[tok] = 0
    args.query_words.close()
    del vocab

    random.seed(15452)
    print('token', 'counter', 'context_len', 'focus_index', 'focus_len', 'context', file=args.output, sep='\t')
    for token, context, focus_start, focus_len in generate_contexts(sample_probs, args.corpus):
        assert len(context) <= 510
        counter[token] += 1
        print(token, counter[token], len(context), focus_start, focus_len, ' '.join(context), file=args.output,
              sep='\t')

    args.output.close()
