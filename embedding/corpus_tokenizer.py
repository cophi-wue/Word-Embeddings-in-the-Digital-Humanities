import argparse
import collections
import itertools
import os
import sys

import regex
from spacy.lang.de import German
from tqdm import tqdm
from transformers import BertTokenizer


def _tokenize_text(text):
    # emulates BERT's tokenization into words (not pieces),
    # cf. BasicTokenizer and _is_punctuation from the transformers package.
    # By fully tokenizing into words, we avoid (finer) re-tokenization in
    # CachedTokenizedSentence, and can immediately perform the wordpiece
    # tokenization.
    for s in regex.split(r'(?:\p{Other}|\p{Separator})|(\p{punct}|[\$+<=>^`|~]|\p{Han})', text,
                         flags=regex.UNICODE | regex.WORD):
        # if len(s) > 0 and s[0] not in {' ', '\n'}:
        if s is None or len(s) == 0:
            continue

        yield s


def sentencize(text):
    nlp = German()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    nlp.max_length = 100_000_000
    for sent in nlp(text).sents:
        yield sent.string


def sentence_iter(corpus_file):
    with tqdm(unit='B', unit_scale=True, smoothing=0.05, total=os.path.getsize(corpus_file.name)) as pbar:
        accumulator = ''
        for line in corpus_file:
            pbar.update(len(line))
            accumulator += line.decode('utf-8')
            if len(accumulator) > 1_000_000:
                sentences = list(sentencize(accumulator))
                for sent in sentences[:-2]:
                    tokenized = list(_tokenize_text(sent))
                    if len(tokenized) > 510:
                        continue
                    yield tokenized

                accumulator = sentences[-1]

        # process remaining text in the accumulator
        for sent in sentencize(accumulator):
            tokenized = list(_tokenize_text(sent))
            if len(tokenized) > 510:
                continue

    corpus_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=argparse.FileType('rb'), required=True, help='Unprocessed corpus file')
    parser.add_argument('--output', type=argparse.FileType('w'), required=True,
                        help='File to write the tokenized and sentencized corpus to')
    parser.add_argument('--vocab-out', dest='vocab_file', type=argparse.FileType('w'), required=True,
                        help='File to write token frequencies to')
    parser.add_argument('--vocab-limit', dest='vocab_limit', type=int,
                        help='Limits the written vocabulary to the top N most frequent words', default=20_000_000)
    args = parser.parse_args()
    print(args, file=sys.stderr)

    tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")
    tokenizer.do_basic_tokenize = False

    frequencies = collections.defaultdict(lambda: 0)
    for sent in sentence_iter(args.input):
        sent = list(sent)

        subword_tokens = list(itertools.chain(*map(tokenizer.wordpiece_tokenizer.tokenize, sent)))
        if len(subword_tokens) > 510:
            continue
        print(*subword_tokens, end='\n', sep=' ', file=args.output)
        for tok in sent:
            frequencies[tok] += 1
    args.output.close()

    sorted_output = list(reversed(sorted(frequencies.items(), key=lambda it: it[1])))
    for o in tqdm(sorted_output[:args.vocab_limit]):
        print(o[0], o[1], file=args.vocab_file)
    args.vocab_file.close()
