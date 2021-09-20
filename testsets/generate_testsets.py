import pandas
import re

import argparse

import collections
import itertools
import sys

import regex


def generate_germanet_testset(gn_data_file, restrict_vocab=None, min_relation_size=10):
    tuples = [x.strip().split('\t') for x in gn_data_file]

    synsets_of_word = collections.defaultdict(set)
    for t in tuples:
        synsets_of_word[t[0]].add(t[1])
        synsets_of_word[t[3]].add(t[4])

    unambiguous = set(word for word, synsets in synsets_of_word.items() if len(synsets) == 1)
    unambiguous = set(word for word in unambiguous if regex.fullmatch(r'\p{L}+', word, flags=regex.UNICODE))

    vocab = (unambiguous & restrict_vocab) if restrict_vocab is not None else unambiguous

    selected_relations = {
        'LexRel.has_active_usage',
        'LexRel.has_appearance',
        'LexRel.has_attribute',
        'LexRel.has_component',
        'LexRel.has_consistency_of',
        'LexRel.has_content',
        'LexRel.has_diet',
        'LexRel.has_eponym',
        'LexRel.has_function',
        'LexRel.has_goods',
        'LexRel.has_habitat',
        'LexRel.has_ingredient',
        'LexRel.has_location',
        'LexRel.has_manner_of_functioning',
        'LexRel.has_material',
        'LexRel.has_member',
        'LexRel.has_no_property',
        'LexRel.has_occasion',
        'LexRel.has_origin',
        'LexRel.has_other_property',
        'LexRel.has_owner',
        'LexRel.has_part',
        'LexRel.has_product',
        'LexRel.has_production_method',
        'LexRel.has_prototypical_holder',
        'LexRel.has_prototypical_place_of_usage',
        'LexRel.has_purpose_of_usage',
        'LexRel.has_raw_product',
        'LexRel.has_relation',
        'LexRel.has_specialization',
        'LexRel.has_time',
        'LexRel.has_topic',
        'LexRel.has_usage',
        'LexRel.has_user',
        'LexRel.is_comparable_to',
        'LexRel.is_container_for',
        'LexRel.is_location_of',
        'LexRel.is_measure_of',
        'LexRel.is_member_of',
        'LexRel.is_part_of',
        'LexRel.is_product_of',
        'LexRel.is_prototypical_holder_for',
        'LexRel.is_storage_for',
        'LexRel.has_pertainym'
    }

    gn_output = []
    for t in tuples:
        if (t[0] not in vocab) or (t[3] not in vocab):
            continue

        if t[2] in selected_relations:
            gn_output.append((t[2], t[0], t[3]))

    df = pandas.DataFrame(gn_output)
    relation_size = df.groupby(0).size()
    df = df[df.apply(lambda x: relation_size[x[0]] > min_relation_size, axis=1)]
    return df.values


def generate_toefl_testset(toefl_data_file, restrict_vocab=None):
    output = []
    for line in toefl_data_file:
        if line.startswith(':'):
            continue
        prompt = tuple(line.strip().split(' '))
        if restrict_vocab and any(x not in restrict_vocab for x in prompt):
            continue
        else:
            output.append(prompt)

    return output


def generate_schm_testset(schm_data_file, restrict_vocab=None):
    output = []

    replacements = {'vorhergehend': 'vorhergehend', 'heilig': 'heilig', 'ficken': 'ficken', 'trinken': 'trinken',
        'japanisch': 'japanisch', 'schlau': 'schlau', 'flüssig': 'flüssig', 'amerikanisch': 'amerikanisch',
        'dumm': 'dumm', 'fbi': 'FBI', 'spring': 'Sprung', 'opec': 'OPEC'}

    pat = regex.compile(r'(\p{L}+)-(\p{L}+)\t([\p{N}.]+)')
    for line in schm_data_file:
        if line.startswith(':'):
            continue
        m = pat.fullmatch(line.strip())
        a = m.group(1)
        b = m.group(2)
        value = m.group(3)

        if a in replacements:
            a = replacements[a]
        else:
            a = a.capitalize()
        if b in replacements:
            b = replacements[b]
        else:
            b = b.capitalize()

        if restrict_vocab is not None and (a not in restrict_vocab or b not in restrict_vocab):
            continue
        else:
            output.append((str(float(value)), a, b))

    return output


def generate_simlex_testset(schm_data_file, restrict_vocab=None):
    output = []

    pat = regex.compile(r'(\p{L}+),(\p{L}+),(?:.*),([\p{N}.]+)$')
    for line in itertools.islice(schm_data_file, 1, None):
        m = pat.fullmatch(line.strip())
        a = m.group(1)
        b = m.group(2)
        value = m.group(3)

        if restrict_vocab is not None and (a not in restrict_vocab or b not in restrict_vocab):
            continue
        else:
            output.append((str(float(value)), a, b))

    return output


def generate_men_testset(men_data_file, restrict_vocab=None):
    output = []

    for line in itertools.islice(men_data_file, 1, None):
        x = line.split('\t')
        a = x[2]
        b = x[3]
        value = x[4]

        if restrict_vocab is not None and (a not in restrict_vocab or b not in restrict_vocab):
            continue
        else:
            output.append((value, a, b))

    return output


def generate_duden_testset(duden_data_file, restrict_vocab):
    df = pandas.read_csv(duden_data_file, sep='\t', index_col=False, header=0)
    df = df[['base', 'target', 'cand1', 'cand2', 'cand3', 'cand4']]
    if restrict_vocab is not None:
        df = df[df.apply(lambda row: all(w in restrict_vocab for w in row), axis=1)]

    return df.values


def generate_wiktionary_dataset(wiktionary_dataset, restrict_vocab, min_relation_size=0):
    df = pandas.read_csv(wiktionary_dataset, sep='\t', index_col=False, header=None)
    if restrict_vocab is not None:
        df = df[df.apply(lambda row: all(w in restrict_vocab for w in row[[1, 2]]), axis=1)]

    relation_size = df.groupby(0).size()
    df = df[df.apply(lambda x: relation_size[x[0]] > min_relation_size, axis=1)]
    return df.values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--germanet', dest='gn_data_file', type=argparse.FileType('r'), default='germanet_relations.tsv')
    parser.add_argument('--toefl', dest='toefl_data_file', type=argparse.FileType('r'), default='analogies/de_toefl_subset.txt')
    parser.add_argument('--schm', dest='schm_data_file', type=argparse.FileType('r'), default='analogies/de_re-rated_Schm280.txt')
    parser.add_argument('--simlex', dest='simlex_data_file', type=argparse.FileType('r'), default='SimLex_ALL_Langs_TXT_Format/MSimLex999_German.txt')
    parser.add_argument('--men', dest='men_data_file', type=argparse.FileType('r'), default='MEN_de/MEN_dataset_de_full.tsv')
    parser.add_argument('--duden', dest='duden_data_file', type=argparse.FileType('r'), default='duden_prompts.tsv')
    parser.add_argument('--wiktionary', dest='wiktionary_data_file', type=argparse.FileType('r'), default='wiktionary_relations.tsv')
    parser.add_argument('--vocab',  type=argparse.FileType('r'))
    parser.add_argument('--min-relation-size', type=int, default=10)
    parser.add_argument('--output', type=argparse.FileType('w'), default='testsets.tsv')

    args = parser.parse_args()
    restrict_vocab = None
    if args.vocab:
        restrict_vocab = set(x.strip() for x in args.vocab)
        args.vocab.close()

    print('dataset', 'value', '0', '1', '2', '3', '4', '5', sep='\t', file=args.output)
    for line in generate_germanet_testset(args.gn_data_file, restrict_vocab=restrict_vocab, min_relation_size=args.min_relation_size):
        print('germanet', *line, '', '', '', '', sep='\t', file=args.output)

    for line in generate_toefl_testset(args.toefl_data_file, restrict_vocab=restrict_vocab):
        print('toefl', '', *line, '', sep='\t', file=args.output)

    for line in generate_schm_testset(args.schm_data_file, restrict_vocab=restrict_vocab):
        print('schm280', *line, '', '', '', '', sep='\t', file=args.output)

    for line in generate_simlex_testset(args.simlex_data_file, restrict_vocab=restrict_vocab):
        print('simlex', *line, '', '', '', '', sep='\t', file=args.output)

    for line in generate_men_testset(args.men_data_file, restrict_vocab=restrict_vocab):
        print('men', *line, '', '', '', '', sep='\t', file=args.output)

    for line in generate_duden_testset(args.duden_data_file, restrict_vocab=restrict_vocab):
        print('duden', '', *line, sep='\t', file=args.output)

    for line in generate_wiktionary_dataset(args.wiktionary_data_file, restrict_vocab=restrict_vocab, min_relation_size=args.min_relation_size):
        print('wiktionary', *line, '', '', '', '', sep='\t', file=args.output)

    args.output.close()


if __name__ == "__main__":
    main()
