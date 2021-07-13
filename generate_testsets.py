import argparse

import collections
import itertools
import sys

import regex


def generate_germanet_testset(gn_data_file, restrict_vocab=None):
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

    return list(sorted(gn_output, key=lambda x: x[0]))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--germanet', dest='gn_data_file', type=argparse.FileType('r'), default='germanet_relations.tsv')
    parser.add_argument('--vocab',  type=argparse.FileType('r'))
    parser.add_argument('--output', type=argparse.FileType('w'), default='testsets.tsv')

    args = parser.parse_args()
    restrict_vocab = None
    if args.vocab:
        restrict_vocab = set(x.strip() for x in args.vocab)
        args.vocab.close()

    for gn_line in generate_germanet_testset(args.gn_data_file, restrict_vocab=restrict_vocab):
        print('germanet\t' + '\t'.join(gn_line), file=args.output)


    # TODO wiktionary, TOEFL, Schm280, MEN, SimLex
    args.output.close()


if __name__ == "__main__":
    main()
