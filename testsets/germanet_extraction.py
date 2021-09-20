import argparse

from germanetpy.filterconfig import Filterconfig
from germanetpy.germanet import Germanet
from germanetpy.synset import WordCategory


def filter_words(wordclass, germanet, return_synsets=False):
    if wordclass == 'N':
        wordclass = WordCategory.nomen
    elif wordclass == 'V':
        wordclass = WordCategory.verben
    elif wordclass == 'ADJ':
        wordclass = WordCategory.adj
    filterconfig = Filterconfig('.+', regex=True)
    filterconfig.word_categories = [wordclass]
    all_words = list(filterconfig.filter_lexunits(germanet))
    if return_synsets is True:
        all_words = [r.synset for r in all_words]
        all_words = list(set(all_words))
    return all_words


def rel_class_set(result, wordclass, outfile):
    for lex_unit in result:
        for hypernym in lex_unit.synset.direct_hypernyms:
            for hyper_lex in hypernym.lexunits:
                outfile.write('{}\t{}\t{}_has_direct_hypernym\t{}\t{}\n'.format(lex_unit.orthform, lex_unit.synset.id, wordclass, hyper_lex.orthform, hyper_lex.synset.id))

        for hyponym in lex_unit.synset.direct_hyponyms:
            for hypo_lex in hyponym.lexunits:
                outfile.write('{}\t{}\t{}_has_direct_hyponym\t{}\t{}\n'.format(lex_unit.orthform, lex_unit.synset.id, wordclass, hypo_lex.orthform, hypo_lex.synset.id))

        for rel, values in lex_unit.relations.items():
            for v in values:
                outfile.write('{}\t{}\t{}\t{}\t{}\n'.format(lex_unit.orthform, lex_unit.synset.id, str(rel), v.orthform, v.synset.id))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--germanet', dest='gn_input', type=str, default='GN_V140/GN_V140_XML')
    parser.add_argument('--output', type=argparse.FileType('w'), default='germanet_relations.tsv')

    args = parser.parse_args()
    germanet = Germanet(args.gn_input)

    for wordclass in ['N', 'V', 'ADJ']:
        words = filter_words(wordclass, germanet, return_synsets=False)
        rel_class_set(words, wordclass, args.output)

    args.output.close()


if __name__ == "__main__":
    main()
