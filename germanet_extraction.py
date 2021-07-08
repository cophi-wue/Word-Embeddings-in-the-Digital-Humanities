import random
import os
import re

from germanetpy.germanet import Germanet
from germanetpy.synset import WordCategory
from germanetpy.filterconfig import Filterconfig


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

                        
def rel_class_set(result, wordclass, outpath):
    for lex_unit in result:
        for hypernym in lex_unit.synset.direct_hypernyms:
            for hyper_lex in hypernym.lexunits:
                with open(outpath, 'a') as outfile:
                    outfile.write('{}\t{}_has_direct_hypernym\t{}\n'.format(lex_unit.orthform, wordclass, hyper_lex.orthform))

        for hyponym in lex_unit.synset.direct_hyponyms:
            for hypo_lex in hyponym.lexunits:
                with open(outpath, 'a') as outfile:
                    outfile.write('{}\t{}_has_direct_hyponym\t{}\n'.format(lex_unit.orthform, wordclass, hypo_lex.orthform))

        for rel, values in lex_unit.relations.items():
            for v in values:
                with open(outpath, 'a') as outfile:
                    outfile.write('{}\t{}\t{}\n'.format(lex_unit.orthform, str(rel), v.orthform))


def main():
    data_path = "../GermaNet/GN_V140/GN_V140_XML"
    germanet = Germanet(data_path)
    
    for wordclass in ['N', 'V', 'ADJ']:
        words = filter_words(wordclass, germanet, return_synsets=False)
        rel_class_set(words, wordclass, 'germanet_RC.csv')
        
        
if __name__ == "__main__":
    main()
