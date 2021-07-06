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


def word_choice_set(result, outpath):
    for lex_unit in result:
        ant = None
        hyp = None
        syn = None
        synset = lex_unit.synset
        hypernyms = [x.lexunits for x in synset.direct_hypernyms]
        hypernyms = [item for sublist in hypernyms for item in sublist]
        try:
            hyp = random.choice(hypernyms).orthform
            for rel, values in lex_unit.relations.items():
                if str(rel)=='LexRel.has_synonym':
                    syn = random.choice(list(values)).orthform
                if str(rel)=='LexRel.has_antonym':
                    ant = random.choice(list(values)).orthform
        except IndexError:
            continue
        if ant and hyp and syn:
            rand = random.choice(result).orthform
            with open(outpath, 'a') as outfile:
                outfile.write('{}\t{}\t{}\t{}\t{}\n'.format(lex_unit.orthform, syn, hyp, ant, rand))

                        
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

                    
def remove_gender_pairs_WC(path):
    fokuswords = []
    with open(path, 'r') as f:
        lines = f.readlines()
        os.remove(path)
        for line in lines:
            words = re.split('\t|\n', line)[:5]
            if words[0] == words[1]+'in' or words[1] == words[0]+'in' or words[1] in fokuswords or words[0] in fokuswords:
                pass
            else:
                fokuswords.append(words[0])
                with open(path, 'a') as outfile:
                    outfile.write('{}\t{}\t{}\t{}\t{}\n'.format(words[0], words[1], words[2], words[3], words[4]))
     
    
def main():
    data_path = "../GermaNet/GN_V140/GN_V140_XML"
    germanet = Germanet(data_path)
    
    for wordclass in ['N', 'V', 'ADJ']:
        words = filter_words(wordclass, germanet, return_synsets=False)
        word_choice_set(words, 'germanet_WC.csv')
        rel_class_set(words, wordclass, 'germanet_RC.csv')
    remove_gender_pairs_WC('germanet_WC.csv')
        
        
if __name__ == "__main__":
    main()