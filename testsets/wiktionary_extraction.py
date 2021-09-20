import argparse
import os
import unicodedata

import Levenshtein
import pandas
import pandas as pd
import re
import regex
from bz2file import BZ2File
from tqdm import tqdm
from wiktionary_de_parser import Parser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wiktionary', type=str, default='dewiktionary-20210701-pages-articles.xml.bz2')
    parser.add_argument('--output', type=argparse.FileType('w'), default='wiktionary_relations.tsv')

    args = parser.parse_args()

    bz = BZ2File(args.wiktionary)

    substantive = []
    adjektive = []
    verben = []
    derivs = []
    adj_targets = ["bar", "en", "erig", "ern", "fach", "frei", "haft", "ig", "isch",
               "lich", "los", "mäßig", "sam", "sch"]
    subst_targets = ["chen", "e", "ei", "el", "en", "er", "heit", "ien", "iker", "in", "keit", "lein", "ler", "ling", "mut",
               "nis", "rich", "sal", "schaft", "sel", "tum", "ung"]

    with tqdm(unit='B', unit_scale=True, smoothing=0.05, total=os.path.getsize(args.wiktionary)) as pbar:
        for record in Parser(bz):
            pbar.update(bz._fp.tell() - pbar.n)

            if not regex.fullmatch(r'\p{L}+', record['title']): continue

            if 'langCode' not in record or record['langCode'] != 'de':
                continue

            if re.search(r'\|Adjektiv\|', record['wikitext']):
                try:
                    target, lemma, base = process_deriv(record, adj_targets)
                    derivs.append(['adj_'+target, lemma, base])
                except: pass
            if re.search(r'\|Substantiv\|', record['wikitext']):
                try:
                    target, lemma, base = process_deriv(record, subst_targets)
                    derivs.append(['subst_'+target, lemma, base])
                except: pass


            if 'flexion' in record.keys():
                flexion = record["flexion"]
                wortart = list(record["pos"].keys())[0]
                if wortart == "Substantiv":
                    substantive.append(flexion)
                if wortart == "Adjektiv":
                    adjektive.append(flexion)
                if wortart == "Verb":
                    verben.append(flexion)
                    flexion["Infinitiv"] = record["title"]

    print_verb_infl(verben, args.output)
    print_adj_infl(adjektive, args.output)
    print_subst_infl(substantive, args.output)
    print_deriv(derivs, args.output)


def process_deriv(record, targets):
    for t in targets:
        lemma = record['title']
        if not lemma.endswith(t): continue
        herkunft = re.search(r'{{(Herkunft|Ableitung)}}[^{]*(\[\[Ableitung]][^{]*){{', record['wikitext'], re.MULTILINE)
        if herkunft is None: continue
        herkunft = herkunft.group(2).replace('\n', ' ')
        if not re.search(r"''\[\[-"+t+"]]", herkunft): continue
        base = [b[0] for b in regex.findall(r"''\[\[(\p{L}+)]](.,;)?''", herkunft)]

        def check_prefix(a,b):
            return unicodedata.normalize('NFD', a[0]).lower() != unicodedata.normalize('NFD', b[0]).lower()

        if len(base) == 0: continue

        if len(base) == 1:
            candidate = base[0]
            if not check_prefix(candidate, lemma): continue
            return t, lemma, candidate
        else:
            # heuristic by closest levenshtein distance
            distances = [(b, Levenshtein.distance(lemma.lower(), b.lower() + t)) for b in base if check_prefix(lemma, b)]
            candidate, dist = min(distances, key=lambda x: x[1])

            if dist <= 3:
                return t, lemma, candidate






def print_subst_infl(substantive, out):
    substantive = pd.DataFrame(substantive)
    labels = dict([('Nominativ Singular', 'nom_sg'), ('Nominativ Plural', 'nom_pl'), ('Dativ Plural', 'dat_pl'),
                   ('Genitiv Singular', 'gen_sg')])
    substantive = substantive[labels.keys()].dropna().rename(columns=labels)
    substantive.drop_duplicates(subset='nom_sg', keep=False, inplace=True)
    substantive = substantive[
        substantive.applymap(lambda x: len(x) >= 2 and regex.fullmatch(r'\w+', x) is not None).all(axis=1)]
    for col in labels.values():
        if col == 'nom_sg': continue
        if col == 'nom_pl' or col == 'dat_pl':
            selection = substantive[substantive['dat_pl'] != substantive['nom_pl']][['nom_sg', col]]
        else:
            selection = substantive[['nom_sg', col]]
        selection = selection[selection.apply(lambda x: x == selection[col]).sum(axis=1) == 1].drop_duplicates()
        for i, row in selection.iterrows():
            print('infl_subst_' + col, row['nom_sg'], row[col], sep='\t', file=out)


def print_adj_infl(adjektive, out):
    adjektive = pd.DataFrame(adjektive)
    adjektive.drop_duplicates(subset='Positiv', keep=False, inplace=True)
    for col in ['Komparativ', 'Superlativ']:
        selection = adjektive[adjektive.apply(lambda x: x == adjektive[col]).sum(axis=1) == 1][
            ['Positiv', col]].drop_duplicates()
        for i, row in selection.iterrows():
            print('infl_adj_' + col.lower(), row['Positiv'], row[col], sep='\t', file=out)


def print_verb_infl(verben, out):
    verben = pd.DataFrame(verben)
    verben = verben.drop(verben[verben.Präsens_ich.isna()].index)
    labels = dict([('Infinitiv', 'inf'), ('Präsens_ich', 'sg_1p_präsens'), ('Präsens_du', 'sg_2p_präsens'),
                   ('Präteritum_ich', 'sg_1p_prät_indikativ'), ('Partizip II', 'partizip_perfekt'),
                   ('Konjunktiv II_ich', 'sg_1p_prät_konjunktiv')])
    verben = verben[labels.keys()].dropna().rename(columns=labels)
    # verben.drop_duplicates(subset='inf', inplace=True)
    verben = verben[verben.applymap(lambda x: len(x) >= 2 and regex.fullmatch(r'\w+', x) is not None).all(axis=1)]
    for col in labels.values():
        if col == 'inf': continue
        selection = verben[verben.apply(lambda x: x == verben[col]).sum(axis=1) == 1][['inf', col]].drop_duplicates()
        for i, row in selection.iterrows():
            print('infl_verb_' + col, row['inf'], row[col], sep='\t', file=out)


def print_deriv(derivs, out):
    df = pandas.DataFrame(derivs, columns=['derivation', 'base', 'lemma'])
    df.drop_duplicates(subset=['derivation', 'base'], keep=False, inplace=True)
    for row in df.sort_values('derivation').itertuples():
        print('derivations_'+row.derivation, row.lemma, row.base, sep='\t', file=out)



if __name__ == "__main__":
    main()
