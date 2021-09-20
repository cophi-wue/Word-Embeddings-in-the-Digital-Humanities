import argparse
import itertools
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from scipy.spatial.distance import cosine


def read_syn_dict(source_dir):
    main_word = "ztz"
    syn_dict = dict()

    syns = ""
    files = os.listdir(f"{source_dir}/OEBPS/Text")
    for fname in files:
        soup = BeautifulSoup(open(f"{source_dir}/OEBPS/Text/{fname}").read(), 'html.parser')
        paras = soup.findAll("p")

        for para in paras:
            if len(list(para.children)) > 0 and list(para.children)[0].name == "span" and \
                    list(para.children)[0]["class"][0] == 'blue1':

                if main_word != "ztz":
                    syns = [re.sub("[^A-Za-zäöüÖÜÄß]", "", x) for x in syns.split(", ") if " " not in list(x)]

                    if len(syns) == 0 or len(main_word) == 1 or "," in list(main_word) or list(main_word)[0] in ["1", "2", "3"]:
                        pass
                    else:
                        if len(syns[0]) != 0:
                            syn_dict[main_word] = syns

                main_word = para.text
                syns = ""
            if 'class' in para.attrs.keys() and para["class"][0] == "noindent" and "b" not in [x.name for x in para.children] and "small" not in [x.name for x in para.children] and "sup" not in [x.name for x in para.children]:
                syns += para.text

    return syn_dict


def generate_dataset(syn_dict):
    mylists = [[x] + syn_dict[x] for x in syn_dict.keys()]
    cand = [k for k, v in Counter(itertools.chain(*mylists)).items() if v > 2]

    dataset = dict()
    for c in cand:
        right = []
        wrong = []
        for l in mylists:
            if c in l:
                right += l
                for false_friend in l:
                    for l2 in mylists:
                        if false_friend in l2 and c not in l2:
                            wrong += l2

        if len(right) > 1 and len(wrong) > 1:
            dataset[c] = {"right": list(set(right)), "wrong": list(set(wrong))}

    return dataset


def filter_dataset(dataset, ratings):
    basis = []
    for k in dataset.keys():
        right = dataset[k]["right"]
        wrong = dataset[k]["wrong"]

        right_ = [x for x in right if x not in wrong and " " not in x and x != k]
        wrong_ = [x for x in wrong if x not in right and " " not in x and x != k]

        if len(wrong_) < 4 or len(right_) < 2:
            continue
        if k not in ratings.index:
            continue

        k_vec = ratings.loc[k, :].values
        r_choice, r_cos = max([(r, 1 - cosine(k_vec, ratings.loc[r, :].values) if r in ratings.index else -np.inf) for r in right_], key=lambda x: x[1])

        w_frame = pd.DataFrame(wrong_, columns=['word'])
        w_frame['cos'] = w_frame['word'].apply(
            lambda w: 1 - cosine(k_vec, ratings.loc[w, :].values) if w in ratings.index else None)

        w_frame = w_frame[~w_frame['cos'].isna()]
        if len(w_frame) < 4:
            continue

        w_frame = w_frame.sort_values("cos", ascending=False)
        false_words = list(w_frame["word"])[:4]
        score = r_cos - w_frame['cos'].iloc[:4].mean()
        basis.append([k] + [r_choice] + false_words + [score])

    # if k == "Verzeichnis":
    #     break
    frame = pd.DataFrame(basis)
    frame.columns = ["base", "target", "cand1", "cand2", "cand3", "cand4", "schulte_cosine"]
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--duden', dest='source_dir', type=str, default='duden_sources')
    parser.add_argument('--output', type=argparse.FileType('w'), default='duden_prompts.tsv')
    parser.add_argument('--ratings', type=argparse.FileType('r'), default='affective_norms.txt')

    args = parser.parse_args()

    print('parsing epub files', file=sys.stderr)
    syn_dict = read_syn_dict(args.source_dir)
    print('generating dataset', file=sys.stderr)
    right_wrong_dataset = generate_dataset(syn_dict)
    filtered_frame = filter_dataset(right_wrong_dataset, pd.read_csv(args.ratings, sep='\t', index_col=0))

    filtered_frame.to_csv(args.output, sep="\t", index=None)
    args.output.close()


if __name__ == "__main__":
    main()
