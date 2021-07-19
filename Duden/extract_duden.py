import json
import os
import re
import pandas as pd
import numpy as np
from collections import Counter
from lxml import etree
from bs4 import BeautifulSoup
from scipy.spatial.distance import cosine




main_word = "ztz"
syns = ""
output = []
syn_dict = dict()
files = os.listdir("work/OEBPS/Text")

for fname in files:
    
    soup = BeautifulSoup(open("work/OEBPS/Text/"+fname).read(), 'html.parser')
    paras = soup.findAll("p")
    
    for para in paras:

        if list(para.children)[0].name == "span" and list(para.children)[0]["class"][0] == 'blue1':

            if main_word != "ztz":
                syns = [re.sub("[^A-Za-zäöüÖÜÄß]","",x) for x in syns.split(", ") if " " not in list(x)]

                if len(syns) == 0 or len(main_word) == 1 or "," in list(main_word) or list(main_word)[0] in ["1","2","3"]:
                    pass
                else:
                    if len(syns[0]) != 0:

                        output.append([main_word, syns])
                        syn_dict[main_word] = syns

            main_word = para.text
            syns = ""
        if para["class"][0] == "noindent" and "b" not in [x.name for x in para.children] and "small" not in [x.name for x in para.children] and "sup" not in [x.name for x in para.children]:
            syns += para.text



out = pd.DataFrame(output)
out.columns = ["word","syn"]
#out.to_csv("duden_synonyme.tsv", sep="\t")
#with open("duden_synonyme.json","w") as f:
#    json.dump(syn_dict,f)

data = syn_dict

mylists = []
for k in list(data.keys()):
    
    mylists.append(data[k]+[k])
    
for l in mylists:
    for word in l:
        candidates = [x for x in mylists if l in x]
        
        if len(candidates) > 0:
            print(l)
            print(candidates)
            print("----")

frame = pd.DataFrame.from_dict(Counter([x for y in mylists for x in y]), orient="index")
cand = list(frame[frame[0]>2].index)

dataset = dict()
for c in cand:
    right = []
    wrong = []
    for l in mylists:
      
        if c in l:
            right+=l
            for false_friend in l:
                
                for l2 in mylists:
      
                    if false_friend in l2 and c not in l2:
                        wrong+=l2
    if len(right) > 1 and len(wrong) > 1:          
        dataset[c] = {"right":list(set(right)),"wrong":list(set(wrong))}
        
#with open("duden_dataset.json","w") as f:
#    json.dump(dataset,f)
    
data = dataset
schulte = pd.read_csv("affective_norms.txt", sep="\t", index_col=0)

basis = []
for k in data.keys():
    
    right = data[k]["right"]
    wrong = data[k]["wrong"]
    
    right_ = [x for x in right if x not in wrong and " " not in x and x != k]
    wrong_ = [x for x in wrong if x not in right and " " not in x and x != k]
    
    if len(wrong_) < 4 or len(right_) < 2:
        continue
    try:
        k_vec = np.array(schulte.loc[k,:])
    except:
        continue

    i = 0
    c = 0
    r_choice = ""
    
    for r in right_:
        try:
            cos = 1-cosine(k_vec,np.array(schulte.loc[r,:]))

            if cos > c:
                c = cos
                r_choice = r
        except:
            continue
            
    selection = []

    for w in wrong_:
        try:
            cos = 1-cosine(k_vec,np.array(schulte.loc[w,:]))
            selection.append([w,cos])
        except:
          
            continue
    

    if len(selection) < 4:
        continue
    
        
    sel_frame = pd.DataFrame(selection)
    sel_frame.columns = ["word","cos"]
    sel_frame = sel_frame.sort_values("cos",ascending=False)
    false_words = list(sel_frame["word"])[:4]
        
    score = (1-cosine(k_vec,np.array(schulte.loc[r_choice,:])))-np.mean(sel_frame.iloc[:3,1])
    
        
    basis.append([k]+[r_choice]+false_words+[score])
    
   # if k == "Verzeichnis":
   #     break
frame = pd.DataFrame(basis)
frame.columns = ["base","target","cand1","cand2","cand3","cand4","schulte_cosine"]
frame.to_csv("duden_dataset.tsv", sep="\t")

