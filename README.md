# Word Embeddings in the Digital Humanities

This repository serves as a hub for all test datasets used in "Type- and Token-based Word Embeddings in the Digital Humanities".

## Quick Start

To fully reproduce the used test dataset, follow the following steps:

```shell
# 0. Install requirements
pip install -r requirements.txt

# 1. Acquire a copy of Germanet v14, place it in folder GNV140, and verify its integrity
cat $(find GN_V140/GN_V140_XML/* | sort) | sha256sum
# > 09ca06d178edf193648807cb983181670fd443b458e8c148a012808780962925  -

# 2. Download the Schm280 and TOEFL dataset from IMS Stuttgart,
#    and download the MSimlex999 dataset from the Project's website
wget https://www.ims.uni-stuttgart.de/documents/ressourcen/lexika/analogies_ims/analogies.zip
unzip analogies.zip
wget https://leviants.com/wp-content/uploads/2020/01/SimLex_ALL_Langs_TXT_Format.zip
unzip SimLex_ALL_Langs_TXT_Format.zip

# 3. TODO Duden, Wiktionary

# 4. Generate all Germanet Relations from the Corpus
python germanet_extraction.py --germanet ./GN_V140/GN_V140_XML

# 5. Merge all datasets and filter with our evaluation vocabulary
python generate_testsets.py --vocab ./evaluation_vocabulary

# Done: The full testset table is stored as testsets.tsv
```

## GermaNet

To use `germanet_extraction.py`, please make sure to have the [germanetpy](https://github.com/Germanet-sfs/germanetpy) Python API installed and have a copy of the GermaNet dataset available. To get a copy visit the official Website of [GermaNet](https://uni-tuebingen.de/fakultaeten/philosophische-fakultaet/fachbereiche/neuphilologie/seminar-fuer-sprachwissenschaft/arbeitsbereiche/allg-sprachwissenschaft-computerlinguistik/ressourcen/lexica/germanet-1/).
Run `python germanet_extraction.py <path/to/GNV140_XML>`.
The script will output one TSV file containing all relation triples of GermaNet.
Note that not all triples are present in the relation classification (RC) task, but only the subset that is generated from `generate_testsets.py`.

## MEN

## Schm280 and TOEFL

Please refer to the paper [_Multilingual Reliability and “Semantic” Structure ofContinuous Word Spaces_](https://aclanthology.org/W15-0105.pdf) for a detailed description of the datasets. Both Schm280 and TOEFL can be downloaded from the [website of the University of Stuttgart](https://www.ims.uni-stuttgart.de/forschung/ressourcen/lexika/analogies/).

## Duden

You need to buy a digital copy of "Das Wörterbuch der Synonyme" (EAN: 9783411913169) and place it in the folder duden. To gegerate the dataset
rename the epub to "work.epub", extract its content and run "extract_duden.py". 

## Wiktionary



## SimLex-999
