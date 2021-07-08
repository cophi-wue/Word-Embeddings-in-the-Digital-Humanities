# Word Embeddings in the Digital Humanities

This repository serves as a hub for all test datasets used in "Type- and Token-based Word Embeddings in the Digital Humanities".

## GermaNet

To use germanet_extraction.py, please make sure to have the [germanetpy](https://github.com/Germanet-sfs/germanetpy) Python API installed and have a copy of the GermaNet dataset available. To get a copy visit the official Website of [GermaNet](https://uni-tuebingen.de/fakultaeten/philosophische-fakultaet/fachbereiche/neuphilologie/seminar-fuer-sprachwissenschaft/arbeitsbereiche/allg-sprachwissenschaft-computerlinguistik/ressourcen/lexica/germanet-1/).
Before running the python script, please make sure to change the filepath to your local GermaNet copy in line 83. The script will output two TSV files: one file listing all triples of GermaNet for the relation classification (RC) task, and one creating the word choice (WC) dataset as described in the paper. As the process for constructing the WC testset includes randomly selecting words, different runs of the same script will result in slightly different sets of test instances each time.

## MEN

## Schm280 and TOEFL

Please refer to the paper [_Multilingual Reliability and “Semantic” Structure ofContinuous Word Spaces_](https://aclanthology.org/W15-0105.pdf) for a detailed description of the datasets. Both Schm280 and TOEFL can be downloaded from the [website of the University of Stuttgart](https://www.ims.uni-stuttgart.de/forschung/ressourcen/lexika/analogies/).

## Duden

## SimLex-999
