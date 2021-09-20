# Word Embeddings in the Digital Humanities

This repository serves as a hub for followng resources used in “Type- and Token-based Word Embeddings in the Digital Humanities”:
1. Scripts and artifacts to fully reproduce the used test dataset,
2. Scripts to perform the embedding and evaluation.

## Datasets

To fully reproduce the used test dataset, follow these steps. **Note that we restrict our test set to those tokens present in file `evaluation/evaluation_vocabulary`.**

```shell
# 0. Switch to testsets directory
cd testsets/

# 1. Install requirements
pip install -r requirements.txt

# 2. Download the Schm280 and TOEFL dataset from IMS Stuttgart,
#    download the MSimlex999 dataset from the Project's website,
#    and download the affection rating dataset from IMS Stuttgart
wget https://www.ims.uni-stuttgart.de/documents/ressourcen/lexika/analogies_ims/analogies.zip
unzip analogies.zip
wget https://leviants.com/wp-content/uploads/2020/01/SimLex_ALL_Langs_TXT_Format.zip
unzip SimLex_ALL_Langs_TXT_Format.zip
wget https://www.ims.uni-stuttgart.de/documents/ressourcen/experiment-daten/affective_norms.txt.gz
gunzip affective_norms.txt.gz

# 3. Acquire a copy of Germanet v14, place it in folder "GNV140", and verify its integrity
cat $(find GN_V140/GN_V140_XML/* | sort) | sha256sum
# > 09ca06d178edf193648807cb983181670fd443b458e8c148a012808780962925  -

# 4. Generate all Germanet Relations from the Corpus
python germanet_extraction.py --germanet ./GN_V140/GN_V140_XML

# 5. Acquire a copy of "Das Wörterbuch der Synonyme", verify its integrity, and extract
#    its content
sha256sum duden_synonym_woerterbuch.epub
# > 8389728c500fc8653bc5a7804e6c4fa2fe93eb5e8ef81679d4ac02ce00916407  duden_synonym_woerterbuch.epub
unzip duden_synonym_woerterbuch.epub -d duden_sources

# 6. Generate Duden synonymy prompts
python duden_extraction.py --duden duden_sources --ratings affective_norms.txt

# 7. Download the 2021-07-01 German Wiktionary Database dump
wget https://dumps.wikimedia.org/dewiktionary/20210701/dewiktionary-20210701-pages-articles.xml.bz2

# 8. Generate Wiktionary relation pairs
python wiktionary_extraction.py

# 9. Merge all datasets and filter with our evaluation vocabulary
python generate_testsets.py --vocab ./evaluation_vocabulary

# Done: The full testset table is stored as testsets.tsv
```

### GermaNet

To generate the (full) GermaNet relation dataset, you need to get a copy of the GermaNet database.
You can apply for a license for GermaNet at the [website of the University of Tübingen](https://uni-tuebingen.de/fakultaeten/philosophische-fakultaet/fachbereiche/neuphilologie/seminar-fuer-sprachwissenschaft/arbeitsbereiche/allg-sprachwissenschaft-computerlinguistik/ressourcen/lexica/germanet-1/).
See also the papers [Birgit Hamp and Helmut Feldweg, “GermaNet – a Lexical-Semantic Net for German”](https://aclanthology.org/W97-0802/) and [Verena Henrich and Erhard Hinrichs, “GernEdiT – The GermaNet Editing Tool”](http://www.lrec-conf.org/proceedings/lrec2010/pdf/264_Paper.pdf).
We use Release 14.

You can then generate the full dataset of all relations by invoking `python evaluation/germanet_extraction.py --germanet GN_V140_Root/GN_V140_XML --output output_germanet.tsv`

### MEN

We have machine-translated the [original MEN Test Collection](https://staff.fnwi.uva.nl/e.bruni/MEN) by Bruni, Tran, and Baroni.
See the separate [README](./testsets/MEN_de/README.md) for further details and license information.
See also the publication [Elia Bruni, Nam Khanh Tran and Marco Baroni, “Multimodal Distributional Semantics”](https://doi.org/10.1613/jair.4135).

### Schm280 and TOEFL

Please refer to the paper [“Multilingual Reliability and ‘Semantic’ Structure of Continuous Word Spaces”](https://aclanthology.org/W15-0105.pdf) for a detailed description of the datasets.
Both Schm280 and TOEFL can be separately downloaded from the [website of the University of Stuttgart](https://www.ims.uni-stuttgart.de/forschung/ressourcen/lexika/analogies/).

### Duden

To generate the (full) datast, you need (a) to buy a digital copy of _Das Wörterbuch der Synonyme_ (EAN: 9783411913169) and (b)
need to download the (free) [Affective Norms Wordlist](http://www.schulteimwalde.de/resources/affective-norms.html) by Koper and Schute im Walde.
The Wordlist is described in the paper [Maximilian Köper and Sabine Schulte im Walde, “Automatically Generated Norms of Abstractness, Arousal, Imageability and Valence for 350 000 German Lemmas”](https://aclanthology.org/L16-1413/).

You can then generate the full dataset by invoking `python evaluation/duden_extraction.py --duden extracted_epub_root --ratings affective_norms.txt --output output_duden.tsv`.

### Wiktionary

You can generate a (full) dataset of relations from a specific German Wiktionary database dump.
We use the July 2021 dump, which you can download [here](https://dumps.wikimedia.org/dewiktionary/20210701/) (“Articles, templates, media/file descriptions, and primary meta-pages”).

After downloading, you can generate the full dataset by invoking `python evaluation/wiktionary_extraction.py --wiktionary wiktionary_dump.xml.bz2 --output output_wiktionary.tsv`.

### SimLex-999

See the paper [Ira Leviant and Roi Reichart, “Separated by an Un-common Language: Towards Judgment Language Informed Vector Space Modeling”](https://arxiv.org/pdf/1508.00106.pdf) for more information and the corresponding [website](https://leviants.com/multilingual-simlex999-and-wordsim353/) to download the dataset.

## Embedding and Evaluation

Here, we describe the steps to generate a distilled type-based embedding from BERT's Transformer output for a set of query words, sampling contextualized sentences from a corpus.

1. Install requirements:
   ```
   pip install -r embedding/requirements.txt
   ```


2. Tokenize the entire corpus for BERT and generate frequency statistics.
   ```
   python ./embedding/corpus_tokenizer.py --input CORPUS_FILE \
      --output processed_corpus.txt --vocab-out corpus_vocab.txt
   ```


3. Sample (100) context sentences for query words in line-separated list `query_words.txt`.
   ```
   python ./embedding/corpus_tokenizer.py --corpus processed_corpus.txt \
       --vocab corpus_vocab.txt --query-words query_words.txt --count 100 \
       --output contexts.txt
   ```

4. Perform the BERT forward passes. You can pass a comma-separated list of desired vectorization/pooling/aggregation methods to use.
   ```
   python ./embedding/embedder.py --contexts context.txt \
       --vectorizations sum --poolings nopooling,mean --aggregations mean,median \
       --output-prefix embeddings_
   ```
   This will write an embedding for each distillation combination, e.g. `embeddings_sum-nopooling-mean.bin`, `embeddings_sum-nopooling-median.bin`, ...


5. Evaluate embedding(s) on a testsets file, generated by `generate_testsets.py` as described above.
   ```
   python ./embedding/evaluation.py --testsets testsets.tsv --output evaluation_output.tsv embeddings_*.bin
   ```

## License

This repository (with exception to the German translation of the MEN Test Collection) is licensed under TODO

