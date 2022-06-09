# clef-hipe-2022-l3i

For obtaining the contexts in CoNLL format, we run `context_process.py`, as in the following example, for AJMC in English.
Before doing so, we need a file that contains all contexts in one, and thus:

`cat datasets/hipe2020/en/kb/en_wk5m_simple_time/HIPE-2022-v2.1-ajmc-train-en.kb datasets/ajmc/en/kb/en_wk5m_simple_time/HIPE-2022-v2.1-ajmc-dev-en.kb datasets/ajmc/en/kb/en_wk5m_simple_time/HIPE-2022-v2.1-ajmc-test_allmasked-en.kb > datasets/ajmc/en/kb/en_wk5m_simple_time/HIPE-2022-v2.1-ajmc-en.kb`

And then:

```
python context_process.py --retrieval_file datasets/ajmc/en/kb/en_wk5m_simple_time/HIPE-2022-v2.1-ajmc-en.kb \
    --conll_folder datasets/ajmc/en/ \
    --lang en \
    --version 3 \
    --dataset ajmc
```

Which will generate the files in a folder `hipe_doc_full_wiki_v3` in `datasets/ajmc/en/`.


