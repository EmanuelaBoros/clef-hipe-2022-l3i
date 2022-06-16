#### Run the code

Details in run.sh

##### Dataset Annotation


```
TOKEN	NE-COARSE-LIT	NE-COARSE-METO	NE-FINE-LIT	NE-FINE-METO	NE-FINE-COMP	NE-NESTED	NEL-LIT	NEL-METO	MISC
Wienstrasse	I-LOC	O	O	O	O	O	null	O	SpaceAfter

```

#### Requirements
```
pip install -r requirements.txt
```

#### CLEF-HIPE-2020-scorer


```
python clef_evaluation.py -o ../clef-hipe-2022-l3i/KB-NER/kb/datasets/hipe2020/de/results/ --pred ../clef-hipe-2022-l3i/KB-NER/kb/datasets/hipe2020/de/results/predictions_test.tsv --ref ../clef-hipe-2022-l3i/KB-NER/kb/datasets/hipe2020/de/HIPE-2022-v2.1-hipe2020-test-ELmasked-de.tsv  --task nerc_fine
```
