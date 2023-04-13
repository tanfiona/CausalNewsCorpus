
### Steps to process Subtask 2 Annotations

1. Download JSON annotations from WebAnno tool

2. [`unzip.py`](unzip.py) : Unzip latest annotations (from annotations folder)

3. [`subtask2_annotators.py`](subtask2_annotators.py) : Check annotators' work for basic errors, like missing spans, links, etc. If all okay, agreement scores can be calculated.

4. Curate annotations on WebAnno tool

5. [`unzip.py`](unzip.py) : Unzip latest annotations (from curation folder)

6. [`subtask2.py`](subtask2.py) : Check curators' work for basic errors, like missing spans, links, etc. If all okay, final annotations are extracted and consolidated.

### Generating competition datasets

7. http://localhost:8888/notebooks/61%20Challenges/2022_CASE_/Documentation/Subtask2/Get%20CNC_Version2.ipynb

8. http://localhost:8888/notebooks/61%20Challenges/2022_CASE_/Documentation/Subtask2/Load%20Dataset.ipynb

9. http://localhost:8888/notebooks/61%20Challenges/2022_CASE_/Documentation/Subtask2/cleaned/GroupData.ipynb

10. http://localhost:8888/notebooks/61%20Challenges/2022_CASE_/Documentation/Subtask2/Get%20Competition%20Datasets.ipynb
Copy from "D:\61 Challenges\2022_CASE_\Documentation\Subtask2\data_v2" to "data" here
