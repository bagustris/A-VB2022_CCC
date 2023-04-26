
# A-VB Feature-based
Another version of https://github.com/bagustris/A-VB2022 using CCC loss function (please take a look that repository before trying code in this repository).
Feature extraction for 'w2v2-r-er' and 'w2v2-r-vad' are given there. Also install requirements from there (inside `feature_based` directory).

We provide a 'feature-based' approach for all four tasks to reproduce results in our ICASSP paper (see citation). 

## Installation

First, make sure you download the data and features, placing them within your working directory. For more info and instructions on how to access the competition data visit [competitions.hume.ai](http://www.competitions.hume.ai). 

We suggest creating a virtual environment, and installing the `requirements.txt`

```
conda create -n avb-2022-ccc python=3.8
conda activate avb-2022-ccc
pip install -r requirements.txt
```

## Example

After the data, labels, and features are downloaded to your working directory, when running `main.py` set to `-d ./` i.e., where `features/` and `labels/` are, and run: 

_A-VB High Baseline_

```
python main.py -d /data/A-VB/ -f w2v2-R-emo-vad -t type -e 100 -lr 0.0005 -bs 8 -p 10 --n_seeds 20
```

Follow the same procedure for each task altering `-t`. 


| Option         | Description                                  |
| -------------- | -------------------------------------------- |
| `-d`           | Set the path to working dir                  |
| `-f`           | Feature set e.g., `eGeMAPS`                  |
| `-t`           | Task ['high','two','culture','type']         |
| `-e`           | Number of Epochs (default: `20`)             |
| `-lr`          | Learning Rate  (default: `0.001`)            |
| `-bs`          | Batch Size  (default: `8`)                   |
| `-p`           | Early Stopping Patience (default: `5`)       |
| `--n_seeds`    | Number of Seeds to run for                   |
| `--verbose`    | Maximum verbosity, (default: quiet)          |


## Citation

```
[1] B. T. Atmaja and A. Sasou, “EVALUATING VARIANTS OF WAV2VEC 2.0 ON AFFECTIVE VOCAL BURST TASKS,” 
in ICASSP, IEEE International Conference on Acoustics, Speech and Signal Processing - Proceedings, 
June 2023, vol. 2023-June.
```

