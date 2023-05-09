***The paper for this repo has been published: [Evaluating Variants of wav2vec 2.0 on Affective Vocal Burst Tasks.
](https://ieeexplore.ieee.org/document/10096552)***


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

> B. T. Atmaja and A. Sasou, “Evaluating Variants of wav2vec 2.0 on Affective Vocal Burst Tasks,” 
> in ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 
> Jun. 2023, pp. 1–5, doi: 10.1109/ICASSP49357.2023.10096552.


