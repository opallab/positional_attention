# Positional attention for Neural Algorithmic Reasoning



This repository contains the code for the paper Positional Attention: Out-of-Distribution Generalization and Expressivity for Neural Algorithmic Reasoning.
To run the experiments present in the paper, specify the type of experiment (e.g. `experiment_sample_or_size.py` or `experiment_scale.py`), along with the task (e.g.  `min`, `sum`, `sort`, `median`, or `maxsub`), parameter file and output directory. For example:
```
python experiments.py --params params/sample_params.json --task min --savepath ./experiment
```

