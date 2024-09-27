# Positional Attention for Neural Algorithmic Reasoning

This repository contains the code for the paper **Positional Attention: Out-of-Distribution Generalization and Expressivity for Neural Algorithmic Reasoning**. To run the experiments presented in the paper, first install the necessary dependencies by executing:

```bash
pip install -r requirements.txt
```

There are two scripts available for running the experiments: `experiment_scale.py` and `experiment_sample_or_size.py`. The first script generates the results for Figures 2 and 5 in the main paper, as well as for sections C.1 and C.2 in the Appendix. The second script is used for the remaining figures.

To execute any of these experiments, specify the task using `--task` followed by the desired task (e.g., `min`, `sum`, `sort`, `median`, or `maxsub`). Additionally, provide the file path for the experiment parameters using `--params` (parameter files are located in the `/params` directory). Finally, ensure you specify an output directory with `--savepath`. For example:

```bash
python experiments.py --params params/sample_params.json --task min --savepath ./experiment
```

In the `/results` directory, you will also find the output from all our experiments.




