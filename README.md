# Positional Attention for Neural Algorithmic Reasoning

This repository contains the code for the paper **Positional Attention: Out-of-Distribution Generalization and Expressivity for Neural Algorithmic Reasoning**. To run the experiments presented in the paper, first install the necessary dependencies by executing:

```bash
pip install -r requirements.txt
```

There are five scripts available for running the experiments: 
- The script `experiment_scale.py` generates the results for Figures 2 and 5 in the main paper, as well as for sections C.1 and C.2 in the Appendix.
- The script `experiment_sample_or_size` generates the results for Figures 3 and 4 in the main paper, as well as for sections C.4 and C.5 in the Appendix.
- The script `experiments_nlp` generates the results for the multimodal task described in section C.6 of the Appendix.
- The notebook `accuracy_plots.ipynb` generates the results for the accuracy measures in section C.7 of the Appendix.
- The script `experiment_model_variations.py` generates the results for the ablation study in Appendix G.

To execute any of these python scripts, specify the task using `--task` followed by the desired task (e.g., `min`, `sum`, `sort`, `median`, or `maxsub`). Additionally, provide the file path for the experiment parameters using `--params` (parameter files are located in the `/params` directory). Finally, ensure you specify an output directory with `--savepath`. For example:

```bash
python experiment_sample_or_size.py --params params/sample_params.json --task min --savepath ./experiment
```

In the `/results` directory, you will also find the output from all our experiments.




