# On Verbalized Confidence Scores for LLMs

This repository is the official implementation of [On Verbalized Confidence Scores for LLMs](https://arxiv.org/abs/2412.14737).

## Abstract

The rise of large language models (LLMs) and their tight integration into our daily life make it essential to dedicate efforts towards their trustworthiness.
Uncertainty quantification for LLMs can establish more human trust into their responses, but also allows LLM agents to make more informed decisions based on each other's uncertainty.
To estimate the uncertainty in a response, internal token logits, task-specific proxy models, or sampling of multiple responses are commonly used.
This work focuses on asking the LLM itself to verbalize its uncertainty with a confidence score as part of its output tokens, which is a promising way for prompt- and model-agnostic uncertainty quantification with low overhead.
Using an extensive benchmark, we assess the reliability of verbalized confidence scores with respect to different datasets, models, and prompt methods.
Our results reveal that the reliability of these scores strongly depends on how the model is asked, but also that it is possible to extract well-calibrated confidence scores with certain prompt methods.
We argue that verbalized confidence scores can become a simple but effective and versatile uncertainty quantification method in the future.

<p align="center">
    <img src="https://github.com/user-attachments/assets/dc20d83d-82cc-46ab-83e4-ae349c6895b7" width="75%"><br>
    <em>Figure: Calibration diagrams for prompt method `basic` (left) and `combo` (right). The color intensity of each bar is proportional to the bin size on a log scale.</em>
</p>

## Setup

After cloning this repository, update its submodules with
```bash
git submodule update --init
```

We use `Python 3.12.4` with the following dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run evaluation

We provide all raw responses and predictions from our evaluated LLMs in `results.zip`, which can be downloaded [here](https://doi.org/10.5281/zenodo.14531920) should be extracted into the `results` folder. The `results.zip` consists of the following directories:
* `responses`: contains all raw responses in JSON format
* `predictions`: contains all predicted confidence scores of valid responses and the corresponding ground-truth correctness labels in numpy format
* `predictions_sampled`: contains 1000 random samples (with replacement) from `predictions` used for evaluation in our paper

To evaluate these responses and predictions, use the notebook `evaluation.ipynb`.

The plots in our paper are generated with the notebook `paper_plots.ipynb`.

### Run inference

To run inference interactively, use the notebook `inference.ipynb`.

To run inference on the command-line, use
```bash
python scripts/run_inference.py --datasets "arc-c ..." --model "gemma1.1-2b-it" --methods "basic_1s ..."
```
or
```bash
python scripts/run_inference_batch_api.py --datasets "arc-c ..." --model "gpt4o" --methods "basic_1s ..."
```
depending on the used model. The command-line arguments `--datasets` and `--methods` accept a space-separated list of dataset or prompt method names, respectively.

A list of available dataset, model, and prompt method names can be found at the end of each of the files `src/benchmark/(datasets|models|prompt_methods).py`.

## Citation

If you find this work useful, please consider citing this paper:
```bibtex
@article{yang2024verbalized,
    title   = {On Verbalized Confidence Scores for LLMs},
    author  = {Yang, Daniel and Tsai, Yao-Hung Hubert and Yamada, Makoto},
    year    = {2024},
    journal = {arXiv preprint arXiv:2412.14737},
}
```

## License

All content in this repository is licensed under the MIT license. See [LICENSE](LICENSE) for details.
