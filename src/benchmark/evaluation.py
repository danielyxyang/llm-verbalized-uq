import json
import logging
import os
import random
from collections import defaultdict

import numpy as np
import scipy.special

from benchmark.datasets import DATASETS, load_dataset
from benchmark.models import MODELS
from benchmark.prompt_methods import (
    NO_ANSWER_TEXT,
    PROMPT_METHODS,
    load_prompt_method,
)
from utils_ext.math import safe_div

logger = logging.getLogger(__name__)

VALID_ANSWER = "valid_answer"
NO_ANSWER = "no_answer"
INVALID_ANSWER = "invalid_answer"

def load_responses(path, dataset_name, model_name, method_name, dataset_cache=None):
     # load responses
    path_responses = os.path.join(path, dataset_name, model_name, f"{method_name}.json")

    if not os.path.isfile(path_responses):
        return None
    with open(path_responses, "r") as f:
        responses = json.load(f)

    # load dataset
    dataset = load_dataset(dataset_name, dataset_cache=dataset_cache, verbose=False)
    # load prompt method
    prompt_method = load_prompt_method(method_name, verbose=False)

    # parse responses
    responses_grouped = {
        VALID_ANSWER: [],
        NO_ANSWER: [],
        INVALID_ANSWER: [],
    }
    for response in responses:
        # HACK
        if isinstance(next(iter(dataset.id2index)), int):
            response_id = int(response["id"])
        else:
            response_id = response["id"]

        # load prompt
        prompt = dataset[dataset.id2index[response_id]]
        # parse answer and confidence score from response
        answer, confidence = prompt_method.extract_answer(response["responses"])
        response["answer"] = answer
        response["confidence"] = confidence
        # determine correctness and response category
        if response["responses"][0].lower().startswith(NO_ANSWER_TEXT.lower()):
            response["is_correct"] = NO_ANSWER
            response_category = NO_ANSWER
        elif answer is None or confidence is None:
            response["is_correct"] = INVALID_ANSWER
            response_category = INVALID_ANSWER
        elif answer.lower() == NO_ANSWER_TEXT.lower():
            response["is_correct"] = NO_ANSWER
            response_category = NO_ANSWER
        else:
            response["is_correct"] = dataset.evaluate_answer(response["answer"], prompt["correct_answer"])
            if 0 <= response["is_correct"] <= 1 and 0 <= response["confidence"] <= 1:
                response_category = VALID_ANSWER
            else:
                response_category = INVALID_ANSWER
        # add prompt and response to corresponding response category
        responses_grouped[response_category].append((prompt, response))

    return responses_grouped

def load_responses_all(path, dataset_names=None, model_names=None, method_names=None, dataset_cache=None):
    # detect datasets, models and methods
    if dataset_names is None:
        dataset_names = list(sorted(e.name for e in os.scandir(path) if e.is_dir() and not e.name.startswith("_")))
    else:
        dataset_names = dataset_names.split(",")

    if model_names is None:
        model_names = list(sorted(e.name for e in os.scandir(os.path.join(path, dataset_names[0])) if e.is_dir()))
        model_names = [model_name for model_name in MODELS.keys() if model_name in model_names] # sort by dictionary
    else:
        model_names = model_names.split(",")

    if method_names is None:
        method_names = list(sorted(os.path.splitext(e.name)[0] for e in os.scandir(os.path.join(path, dataset_names[0], model_names[0])) if e.is_file() and e.name.endswith(".json")))
    else:
        method_names = method_names.split(",")

    # create dataset cache if needed
    if dataset_cache is None:
        dataset_cache = {}

    # load datasets
    responses_all = defaultdict(lambda: defaultdict(lambda: dict()))
    for dataset_name in dataset_names:
        for model_name in model_names:
            for method_name in method_names:
                responses = load_responses(path, dataset_name.split(":")[0], model_name.split(":")[0], method_name.split(":")[0], dataset_cache=dataset_cache)
                responses_all[dataset_name][model_name][method_name] = responses

    return responses_all

def detect_names_from_dict(responses_all, dataset_names=None, model_names=None, method_names=None):
    if dataset_names is None:
        # dataset_names = list(responses_all.keys())
        dataset_names = [dataset_name_tagged for dataset_name in DATASETS for dataset_name_tagged in responses_all if dataset_name_tagged.split(":")[0] == dataset_name]
    if model_names is None:
        # model_names = list(responses_all[dataset_names[0]].keys())
        model_names = [model_name_tagged for model_name in MODELS for model_name_tagged in responses_all[dataset_names[0]] if model_name_tagged.split(":")[0] == model_name]
    if method_names is None:
        # method_names = list(responses_all[dataset_names[0]][model_names[0]].keys())
        method_names = [method_name_tagged for method_name in PROMPT_METHODS for method_name_tagged in responses_all[dataset_names[0]][model_names[0]] if method_name_tagged.split(":")[0] == method_name]
    return dataset_names, model_names, method_names

def extract_predictions(responses_all, sample=None):
    random.seed(0)

    dataset_names, model_names, method_names = detect_names_from_dict(responses_all)
    y_true_all = defaultdict(lambda: defaultdict(dict))
    y_pred_all = defaultdict(lambda: defaultdict(dict))
    for dataset_name in dataset_names:
        for model_name in model_names:
            for method_name in method_names:
                responses = responses_all[dataset_name][model_name][method_name]
                if responses is None:
                    responses_valid = []
                else:
                    responses_valid = responses[VALID_ANSWER] if sample is None or len(responses[VALID_ANSWER]) == 0 else random.choices(responses[VALID_ANSWER], k=sample)

                y_true_all[dataset_name][model_name][method_name] = [response["is_correct"] for _, response in responses_valid]
                y_pred_all[dataset_name][model_name][method_name] = [response["confidence"] for _, response in responses_valid]

    return y_true_all, y_pred_all

def save_predictions(responses_all, path, sample=None):
    dataset_names, model_names, method_names = detect_names_from_dict(responses_all)
    y_true_all, y_pred_all = extract_predictions(responses_all, sample=sample)

    for dataset_name in dataset_names:
        for model_name in model_names:
            path_y_true = os.path.join(path, "y_true", dataset_name, model_name)
            path_y_pred = os.path.join(path, "y_pred", dataset_name, model_name)
            os.makedirs(path_y_true, exist_ok=True)
            os.makedirs(path_y_pred, exist_ok=True)
            for method_name in method_names:
                np.save(os.path.join(path_y_true, method_name), y_true_all[dataset_name][model_name][method_name])
                np.save(os.path.join(path_y_pred, method_name), y_pred_all[dataset_name][model_name][method_name])

def load_predictions(path):
    dataset_names = list(sorted(e.name for e in os.scandir(os.path.join(path, "y_true"))))
    model_names = list(sorted(e.name for e in os.scandir(os.path.join(path, "y_true", dataset_names[0]))))
    method_names = list(sorted(os.path.splitext(e.name)[0] for e in os.scandir(os.path.join(path, "y_true", dataset_names[0], model_names[0]))))

    y_true_all = defaultdict(lambda: defaultdict(lambda: dict()))
    y_pred_all = defaultdict(lambda: defaultdict(lambda: dict()))
    for dataset_name in dataset_names:
        for model_name in model_names:
            for method_name in method_names:
                y_true_all[dataset_name][model_name][method_name] = np.load(os.path.join(path, "y_true", dataset_name, model_name, f"{method_name}.npy")).tolist()
                y_pred_all[dataset_name][model_name][method_name] = np.load(os.path.join(path, "y_pred", dataset_name, model_name, f"{method_name}.npy")).tolist()

    return y_true_all, y_pred_all

def aggregate_responses(responses_all, dataset_names, model_names, method_names):
    if not isinstance(dataset_names, list):
        dataset_names = [dataset_names]
    if not isinstance(model_names, list):
        model_names = [model_names]
    if not isinstance(method_names, list):
        method_names = [method_names]
    # aggregate responses
    responses_agg = []
    for dataset_name in dataset_names:
        for model_name in model_names:
            for method_name in method_names:
                responses_agg += responses_all[dataset_name][model_name][method_name]
    return responses_agg

# reference: https://github.com/scikit-learn/scikit-learn/blob/8721245511de2f225ff5f9aa5f5fadce663cd4a3/sklearn/calibration.py#L927
def calibration_curve(y_true, y_pred, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_pred, bins[1:-1])

    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins)-1)
    bin_pred = np.bincount(binids, weights=y_pred, minlength=len(bins)-1)
    bin_count = np.bincount(binids, minlength=len(bins)-1)

    prob_true = safe_div(bin_true, bin_count, default=np.nan)
    prob_pred = safe_div(bin_pred, bin_count, default=np.nan)

    return prob_true, prob_pred, bins, bin_count

def empirical_distr(samples, n_bins):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(samples, bins[1:-1])
    bin_count = np.bincount(binids, minlength=len(bins)-1)
    distr = bin_count / np.sum(bin_count)
    return distr

def kl_div(p, q):
    return np.sum(scipy.special.rel_entr(p, q))
