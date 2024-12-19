import logging
import re
import string
from collections import Counter

import numpy as np
from datasets import load_dataset as load_dataset_from_hf

logger = logging.getLogger(__name__)

def load_dataset(dataset_name: str, dataset_cache=None, verbose: bool = True):
    if dataset_cache is not None and dataset_name in dataset_cache:
        # load dataset from cache
        dataset = dataset_cache[dataset_name]
    else:
        if dataset_name in DATASETS:
            dataset_config = DATASETS[dataset_name]
            # load dataset
            dataset = load_dataset_from_hf(**dataset_config.load_args)
            # sample from dataset
            if dataset_config.n_samples is not None:
                dataset_indices = np.random.default_rng(seed=0).choice(len(dataset), dataset_config.n_samples)
                dataset = dataset.select(dataset_indices)
            # format samples in dataset
            dataset = dataset.map(dataset_config.format_sample, with_indices=True, load_from_cache_file=False)
            # patch dataset object with additional properties and methods
            dataset.id2index = {sample["id"]: i for i, sample in enumerate(dataset)}
            dataset.evaluate_answer = dataset_config.evaluate_answer

            # store dataset in cache
            if dataset_cache is not None:
                dataset_cache[dataset_name] = dataset
        else:
            raise ValueError(f"Dataset \"{dataset_name}\" is not supported.")

    if verbose:
        logger.info(f"Loaded dataset \"{dataset_name}\".\n{dataset}")
    return dataset

def format_sample_mc(id, description, question, choices_text, choices_is_correct, choices_letter=None, shuffle=False, seed=None):
    # shuffle choices
    if shuffle:
        choices_order = np.random.default_rng(seed=seed).permutation(len(choices_text))
        choices_text = [choices_text[i] for i in choices_order]
        choices_is_correct = [choices_is_correct[i] for i in choices_order]
    # create prompt components
    if choices_letter is None:
        choices_letter = [chr(ord("A") + i) for i in range(len(choices_text))]
    choices = "\n".join(f"{label}. {text}" for label, text in zip(choices_letter, choices_text))
    correct_answers_letter = [choices_letter[i] for i, label in enumerate(choices_is_correct) if label == 1]
    correct_answers_text = [choices_text[i] for i, label in enumerate(choices_is_correct) if label == 1]
    # create prompt
    return dict(
        id=id,
        description=description,
        content=f"Question: {question}\nChoices:\n{choices}",
        correct_answer=(",".join(correct_answers_letter), ",".join(correct_answers_text)),
    )

def normalize_answer_mc(s):
    return s.lower()

# reference: https://github.com/mandarjoshi90/triviaqa/blob/ca43b5820b107f3970cf4b7d67f7db7a98117b79/evaluation/triviaqa_evaluation.py
def normalize_answer_text(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)
    def lower(text):
        return text.lower()
    def replace_underscore(text):
        return text.replace('_', ' ')
    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()

def normalize_answer_list(s):
    return sorted(s.replace(" ", "").split(","))

ANSWER_PATTERN_MC_SINGLE_LETTER = re.compile(r"(?P<answer>[0-9A-Za-z])(\.(\s+.+)?)?$")
ANSWER_PATTERN_MC_SINGLE_TEXT = re.compile(r"(?P<answer>.+)$")
def evaluate_answer_mc_single(answer: str, correct_answer):
    correct_answer_letter, correct_answer_text = correct_answer
    # apply heuristics to detect a valid MC response
    match = ANSWER_PATTERN_MC_SINGLE_LETTER.match(answer.strip())
    if match:
        answer = match.group("answer")
        return int(normalize_answer_mc(answer) == normalize_answer_mc(correct_answer_letter))
    match = ANSWER_PATTERN_MC_SINGLE_TEXT.match(answer.strip())
    if match:
        answer = match.group("answer")
        return int(normalize_answer_text(answer) == normalize_answer_text(correct_answer_text))
    return -1

ANSWER_PATTERN_MC_MULTIPLE_LETTER = re.compile(r"^(?P<answer>[0-9A-Za-z](\,\s?[0-9A-Za-z])*)([\,\.](\s+.+)?)?$")
def evaluate_answer_mc_multiple(answer: str, correct_answer):
    def iou(a: list, b: list):
        a, b = set(a), set(b)
        return len(a & b) / len(a | b)

    correct_answer_letter, _ = correct_answer
    # apply heuristics to detect a valid MC response
    match = ANSWER_PATTERN_MC_MULTIPLE_LETTER.match(answer)
    if match:
        answer = match.group("answer")
        return iou(normalize_answer_list(answer), normalize_answer_list(correct_answer_letter))
    return -1


# DATASETS

class Dataset:
    load_args = dict()
    n_samples = None

    @staticmethod
    def format_sample(sample: dict, index: int):
        pass

    @staticmethod
    def evaluate_answer(answer: str, correct_answer: str):
        pass


class dataset_arc_c(Dataset):
    load_args = dict(path="allenai/ai2_arc", name="ARC-Challenge", split="validation")

    @staticmethod
    def format_sample(sample: dict, index: int):
        return format_sample_mc(
            id                 = sample["id"],
            description        = "The following multiple-choice question has only one correct answer. Provide only the option letter of the correct answer.",
            question           = sample["question"],
            choices_text       = sample["choices"]["text"],
            choices_is_correct = [label == sample["answerKey"] for label in sample["choices"]["label"]],
            choices_letter     = sample["choices"]["label"],
            shuffle            = False,
        )

    @staticmethod
    def evaluate_answer(answer: str, correct_answer: str):
        return evaluate_answer_mc_single(answer, correct_answer)


class dataset_arc_e(Dataset):
    load_args = dict(path="allenai/ai2_arc", name="ARC-Easy", split="validation")

    @staticmethod
    def format_sample(sample: dict, index: int):
        return format_sample_mc(
            id                 = sample["id"],
            description        = "The following multiple-choice question has only one correct answer. Provide only the option letter of the correct answer.",
            question           = sample["question"],
            choices_text       = sample["choices"]["text"],
            choices_is_correct = [label == sample["answerKey"] for label in sample["choices"]["label"]],
            choices_letter     = sample["choices"]["label"],
            shuffle            = False,
        )

    @staticmethod
    def evaluate_answer(answer: str, correct_answer: str):
        return evaluate_answer_mc_single(answer, correct_answer)


class dataset_civil_comments(Dataset):
    load_args = dict(path="google/civil_comments", split="validation")
    n_samples = 1500

    ANSWER_PATTERN = re.compile(r"(toxicity score: )?(?P<answer>\d*\.?\d+)", re.IGNORECASE)

    @staticmethod
    def format_sample(sample: dict, index: int):
        return dict(
            id             = index,
            description    = "Analyze the toxicity of the following comment and provide only a numeric toxicity score between 0.0 (not toxic) and 1.0 (very toxic).",
            content        = f"Comment: {sample["text"]}",
            correct_answer = sample["toxicity"],
        )

    @staticmethod
    def evaluate_answer(answer: str, correct_answer: str):
        # apply heuristics to detect a valid MC response
        match = dataset_civil_comments.ANSWER_PATTERN.match(answer)
        if match:
            answer = match.group("answer")
            return 1 - abs(float(answer) - correct_answer)
        return -1


class dataset_commonsense_qa(Dataset):
    load_args = dict(path="tau/commonsense_qa", split="validation")

    @staticmethod
    def format_sample(sample: dict, index: int):
        return format_sample_mc(
            id                 = sample["id"],
            description        = "The following multiple-choice question has only one correct answer. Provide only the option letter of the correct answer.",
            question           = sample["question"],
            choices_text       = sample["choices"]["text"],
            choices_is_correct = [label == sample["answerKey"] for label in sample["choices"]["label"]],
            choices_letter     = sample["choices"]["label"],
            shuffle            = False,
        )

    @staticmethod
    def evaluate_answer(answer: str, correct_answer: str):
        return evaluate_answer_mc_single(answer, correct_answer)


class dataset_imdb(Dataset):
    load_args = dict(path="stanfordnlp/imdb", split="test")
    n_samples = 1500

    ANSWER_PATTERN = re.compile(r".*(?P<answer>positive|negative)", re.IGNORECASE)

    @staticmethod
    def format_sample(sample: dict, index: int):
        return dict(
            id             = index,
            description    = "Analyze the sentiment of the following movie review and classify it as either `positive` or `negative`.",
            content        = f"Review: {sample["text"]}",
            correct_answer = "positive" if sample["label"] == 1 else "negative",
        )

    @staticmethod
    def evaluate_answer(answer: str, correct_answer: str):
        # apply heuristics to detect a valid MC response
        match = dataset_imdb.ANSWER_PATTERN.match(answer)
        if match:
            answer = match.group("answer")
            return int(normalize_answer_mc(answer) == normalize_answer_mc(correct_answer))
        return -1


class dataset_logi_qa(Dataset):
    load_args = dict(path="lucasmccabe/logiqa", split="validation")

    @staticmethod
    def format_sample(sample: dict, index: int):
        return format_sample_mc(
            id                 = index,
            description        = "The following multiple-choice question has only one correct answer. Provide only the option letter of the correct answer.",
            question           = f"{sample["context"]} {sample["query"]}",
            choices_text       = sample["options"],
            choices_is_correct = [i == sample["correct_option"] for i in range(len(sample["options"]))],
            shuffle            = False,
        )

    @staticmethod
    def evaluate_answer(answer: str, correct_answer: str):
        return evaluate_answer_mc_single(answer, correct_answer)


class dataset_med_qa(Dataset):
    load_args = dict(path="bigbio/med_qa", name="med_qa_en_source", split="validation", trust_remote_code=True)

    @staticmethod
    def format_sample(sample: dict, index: int):
        return format_sample_mc(
            id                 = index,
            description        = "The following multiple-choice question has only one correct answer. Provide only the option letter of the correct answer.",
            question           = sample["question"],
            choices_text       = [s["value"] for s in sample["options"]],
            choices_is_correct = [s["key"] == sample["answer_idx"] for s in sample["options"]],
            choices_letter     = [s["key"] for s in sample["options"]],
            shuffle            = False,
        )

    @staticmethod
    def evaluate_answer(answer: str, correct_answer: str):
        return evaluate_answer_mc_single(answer, correct_answer)


class dataset_mmlu(Dataset):
    load_args = dict(path="cais/mmlu", name="all", split="validation")

    @staticmethod
    def format_sample(sample: dict, index: int):
        return format_sample_mc(
            id                 = index,
            description        = "The following multiple-choice question has only one correct answer. Provide only the option letter of the correct answer.",
            question           = sample["question"],
            choices_text       = sample["choices"],
            choices_is_correct = [i == sample["answer"] for i in range(len(sample["choices"]))],
            shuffle            = False,
        )

    @staticmethod
    def evaluate_answer(answer: str, correct_answer: str):
        return evaluate_answer_mc_single(answer, correct_answer)


class dataset_sciq(Dataset):
    load_args = dict(path="allenai/sciq", split="validation")

    @staticmethod
    def format_sample(sample: dict, index: int):
        return format_sample_mc(
            id                 = index,
            description        = "The following multiple-choice question has only one correct answer. Provide only the option letter of the correct answer.",
            question           = sample["question"],
            choices_text       = [sample["correct_answer"], sample["distractor1"], sample["distractor2"], sample["distractor3"]],
            choices_is_correct = [1, 0, 0, 0],
            shuffle            = True,
            seed               = index,
        )

    @staticmethod
    def evaluate_answer(answer: str, correct_answer: str):
        return evaluate_answer_mc_single(answer, correct_answer)


class dataset_social_i_qa(Dataset):
    load_args = dict(path="allenai/social_i_qa", split="validation", trust_remote_code=True)

    @staticmethod
    def format_sample(sample: dict, index: int):
        return format_sample_mc(
            id                 = index,
            description        = "The following multiple-choice question has only one correct answer. Provide only the option letter of the correct answer.",
            question           = f"{sample["context"]} {sample["question"]}",
            choices_text       = [sample["answerA"], sample["answerB"], sample["answerC"]],
            choices_is_correct = [i+1 == int(sample["label"]) for i in range(3)],
            shuffle            = False,
        )

    @staticmethod
    def evaluate_answer(answer: str, correct_answer: str):
        return evaluate_answer_mc_single(answer, correct_answer)


class dataset_trivia_qa(Dataset):
    load_args = dict(path="mandarjoshi/trivia_qa", name="unfiltered.nocontext", split="validation")
    n_samples = 1500

    @staticmethod
    def format_sample(sample: dict, index: int):
        return dict(
            id             = sample["question_id"],
            description    = "Provide only a short answer in the form of keywords to the following question.",
            content        = f"Question: {sample["question"]}",
            correct_answer = sample["answer"]["normalized_aliases"],
        )

    @staticmethod
    def evaluate_answer(answer: str, correct_answers: list):

        # def em_score(answer, correct_answer):
        #     return int(answer == correct_answer)

        def f1_score(answer, correct_answer):
            answer_words = answer.split()
            correct_answer_words = correct_answer.split()
            n_common_words = (Counter(answer_words) & Counter(correct_answer_words)).total()
            if n_common_words == 0:
                return 0
            else:
                precision = n_common_words / len(answer_words)
                recall = n_common_words / len(correct_answer_words)
                return (2 * precision * recall) / (precision + recall)

        return max(f1_score(normalize_answer_text(answer), normalize_answer_text(correct_answer)) for correct_answer in correct_answers)


class dataset_truthful_qa_mc1(Dataset):
    load_args = dict(path="truthful_qa", name="multiple_choice", split="validation")

    @staticmethod
    def format_sample(sample: dict, index: int):
        return format_sample_mc(
            id                 = index,
            description        = "The following multiple-choice question has only one correct answer. Provide only the option letter of the correct answer.",
            question           = sample["question"],
            choices_text       = sample["mc1_targets"]["choices"],
            choices_is_correct = sample["mc1_targets"]["labels"],
            shuffle            = True,
            seed               = index,
        )

    @staticmethod
    def evaluate_answer(answer: str, correct_answer: str):
        return evaluate_answer_mc_single(answer, correct_answer)


class dataset_truthful_qa_mc2(Dataset):
    load_args = dict(path="truthful_qa", name="multiple_choice", split="validation")

    @staticmethod
    def format_sample(sample: dict, index: int):
        return format_sample_mc(
            id                 = index,
            description        = "The following multiple-choice question has multiple correct answers. Provide only a comma-separated list of the option letters of the correct answers.",
            question           = sample["question"],
            choices_text       = sample["mc2_targets"]["choices"],
            choices_is_correct = sample["mc2_targets"]["labels"],
            shuffle            = True,
            seed               = index,
        )

    @staticmethod
    def evaluate_answer(answer: str, correct_answer: str):
        return evaluate_answer_mc_multiple(answer, correct_answer)


DATASETS = {
    "arc-c":           dataset_arc_c,
    "arc-e":           dataset_arc_e,
    "civil_comments":  dataset_civil_comments,
    "commonsense_qa":  dataset_commonsense_qa,
    "imdb":            dataset_imdb,
    "logi_qa":         dataset_logi_qa,
    "med_qa":          dataset_med_qa,
    "mmlu":            dataset_mmlu,
    "sciq":            dataset_sciq,
    "social_i_qa":     dataset_social_i_qa,
    "trivia_qa":       dataset_trivia_qa,
    "truthful_qa-mc1": dataset_truthful_qa_mc1,
    "truthful_qa-mc2": dataset_truthful_qa_mc2,
}
