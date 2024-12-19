import logging
import re

logger = logging.getLogger(__name__)

NO_ANSWER_TEXT = "NO ANSWER"
PROMPT_NO_ANSWER = f"If you cannot provide an answer, answer with `{NO_ANSWER_TEXT}`."

PATTERN_SEP = r"\n(.*\n)*?"
PATTERN_ANSWER = r".+"
PATTERN_FLOAT = r"\d*\.?\d+"


def load_prompt_method(method_name: str, verbose: bool = True):
    if method_name in PROMPT_METHODS:
        prompt_method = PROMPT_METHODS[method_name]
    else:
        raise ValueError(f"Prompt method \"{method_name}\" is not supported.")

    if verbose:
        # extract example prompts from prompt method
        prompts = []
        def extract_sample_prompt(prompt, **args):
            prompts.append(prompt)
            return "<ANSWER>", None
        prompt_method.generate_response(extract_sample_prompt, {"id": "<SAMPLE ID>", "description": "<TASK DESCRIPTION>", "content": "<TASK CONTENT>"})

        msg = f"Loaded prompt method \"{method_name}\"."
        for i, prompt in enumerate(prompts):
            msg += f"\n=== START OF PROMPT {i+1} ==="
            msg += f"\n{prompt["description"]}\n----------\n{prompt["content"]}"
            msg += f"\n=== END OF PROMPT {i+1} ==="
        logger.info(msg)

    return prompt_method

def extract_from_response(response: str, patterns: list, names: str | list):
    # remove formatting
    response = response.replace("**", "")
    response = response.replace("__", "")
    response = response.replace("```", "")
    # try each pattern
    for pattern in patterns:
        match = pattern.search(response)
        if match is not None:
            if isinstance(names, str):
                return match.group(names)
            else:
                return (match.group(name) for name in names)
    if isinstance(names, str):
        return None
    else:
        return (None for _ in names)

def normalize_confidence(confidence: str, normalize_fn: callable = float):
    if confidence is not None:
        confidence = normalize_fn(confidence)
    return confidence

# PROMPT METHODS

class PromptMethod:
    def generate_response(self, model, prompt: dict, **args):
        pass

    def extract_answer(self, responses: list):
        pass


class prompt_method_basic_1s(PromptMethod):
    response_patterns = [
        re.compile(fr"Answer: (?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}Confidence: (?P<confidence>{PATTERN_FLOAT})%?"),
        re.compile(fr"(?P<answer>{PATTERN_ANSWER})\n+(Confidence: )?(?P<confidence>{PATTERN_FLOAT})%?"),
    ]

    def generate_response(self, model, prompt: dict, **args):
        response, statistics = model({
            "id": prompt["id"],
            "description": f"{prompt["description"]} After your answer, provide a confidence score in percentage which measures how confident you are in your answer. Use the following format to respond:\n```\nAnswer: [Write your answer here.]\nConfidence: [Write your confidence score here.]\n```\n{PROMPT_NO_ANSWER}",
            "content": prompt["content"],
        }, **args)
        return [response], [statistics]

    def extract_answer(self, responses: list):
        answer, confidence = extract_from_response(responses[0], self.response_patterns, ("answer", "confidence"))
        confidence = normalize_confidence(confidence, lambda c: float(c) / 100)
        return answer, confidence


class prompt_method_basic_1s_scorefloat(PromptMethod):
    response_patterns = [
        re.compile(fr"Answer: (?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}Confidence: (?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(?P<answer>{PATTERN_ANSWER})\n+(Confidence: )?(?P<confidence>{PATTERN_FLOAT})"),
    ]

    def generate_response(self, model, prompt: dict, **args):
        response, statistics = model({
            "id": prompt["id"],
            "description": f"{prompt["description"]} After your answer, provide a confidence score between 0.0 and 1.0 which measures how confident you are in your answer. Use the following format to respond:\n```\nAnswer: [Write your answer here.]\nConfidence: [Write your confidence score here.]\n```\n{PROMPT_NO_ANSWER}",
            "content": prompt["content"],
        }, **args)
        return [response], [statistics]

    def extract_answer(self, responses: list):
        answer, confidence = extract_from_response(responses[0], self.response_patterns, ("answer", "confidence"))
        confidence = normalize_confidence(confidence)
        return answer, confidence


class prompt_method_basic_1s_scoreletter(PromptMethod):
    confidence_scores = {
        "A": 0.9,
        "B": 0.7,
        "C": 0.5,
        "D": 0.3,
        "E": 0.1,
    }
    response_patterns = [
        re.compile(fr"Answer: (?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}Confidence: (?P<confidence>[A-E])"),
        re.compile(fr"(?P<answer>{PATTERN_ANSWER})\n+(Confidence: )?(?P<confidence>[A-E])"),
    ]

    def generate_response(self, model, prompt: dict, **args):
        response, statistics = model({
            "id": prompt["id"],
            "description": f"{prompt["description"]} After your answer, provide a confidence score between A (very high confidence) and E (very low confidence) which measures how confident you are in your answer. Use the following format to respond:\n```\nAnswer: [Write your answer here.]\nConfidence: [Write your confidence score here.]\n```\n{PROMPT_NO_ANSWER}",
            "content": prompt["content"],
        }, **args)
        return [response], [statistics]

    def extract_answer(self, responses: list):
        answer, confidence = extract_from_response(responses[0], self.response_patterns, ("answer", "confidence"))
        confidence = normalize_confidence(confidence, lambda c: self.confidence_scores[c])
        return answer, confidence


class prompt_method_basic_1s_scoretext(PromptMethod):
    confidence_scores = {
        "very high": 0.9,
        "high": 0.7,
        "medium": 0.5,
        "low": 0.3,
        "very low": 0.1,
    }
    response_patterns = [
        re.compile(fr"Answer: (?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}Confidence: (?P<confidence>{"|".join(confidence_scores)})", re.IGNORECASE),
        re.compile(fr"(?P<answer>{PATTERN_ANSWER})\n+(Confidence: )?(?P<confidence>{"|".join(confidence_scores)})", re.IGNORECASE),
    ]

    def generate_response(self, model, prompt: dict, **args):
        response, statistics = model({
            "id": prompt["id"],
            "description": f"{prompt["description"]} After your answer, provide one of the following confidence scores which measures how confident you are in your answer: {", ".join(self.confidence_scores)}. Use the following format to respond:\n```\nAnswer: [Write your answer here.]\nConfidence: [Write your confidence score here.]\n```\n{PROMPT_NO_ANSWER}",
            "content": prompt["content"],
        }, **args)
        return [response], [statistics]

    def extract_answer(self, responses: list):
        answer, confidence = extract_from_response(responses[0], self.response_patterns, ("answer", "confidence"))
        confidence = normalize_confidence(confidence, lambda c: self.confidence_scores[c.lower()])
        return answer, confidence


class prompt_method_basic_1s_probscore(PromptMethod):
    response_patterns = [
        re.compile(fr"Answer: (?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}Probability: (?P<confidence>{PATTERN_FLOAT})%?"),
        re.compile(fr"(?P<answer>{PATTERN_ANSWER})\n+(Probability: )?(?P<confidence>{PATTERN_FLOAT})%?"),
    ]

    def generate_response(self, model, prompt: dict, **args):
        response, statistics = model({
            "id": prompt["id"],
            "description": f"{prompt["description"]} After your answer, provide the probability between 0.0 and 1.0 that your answer is correct for the given task. Use the following format to respond:\n```\nAnswer: [Write your answer here.]\nProbability: [Write your probability here.]\n```\n{PROMPT_NO_ANSWER}",
            "content": prompt["content"],
        }, **args)
        return [response], [statistics]

    def extract_answer(self, responses: list):
        answer, confidence = extract_from_response(responses[0], self.response_patterns, ("answer", "confidence"))
        confidence = normalize_confidence(confidence)
        return answer, confidence


class prompt_method_basic_1s_probscore_inline(PromptMethod):
    response_patterns = [
        re.compile(fr"Answer: (?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}Probability: (?P<confidence>{PATTERN_FLOAT})%?"),
        re.compile(fr"(?P<answer>{PATTERN_ANSWER})\n+(Probability: )?(?P<confidence>{PATTERN_FLOAT})%?"),
    ]

    def generate_response(self, model, prompt: dict, **args):
        response, statistics = model({
            "id": prompt["id"],
            "description": f"{prompt["description"]} After your answer, provide the probability between 0.0 and 1.0 that your answer is correct for the given task. Use the following format to respond:\n```\nAnswer: [Write your answer here. No extra comments, only the answer.]\nProbability: [Write your probability between 0.0 and 1.0 that your answer is correct here. No extra comments, only the probability.]\n```\n{PROMPT_NO_ANSWER}",
            "content": prompt["content"],
        }, **args)
        return [response], [statistics]

    def extract_answer(self, responses: list):
        answer, confidence = extract_from_response(responses[0], self.response_patterns, ("answer", "confidence"))
        confidence = normalize_confidence(confidence)
        return answer, confidence


class prompt_method_basic_1s_1shot(PromptMethod):
    # commonsense_qa/train: 23505889b94e880c3e89cff4ba119860
    PROMPT_EXAMPLE_ONE_SHOT_SCORES = [0.47]
    PROMPT_EXAMPLE_ONE_SHOT = """Here is an example:

Question: The fox walked from the city into the forest, what was it looking for?
Choices:
A. pretty flowers.
B. hen house
C. natural habitat
D. storybook
E. dense forest
Answer: A
Confidence: 47%"""

    response_patterns = [
        re.compile(fr"Answer: (?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}Confidence: (?P<confidence>{PATTERN_FLOAT})%?"),
        re.compile(fr"(?P<answer>{PATTERN_ANSWER})\n+(Confidence: )?(?P<confidence>{PATTERN_FLOAT})%?"),
    ]

    def generate_response(self, model, prompt: dict, **args):
        response, statistics = model({
            "id": prompt["id"],
            "description": f"{prompt["description"]} After your answer, provide a confidence score in percentage which measures how confident you are in your answer. Use the following format to respond:\n```\nAnswer: [Write your answer here.]\nConfidence: [Write your confidence score here.]\n```\n{PROMPT_NO_ANSWER} {self.PROMPT_EXAMPLE_ONE_SHOT}",
            "content": prompt["content"],
        }, **args)
        return [response], [statistics]

    def extract_answer(self, responses: list):
        answer, confidence = extract_from_response(responses[0], self.response_patterns, ("answer", "confidence"))
        confidence = normalize_confidence(confidence, lambda c: float(c) / 100)
        # filter out confidence scores used in few-shot examples
        if confidence in self.PROMPT_EXAMPLE_ONE_SHOT_SCORES:
            confidence = None
        return answer, confidence


class prompt_method_basic_1s_5shot(PromptMethod):
    # commonsense_qa/train: 23505889b94e880c3e89cff4ba119860
    # trivia_qa/unfiltered/train: tc_23
    # mmlu/elementary_mathematics/dev: 0
    # mmlu/business_ethics/dev: 0
    # arc/challenge/train: Mercury_7207498
    PROMPT_EXAMPLE_FIVE_SHOT_SCORES = [0.47, 0.89, 0.77, 0.24, 0.08]
    PROMPT_EXAMPLE_FIVE_SHOT = """Here are five examples:

Question: The fox walked from the city into the forest, what was it looking for?
Choices:
A. pretty flowers.
B. hen house
C. natural habitat
D. storybook
E. dense forest
Answer: A
Confidence: 47%

Question: Which country is Europe's largest silk producer?
Answer: Environment of Italy
Confidence: 89%

Question: The population of the city where Michelle was born is 145,826. What is the value of the 5 in the number 145,826?
Choices:
A. 5 thousands
B. 5 hundreds
C. 5 tens
D. 5 ones
Answer: A
Confidence: 77%

Question: Beyond the business case for engaging in CSR there are a number of moral arguments relating to: negative _______, the _______that corporations possess and the ________ of business and society.
Choices:
A. Externalities, Power, Independence
B. Publicity, Insubstantial resources, Mutual dependence
C. Publicity, Power, Independence
D. Externalities, Power, Mutual dependence
Answer: B
Confidence: 24%

Question: The Moon lacks weather and climate changes like those on Earth. What causes the lack of weather on the Moon?
Answer: the lack of magnetic poles
Confidence: 8%"""

    response_patterns = [
        re.compile(fr"Answer: (?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}Confidence: (?P<confidence>{PATTERN_FLOAT})%?"),
        re.compile(fr"(?P<answer>{PATTERN_ANSWER})\n+(Confidence: )?(?P<confidence>{PATTERN_FLOAT})%?"),
    ]

    def generate_response(self, model, prompt: dict, **args):
        response, statistics = model({
            "id": prompt["id"],
            "description": f"{prompt["description"]} After your answer, provide a confidence score in percentage which measures how confident you are in your answer. Use the following format to respond:\n```\nAnswer: [Write your answer here.]\nConfidence: [Write your confidence score here.]\n```\n{PROMPT_NO_ANSWER} {self.PROMPT_EXAMPLE_FIVE_SHOT}",
            "content": prompt["content"],
        }, **args)
        return [response], [statistics]

    def extract_answer(self, responses: list):
        answer, confidence = extract_from_response(responses[0], self.response_patterns, ("answer", "confidence"))
        confidence = normalize_confidence(confidence, lambda c: float(c) / 100)
        # filter out confidence scores used in few-shot examples
        if confidence in self.PROMPT_EXAMPLE_FIVE_SHOT_SCORES:
            confidence = None
        return answer, confidence


class prompt_method_basic_2s(PromptMethod):
    response_patterns1 = [
        re.compile(fr"(Answer: )?(?P<answer>{PATTERN_ANSWER})"),
    ]
    response_patterns2 = [
        re.compile(fr"(?P<confidence>{PATTERN_FLOAT})%?"),
    ]

    def generate_response(self, model, prompt: dict, **args):
        response1, statistics1 = model({
            "description": prompt["description"],
            "content": prompt["content"],
        }, **args)
        response2, statistics2 = model({
            "id": prompt["id"],
            "description": f"Provide a confidence score in percentage which measures how confident you are that the described task is solved by the given possible answer.",
            "content": f"TASK: {prompt["description"]}\n{prompt["content"]}\n\nPOSSIBLE ANSWER: {response1}",
        }, **args)
        return [response1, response2], [statistics1, statistics2]

    def extract_answer(self, responses: list):
        answer = extract_from_response(responses[0], self.response_patterns1, "answer")
        confidence = extract_from_response(responses[1], self.response_patterns2, "confidence")
        confidence = normalize_confidence(confidence, lambda c: float(c) / 100)
        return answer, confidence


class prompt_method_advanced_1s(PromptMethod):
    response_patterns = [
        re.compile(fr"Answer: (?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}Confidence: (?P<confidence>{PATTERN_FLOAT})%?"),
        re.compile(fr"(?P<answer>{PATTERN_ANSWER})\n+(Confidence: )?(?P<confidence>{PATTERN_FLOAT})%?"),
    ]

    def generate_response(self, model, prompt: dict, **args):
        response, statistics = model({
            "id": prompt["id"],
            "description": f"{prompt["description"]} After your answer, provide a confidence score in percentage. This score should quantify how confident you are in the correctness or plausibility of your answer for the given task. Take your uncertainty in the prompt, the task difficulty, your knowledge availability and other sources of uncertainty into account. Ideally, the score should correspond to the empirical accuracy of your answer. Use the following format to respond:\n```\nAnswer: [Write your answer here.]\nConfidence: [Write your confidence score here.]\n```\n{PROMPT_NO_ANSWER}",
            "content": prompt["content"],
        }, **args)
        return [response], [statistics]

    def extract_answer(self, responses: list):
        answer, confidence = extract_from_response(responses[0], self.response_patterns, ("answer", "confidence"))
        confidence = normalize_confidence(confidence, lambda c: float(c) / 100)
        return answer, confidence


class prompt_method_advanced_1s_probscore(PromptMethod):
    response_patterns = [
        re.compile(fr"Answer: (?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}Probability: (?P<confidence>{PATTERN_FLOAT})%?"),
        re.compile(fr"(?P<answer>{PATTERN_ANSWER})\n+(Probability: )?(?P<confidence>{PATTERN_FLOAT})%?"),
    ]

    def generate_response(self, model, prompt: dict, **args):
        response, statistics = model({
            "id": prompt["id"],
            "description": f"{prompt["description"]} After your answer, provide the probability between 0.0 and 1.0 that your answer is correct or plausible for the given task. Take your uncertainty in the prompt, the task difficulty, your knowledge availability and other sources of uncertainty into account. Use the following format to respond:\n```\nAnswer: [Write your answer here.]\nProbability: [Write your probability here.]\n```\n{PROMPT_NO_ANSWER}",
            "content": prompt["content"],
        }, **args)
        return [response], [statistics]

    def extract_answer(self, responses: list):
        answer, confidence = extract_from_response(responses[0], self.response_patterns, ("answer", "confidence"))
        confidence = normalize_confidence(confidence)
        return answer, confidence


class prompt_method_advanced_1s_tian2023just(PromptMethod):
    response_patterns = [
        re.compile(fr"Answer: (?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}Confidence: (?P<confidence>{PATTERN_FLOAT})%?"),
        re.compile(fr"(?P<answer>{PATTERN_ANSWER})\n+(Confidence: )?(?P<confidence>{PATTERN_FLOAT})%?"),
    ]

    def generate_response(self, model, prompt: dict, **args):
        response, statistics = model({
            "id": prompt["id"],
            "description": f"{prompt["description"]} After your answer, provide a confidence score between 0.0 and 1.0 which corresponds to the probability that your answer is correct for the given task. Take your uncertainty in the prompt, the task difficulty, your knowledge availability and other sources of uncertainty into account. Use the following format to respond:\n```\nAnswer: [Write your answer here.]\nConfidence: [Write your confidence score here.]\n```\n{PROMPT_NO_ANSWER}",
            "content": prompt["content"],
        }, **args)
        return [response], [statistics]

    def extract_answer(self, responses: list):
        answer, confidence = extract_from_response(responses[0], self.response_patterns, ("answer", "confidence"))
        confidence = normalize_confidence(confidence)
        return answer, confidence

class prompt_method_combo_1s(PromptMethod):
    # commonsense_qa/train: 23505889b94e880c3e89cff4ba119860
    # trivia_qa/unfiltered/train: tc_23
    # mmlu/elementary_mathematics/dev: 0
    # mmlu/business_ethics/dev: 0
    # arc/challenge/train: Mercury_7207498
    PROMPT_EXAMPLE_FIVE_SHOT_SCORES = [0.47, 0.89, 0.77, 0.24, 0.08]
    PROMPT_EXAMPLE_FIVE_SHOT = """Here are five examples:

Question: The fox walked from the city into the forest, what was it looking for?
Choices:
A. pretty flowers.
B. hen house
C. natural habitat
D. storybook
E. dense forest
Guess: A
Probability: 0.47

Question: Which country is Europe's largest silk producer?
Guess: Environment of Italy
Probability: 0.89

Question: The population of the city where Michelle was born is 145,826. What is the value of the 5 in the number 145,826?
Choices:
A. 5 thousands
B. 5 hundreds
C. 5 tens
D. 5 ones
Guess: A
Probability: 0.77

Question: Beyond the business case for engaging in CSR there are a number of moral arguments relating to: negative _______, the _______that corporations possess and the ________ of business and society.
Choices:
A. Externalities, Power, Independence
B. Publicity, Insubstantial resources, Mutual dependence
C. Publicity, Power, Independence
D. Externalities, Power, Mutual dependence
Guess: B
Probability: 0.24

Question: The Moon lacks weather and climate changes like those on Earth. What causes the lack of weather on the Moon?
Guess: the lack of magnetic poles
Probability: 0.08"""

    response_patterns = [
        re.compile(fr"(Guess: |Answer: )(?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}Probability: (?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(Guess: |Answer: )?(?P<answer>{PATTERN_ANSWER})\n+(Probability: )?(?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(Guess: |Answer: )?(?P<answer>{PATTERN_ANSWER})(, |. |; | - | \| )(Probability: )?(?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(Guess: |Answer: )?(?P<answer>{PATTERN_ANSWER}) \((Probability: )?(?P<confidence>{PATTERN_FLOAT})\)"),
    ]

    def generate_response(self, model, prompt: dict, **args):
        response, statistics = model({
            "id": prompt["id"],
            "description": f"{prompt["description"]} Provide your best guess and the probability between 0.0 and 1.0 that your best guess is correct or plausible for the given task. Take your uncertainty in the prompt, the task difficulty, your knowledge availability and other sources of uncertainty into account. Use the following format to respond:\n```\nGuess: [most likely guess, as short as possible, only the guess]\nProbability: [probability between 0.0 and 1.0 that your guess is correct, without comments, only the probability]\n```\n{PROMPT_NO_ANSWER} {self.PROMPT_EXAMPLE_FIVE_SHOT}",
            "content": prompt["content"],
        }, **args)
        return [response], [statistics]

    def extract_answer(self, responses: list):
        answer, confidence = extract_from_response(responses[0], self.response_patterns, ("answer", "confidence"))
        confidence = normalize_confidence(confidence)
        # filter out confidence scores used in few-shot examples
        if confidence in self.PROMPT_EXAMPLE_FIVE_SHOT_SCORES:
            confidence = None
        return answer, confidence

class prompt_method_combo_1s_v2(PromptMethod):
    # commonsense_qa/train: 23505889b94e880c3e89cff4ba119860
    # trivia_qa/unfiltered/train: tc_23
    # mmlu/elementary_mathematics/dev: 0
    # mmlu/business_ethics/dev: 0
    # arc/challenge/train: Mercury_7207498
    PROMPT_EXAMPLE_FIVE_SHOT_SCORES = [0.47, 0.89, 0.77, 0.24, 0.08]
    PROMPT_EXAMPLE_FIVE_SHOT = """Here are five examples:

Question: The fox walked from the city into the forest, what was it looking for?
Choices:
A. pretty flowers.
B. hen house
C. natural habitat
D. storybook
E. dense forest
Guess: A
Probability: 0.47

Question: Which country is Europe's largest silk producer?
Guess: Environment of Italy
Probability: 0.89

Question: The population of the city where Michelle was born is 145,826. What is the value of the 5 in the number 145,826?
Choices:
A. 5 thousands
B. 5 hundreds
C. 5 tens
D. 5 ones
Guess: A
Probability: 0.77

Question: Beyond the business case for engaging in CSR there are a number of moral arguments relating to: negative _______, the _______that corporations possess and the ________ of business and society.
Choices:
A. Externalities, Power, Independence
B. Publicity, Insubstantial resources, Mutual dependence
C. Publicity, Power, Independence
D. Externalities, Power, Mutual dependence
Guess: B
Probability: 0.24

Question: The Moon lacks weather and climate changes like those on Earth. What causes the lack of weather on the Moon?
Guess: the lack of magnetic poles
Probability: 0.08"""

    response_patterns = [
        re.compile(fr"(Guess: |Answer: )(?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}Probability: (?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(Guess: |Answer: )?(?P<answer>{PATTERN_ANSWER})\n+(Probability: )?(?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(Guess: |Answer: )?(?P<answer>{PATTERN_ANSWER})(, |. |; | - | \| )(Probability: )?(?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(Guess: |Answer: )?(?P<answer>{PATTERN_ANSWER}) \((Probability: )?(?P<confidence>{PATTERN_FLOAT})\)"),
    ]

    def generate_response(self, model, prompt: dict, **args):
        response, statistics = model({
            "id": prompt["id"],
            "description": f"{prompt["description"]} Provide your best guess and the probability that it is correct or plausible (0.0 to 1.0) for the following question. Take your uncertainty in the prompt, the task difficulty, your knowledge availability and other sources of uncertainty into account. Give ONLY the guess and probability, no other words or explanation. For example:\n```\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\nProbability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n```\n{PROMPT_NO_ANSWER} {self.PROMPT_EXAMPLE_FIVE_SHOT}",
            "content": prompt["content"],
        }, **args)
        return [response], [statistics]

    def extract_answer(self, responses: list):
        answer, confidence = extract_from_response(responses[0], self.response_patterns, ("answer", "confidence"))
        confidence = normalize_confidence(confidence)
        # filter out confidence scores used in few-shot examples
        if confidence in self.PROMPT_EXAMPLE_FIVE_SHOT_SCORES:
            confidence = None
        return answer, confidence

class prompt_method_tian2023just_1s_top1(PromptMethod):
    # https://aclanthology.org/2023.emnlp-main.330/
    # original prompt: "Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question. Give ONLY the guess and probability, no other words or explanation. For example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n\nThe question is: ${THE_QUESTION}"
    response_patterns = [
        re.compile(fr"(Guess: |Answer: )(?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}Probability: (?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(Guess: |Answer: )?(?P<answer>{PATTERN_ANSWER})\n+(Probability: )?(?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(Guess: |Answer: )?(?P<answer>{PATTERN_ANSWER})(, |. |; | - | \| )(Probability: )?(?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(Guess: |Answer: )?(?P<answer>{PATTERN_ANSWER}) \((Probability: )?(?P<confidence>{PATTERN_FLOAT})\)"),
    ]

    def generate_response(self, model, prompt: dict, **args):
        response, statistics = model({
            "id": prompt["id"],
            "description": f"{prompt["description"]} Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question. Give ONLY the guess and probability, no other words or explanation. For example:\n```\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\nProbability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n```\n{PROMPT_NO_ANSWER}",
            "content": prompt["content"],
        }, **args)
        return [response], [statistics]

    def extract_answer(self, responses: list):
        answer, confidence = extract_from_response(responses[0], self.response_patterns, ("answer", "confidence"))
        confidence = normalize_confidence(confidence)
        return answer, confidence


class prompt_method_tian2023just_1s_top1_v1(PromptMethod):
    # modification: best guess -> answer
    response_patterns = [
        re.compile(fr"(Answer: )(?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}Probability: (?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(Guess: |Answer: )?(?P<answer>{PATTERN_ANSWER})\n+(Probability: )?(?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(Guess: |Answer: )?(?P<answer>{PATTERN_ANSWER})(, |. |; | - | \| )(Probability: )?(?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(Guess: |Answer: )?(?P<answer>{PATTERN_ANSWER}) \((Probability: )?(?P<confidence>{PATTERN_FLOAT})\)"),
    ]

    def generate_response(self, model, prompt: dict, **args):
        response, statistics = model({
            "id": prompt["id"],
            "description": f"{prompt["description"]} Provide your answer and the probability that it is correct (0.0 to 1.0) for the following question. Give ONLY the answer and probability, no other words or explanation. For example:\n```\nAnswer: <most likely answer, as short as possible; not a complete sentence, just the answer!>\nProbability: <the probability between 0.0 and 1.0 that your answer is correct, without any extra commentary whatsoever; just the probability!>\n```\n{PROMPT_NO_ANSWER}",
            "content": prompt["content"],
        }, **args)
        return [response], [statistics]

    def extract_answer(self, responses: list):
        answer, confidence = extract_from_response(responses[0], self.response_patterns, ("answer", "confidence"))
        confidence = normalize_confidence(confidence)
        return answer, confidence


class prompt_method_tian2023just_1s_top1_v2(PromptMethod):
    # modification: probability -> confidence score
    response_patterns = [
        re.compile(fr"(Guess: |Answer: )(?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}Confidence: (?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(Guess: |Answer: )?(?P<answer>{PATTERN_ANSWER})\n+(Confidence: )?(?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(Guess: |Answer: )?(?P<answer>{PATTERN_ANSWER})(, |. |; | - | \| )(Confidence: )?(?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(Guess: |Answer: )?(?P<answer>{PATTERN_ANSWER}) \((Confidence: )?(?P<confidence>{PATTERN_FLOAT})\)"),
    ]

    def generate_response(self, model, prompt: dict, **args):
        response, statistics = model({
            "id": prompt["id"],
            "description": f"{prompt["description"]} Provide your best guess and a confidence score indicating the probability that your best guess is correct (0.0 to 1.0) for the following question. Give ONLY the guess and confidence score, no other words or explanation. For example:\n```\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\nConfidence: <the confidence score between 0.0 and 1.0 for your guess, without any extra commentary whatsoever; just the confidence score!>\n```\n{PROMPT_NO_ANSWER}",
            "content": prompt["content"],
        }, **args)
        return [response], [statistics]

    def extract_answer(self, responses: list):
        answer, confidence = extract_from_response(responses[0], self.response_patterns, ("answer", "confidence"))
        confidence = normalize_confidence(confidence)
        return answer, confidence


class prompt_method_tian2023just_1s_top1_v3(PromptMethod):
    # modification: probability -> confidence score + different formulation for "probability that it is correct"
    response_patterns = [
        re.compile(fr"(Guess: |Answer: )(?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}Confidence: (?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(Guess: |Answer: )?(?P<answer>{PATTERN_ANSWER})\n+(Confidence: )?(?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(Guess: |Answer: )?(?P<answer>{PATTERN_ANSWER})(, |. |; | - | \| )(Confidence: )?(?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(Guess: |Answer: )?(?P<answer>{PATTERN_ANSWER}) \((Confidence: )?(?P<confidence>{PATTERN_FLOAT})\)"),
    ]

    def generate_response(self, model, prompt: dict, **args):
        response, statistics = model({
            "id": prompt["id"],
            "description": f"{prompt["description"]} Provide your best guess and a confidence score quantifying how confident you are in the correctness of your answer (0.0 to 1.0) for the given task. Give ONLY the guess and confidence score, no other words or explanation. For example:\n```\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\nConfidence: <the confidence score between 0.0 and 1.0 for your guess, without any extra commentary whatsoever; just the confidence score!>\n```\n{PROMPT_NO_ANSWER}",
            "content": prompt["content"],
        }, **args)
        return [response], [statistics]

    def extract_answer(self, responses: list):
        answer, confidence = extract_from_response(responses[0], self.response_patterns, ("answer", "confidence"))
        confidence = normalize_confidence(confidence)
        return answer, confidence


class prompt_method_tian2023just_1s_top1_v4(PromptMethod):
    # modification: format without inline description
    response_patterns = [
        re.compile(fr"(Guess: |Answer: )(?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}Probability: (?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(Guess: |Answer: )?(?P<answer>{PATTERN_ANSWER})\n+(Probability: )?(?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(Guess: |Answer: )?(?P<answer>{PATTERN_ANSWER})(, |. |; | - | \| )(Probability: )?(?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(Guess: |Answer: )?(?P<answer>{PATTERN_ANSWER}) \((Probability: )?(?P<confidence>{PATTERN_FLOAT})\)"),
    ]

    def generate_response(self, model, prompt: dict, **args):
        response, statistics = model({
            "id": prompt["id"],
            "description": f"{prompt["description"]} Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question. Give ONLY the guess and probability, no other words or explanation. Use the following format to respond:\n```\nGuess: [Write your guess here.]\nProbability: [Write your probability between 0.0 and 1.0 here.]\n```\n{PROMPT_NO_ANSWER}",
            "content": prompt["content"],
        }, **args)
        return [response], [statistics]

    def extract_answer(self, responses: list):
        answer, confidence = extract_from_response(responses[0], self.response_patterns, ("answer", "confidence"))
        confidence = normalize_confidence(confidence)
        return answer, confidence


class prompt_method_tian2023just_1s_top4(PromptMethod):
    # https://aclanthology.org/2023.emnlp-main.330/
    # original prompt: "Provide your ${k} best guesses and the probability that each is correct (0.0 to 1.0) for the following question. Give ONLY the guesses and probabilities, no other words or explanation. For example:\n\nG1: <first most likely guess, as short as possible; not a complete sentence, just the guess!>\n\nP1: <the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!> ... G${k}: <${k}-th most likely guess, as short as possible; not a complete sentence, just the guess!>\n\nP${k}: <the probability between 0.0 and 1.0 that G${k} is correct, without any extra commentary whatsoever; just the probability!> \n\nThe question is: ${THE_QUESTION}"
    response_patterns = [
        re.compile(fr"G1: (?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}P1: (?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"G1: (?P<answer>{PATTERN_ANSWER})(, |; | - )(P1: )?(?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"G1: (?P<answer>{PATTERN_ANSWER}) \((P1: )?(?P<confidence>{PATTERN_FLOAT})\)"),
        re.compile(fr"(G1: )?(?P<answer>{PATTERN_ANSWER})\n+(P1: )?(?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(G1: )?(?P<answer>{PATTERN_ANSWER}) P1: (?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(G1: )?(?P<answer>{PATTERN_ANSWER})[:.] (?P<confidence>{PATTERN_FLOAT})"),
    ]

    def generate_response(self, model, prompt: dict, **args):
        response, statistics = model({
            "id": prompt["id"],
            "description": f"{prompt["description"]} Provide your 4 best guesses and the probability that each is correct (0.0 to 1.0) for the following question. Give ONLY the guesses and probabilities, no other words or explanation. For example:\n```\nG1: <first most likely guess, as short as possible; not a complete sentence, just the guess!>\nP1: <the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!>\n...\nG4: <4-th most likely guess, as short as possible; not a complete sentence, just the guess!>\nP4: <the probability between 0.0 and 1.0 that G4 is correct, without any extra commentary whatsoever; just the probability!>\n```\n{PROMPT_NO_ANSWER}",
            "content": prompt["content"],
        }, **args)
        return [response], [statistics]

    def extract_answer(self, responses: list):
        answer, confidence = extract_from_response(responses[0], self.response_patterns, ("answer", "confidence"))
        confidence = normalize_confidence(confidence)
        return answer, confidence


class prompt_method_xiong2023can_vanilla(PromptMethod):
    # https://github.com/MiaoXiong2320/llm-uncertainty/blob/78a5d2fb1dc0c54352cd5d898cccec1e16c26fc5/query_vanilla_or_cot.py#L80
    # original prompt: "Read the question, provide your answer and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.\nUse the following format to answer:\n```Answer and Confidence (0-100): [ONLY the {answer_type}; not a complete sentence], [Your confidence level, please only include the numerical number in the range of 0-100]%```\nOnly the answer and confidence, don't give me the explanation."
    response_patterns = [
        re.compile(fr"Answer: (?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}Confidence: (?P<confidence>{PATTERN_FLOAT})%?"),
        re.compile(fr"(?P<answer>{PATTERN_ANSWER})\n+(Confidence: )?(?P<confidence>{PATTERN_FLOAT})%?"),
    ]

    def generate_response(self, model, prompt: dict, **args):
        response, statistics = model({
            "id": prompt["id"],
            "description": f"{prompt["description"]} Read the question, provide your answer and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.\nUse the following format to answer:\n```\nAnswer: [ONLY the answer as short as possible; not a complete sentence]\nConfidence: [Your confidence level, please only include the numerical number in the range of 0-100]%\n```\nOnly the answer and confidence, don't give me the explanation. {PROMPT_NO_ANSWER}",
            "content": prompt["content"],
        }, **args)
        return [response], [statistics]

    def extract_answer(self, responses: list):
        answer, confidence = extract_from_response(responses[0], self.response_patterns, ("answer", "confidence"))
        confidence = normalize_confidence(confidence, lambda c: float(c) / 100)
        return answer, confidence


class prompt_method_xiong2023can_cot(PromptMethod):
    # https://github.com/MiaoXiong2320/llm-uncertainty/blob/78a5d2fb1dc0c54352cd5d898cccec1e16c26fc5/query_vanilla_or_cot.py#L78
    # original prompt: "Read the question, analyze step by step, provide your answer and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.\nUse the following format to answer:\n```Explanation: [insert step-by-step analysis here]\nAnswer and Confidence (0-100): [ONLY the {answer_type}; not a complete sentence], [Your confidence level, please only include the numerical number in the range of 0-100]%\n```\nOnly give me the reply according to this format, don't give me any other words."
    response_patterns = [
        re.compile(fr"Answer: (?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}Confidence: (?P<confidence>{PATTERN_FLOAT})%?"),
        re.compile(fr"Answer: (?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}.* Confidence: (?P<confidence>{PATTERN_FLOAT})%?"),
        re.compile(fr"(?P<answer>{PATTERN_ANSWER})\n+(Confidence: )?(?P<confidence>{PATTERN_FLOAT})%?"),
    ]

    def generate_response(self, model, prompt: dict, **args):
        response, statistics = model({
            "id": prompt["id"],
            "description": f"{prompt["description"]} Read the question, analyze step by step, provide your answer and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.\nUse the following format to answer:\n```\nExplanation: [insert step-by-step analysis here]\nAnswer: [ONLY the answer as short as possible; not a complete sentence]\nConfidence: [Your confidence level, please only include the numerical number in the range of 0-100]%\n```\nOnly give me the reply according to this format, don't give me any other words. {PROMPT_NO_ANSWER}",
            "content": prompt["content"],
        }, **args)
        return [response], [statistics]

    def extract_answer(self, responses: list):
        answer, confidence = extract_from_response(responses[0], self.response_patterns, ("answer", "confidence"))
        confidence = normalize_confidence(confidence, lambda c: float(c) / 100)
        return answer, confidence


PROMPT_METHODS = {
    "basic_1s": prompt_method_basic_1s(),
    "basic_1s_scorefloat": prompt_method_basic_1s_scorefloat(),
    "basic_1s_scoreletter": prompt_method_basic_1s_scoreletter(),
    "basic_1s_scoretext": prompt_method_basic_1s_scoretext(),
    "basic_1s_probscore": prompt_method_basic_1s_probscore(),
    "basic_1s_probscore_inline": prompt_method_basic_1s_probscore_inline(),
    "basic_1s_1shot": prompt_method_basic_1s_1shot(),
    "basic_1s_5shot": prompt_method_basic_1s_5shot(),
    "basic_2s": prompt_method_basic_2s(),
    "advanced_1s": prompt_method_advanced_1s(),
    "advanced_1s_probscore": prompt_method_advanced_1s_probscore(),
    "advanced_1s_tian2023just": prompt_method_advanced_1s_tian2023just(),
    "combo_1s": prompt_method_combo_1s(),
    "combo_1s_v2": prompt_method_combo_1s_v2(),
    "tian2023just_1s_top1": prompt_method_tian2023just_1s_top1(),
    "tian2023just_1s_top1_v1": prompt_method_tian2023just_1s_top1_v1(),
    "tian2023just_1s_top1_v2": prompt_method_tian2023just_1s_top1_v2(),
    "tian2023just_1s_top1_v3": prompt_method_tian2023just_1s_top1_v3(),
    "tian2023just_1s_top1_v4": prompt_method_tian2023just_1s_top1_v4(),
    "tian2023just_1s_top4": prompt_method_tian2023just_1s_top4(),
    "xiong2023can_vanilla": prompt_method_xiong2023can_vanilla(),
    "xiong2023can_cot": prompt_method_xiong2023can_cot(),
}
