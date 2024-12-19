import json
import logging
import os
import time

import numpy as np
import tiktoken
import torch
from accelerate.utils import get_max_memory
from jinja2.exceptions import TemplateError
from openai import OpenAI
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from utils_ext.tools import format_size, to_dict

logger = logging.getLogger(__name__)


def load_model(model_name: str, verbose: bool = True):
    if model_name not in MODELS:
        raise ValueError(f"Model \"{model_name}\" is not supported.")

    model = MODELS[model_name]

    # load model
    if isinstance(model, ModelHF):
        # set max_memory to 99% of total amount of available GPU memory
        # reference: https://github.com/huggingface/accelerate/blob/b52803dc6f8d423cd9758cdd6f77ebbd4acba035/src/accelerate/utils/modeling.py#L988
        max_memory = get_max_memory()
        max_memory_hf = {device: memory * 0.99 if isinstance(device, int) else memory for device, memory in max_memory.items()}
        # load model
        model.load(model_args=dict(max_memory=max_memory_hf))
        # compute memory usage on each GPU
        free_memory = get_max_memory()
        used_memory = {device: max_memory[device] - free_memory[device] for device in max_memory}
    else:
        model.load()

    if verbose:
        msg = f"Loaded model \"{model_name}\"."
        msg += f"\nTokenizer system_role: {model.tokenizer_supports_system_role}"
        if isinstance(model, ModelHF):
            msg += f"\nModel dtype:           {model.model.config.torch_dtype}"
            msg += f"\nModel device:          {model.model.device}"
            msg += f"\nModel device_map:      {model.model.hf_device_map if hasattr(model.model, "hf_device_map") else None}"
            msg += f"\nModel offloaded:       {"cpu" in model.model.hf_device_map.values() or "disk" in model.model.hf_device_map.values() if hasattr(model.model, "hf_device_map") else False}"
            msg += f"\nModel footprint:       {format_size(model.model.get_memory_footprint(), unit="GB", decimals=2)}"
            format_memory_dict = lambda memory_dict: {device: format_size(memory, unit="GB", decimals=2) for device, memory in memory_dict.items()}
            msg += f"\nDevice max_memory_hf:  {format_memory_dict(max_memory_hf)}"
            msg += f"\nDevice max_memory:     {format_memory_dict(max_memory)}"
            msg += f"\nDevice free_memory:    {format_memory_dict(free_memory)}"
            msg += f"\nDevice used_memory:    {format_memory_dict(used_memory)}"
        logger.info(msg)
    return model

# MODELS

class Model():
    def load(self, **args):
        pass

    def __call__(self, prompt, verbose=False, **args):
        pass


HF_MODEL_CONFIG_DEFAULT = dict(
    device_map="auto",
    torch_dtype=torch.float16,
)
HF_MODEL_CONFIG_QUANT_8BIT = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
HF_MODEL_CONFIG_QUANT_4BIT = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

class ModelHF(Model):
    def __init__(self, model_name_hf, **model_args):
        self.model_name_hf = model_name_hf
        self.model_args = model_args

    def load(self, tokenizer_args={}, model_args={}):
        # load model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_hf, **tokenizer_args)
        model = AutoModelForCausalLM.from_pretrained(self.model_name_hf, **{
            **HF_MODEL_CONFIG_DEFAULT,
            **self.model_args,
            **model_args,
        })
        # set missing pad tokens
        if tokenizer.pad_token is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model.generation_config.pad_token_id = tokenizer.eos_token_id
        # set missing chat templates
        if tokenizer.chat_template is None:
            if self.model_name_hf.startswith("tiiuae/falcon"):
                tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ system_message.strip() }}{% endif %}{{ '\n\n' + message['role'].title() + ': ' + message['content'].strip().replace('\r\n', '\n').replace('\n\n', '\n') }}{% if loop.last and message['role'] == 'user' and add_generation_prompt %}{{ '\n\nAssistant:' }}{% endif %}{% endfor %}"
        # check for system role support
        try:
            tokenizer.apply_chat_template([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hi."}])
            tokenizer_supports_system_role = True
        except:
            tokenizer_supports_system_role = False

        self.tokenizer = tokenizer
        self.model = model
        self.tokenizer_supports_system_role = tokenizer_supports_system_role

    def __call__(self, prompt, verbose=False, **args):
        # set default parameters
        if "max_new_tokens" not in args:
            args["max_new_tokens"] = 512
        if "do_sample" not in args:
            args["do_sample"] = True

        # start timer
        time_start = time.perf_counter()
        # encode input with chat template
        try:
            prompt_chat = [
                {"role": "system", "content": prompt["description"]},
                {"role": "user", "content": prompt["content"]},
            ]
            input = self.tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        except TemplateError:
            prompt_chat = [
                {"role": "user", "content": prompt["description"] + "\n\n" + prompt["content"]},
            ]
            input = self.tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        input_ids = input["input_ids"]
        input_mask = input["attention_mask"]
        if verbose:
            print("===== START OF PROMPT =====")
            print(self.tokenizer.decode(input_ids[0]))
            print("===== END OF PROMPT =====")
        # generate response
        input_ids = input_ids.to(self.model.device)
        input_mask = input_mask.to(self.model.device)
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=input_mask,
            **args,
        )
        # decode output
        output_ids = output_ids[:, input_ids.shape[1]:]
        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        if verbose:
            print("===== START OF RESPONSE =====")
            print(self.tokenizer.decode(output_ids[0]))
            print("===== END OF RESPONSE =====")
        # stop timer
        time_end = time.perf_counter()

        # collect statistics
        statistics = dict(
            time=time_end-time_start,
            n_input_chars=(len(prompt["description"]), len(prompt["content"])),
            n_input_tokens=input_ids.shape[1],
            n_output_chars=len(output),
            n_output_tokens=output_ids.shape[1],
        )

        return output, statistics

class BatchOverflowException(Exception):
    pass

class ModelAPI_OpenAI(Model):
    # https://platform.openai.com/docs/guides/batch/rate-limits
    LIMIT_BATCH_NUM_REQUESTS = 50000
    # https://platform.openai.com/docs/guides/rate-limits/usage-tiers
    LIMIT_BATCH_TOKENS = {
        # # tier 1
        # "gpt-3.5-turbo-0125": 2000000,
        # "gpt-4-0314": 100000,
        # "gpt-4-turbo-2024-04-09": 90000,
        # "gpt-4o-2024-05-13": 90000,
        # "gpt-4o-mini-2024-07-18": 2000000,

        # tier 2
        "gpt-3.5-turbo-0125": 5000000,
        "gpt-4-0314": 200000,
        "gpt-4-turbo-2024-04-09": 1350000,
        "gpt-4o-2024-05-13": 1350000,
        "gpt-4o-mini-2024-07-18": 20000000,
    }

    def __init__(self, model_name_api, cost_instant, cost_batch, **model_args):
        self.model_name_api = model_name_api
        self.cost_instant = cost_instant # (input, output) [USD per 1 million tokens]
        self.cost_batch = cost_batch # (input, output) [USD per 1 million tokens]
        self.model_args = model_args

        self.batch_mode = False
        logging.getLogger("httpx").setLevel(logging.WARNING)

    def load(self, **args):
        self.client = OpenAI(**self.model_args, **args)
        self.tokenizer = tiktoken.encoding_for_model(self.model_name_api)
        self.tokenizer_supports_system_role = True

    def __call__(self, prompt, verbose=False, **args):
        # set default parameters
        if "max_tokens" not in args:
            args["max_tokens"] = 512
        if "logprobs" not in args:
            args["logprobs"] = True
        if "top_logprobs" not in args:
            args["top_logprobs"] = 5

        # create coroutine
        call_coro = self.__call_coroutine(prompt, verbose=verbose, **args)

        if not self.batch_mode:
            # call coroutine
            prompt_chat = next(call_coro)

            # generate response
            time_start = time.perf_counter()
            completion = self.client.chat.completions.create(
                model=self.model_name_api,
                messages=prompt_chat,
                **args,
            )
            completion = to_dict(completion)
            time_end = time.perf_counter()

            # call coroutine
            output, statistics = call_coro.send((completion, time_end-time_start))
            statistics["completion"] = completion
            return output, statistics
        else:
            # call coroutine
            prompt_chat = next(call_coro)

            # create request
            request_id = str(prompt["id"])
            request = {
                "custom_id": request_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model_name_api,
                    "messages": prompt_chat,
                    **args,
                }
            }
            num_tokens = self.num_tokens_from_messages(prompt_chat)

            # process batch if full
            if (
                len(self.batch_requests) + 1 > ModelAPI_OpenAI.LIMIT_BATCH_NUM_REQUESTS
                or self.batch_num_tokens + num_tokens > 0.80 * ModelAPI_OpenAI.LIMIT_BATCH_TOKENS[self.model_name_api]
            ):
                raise BatchOverflowException()

            # add request to batch
            self.batch_requests[request_id] = request
            self.batch_callbacks[request_id] = call_coro
            self.batch_num_tokens += num_tokens

            return None, None

    def __call_coroutine(self, prompt, verbose=False, **args):
        # create input chat
        prompt_chat = [
            {"role": "system", "content": prompt["description"]},
            {"role": "user", "content": prompt["content"]},
        ]
        if verbose:
            print("===== START OF PROMPT =====")
            for p in prompt_chat:
                print(f"<{p["role"]}>")
                print(p["content"])
                print(f"</{p["role"]}>")
            print("===== END OF PROMPT =====")

        # generate response via coroutine
        completion, time_total = yield prompt_chat

        # extract output and statistics
        output = completion["choices"][0]["message"]["content"]
        output_finish_reason = completion["choices"][0]["finish_reason"]
        if not self.batch_mode:
            output_costs = (
                self.cost_instant[0] * completion["usage"]["prompt_tokens"]
                + self.cost_instant[1] * completion["usage"]["completion_tokens"]
            ) / 1e6
        else:
            output_costs = (
                self.cost_batch[0] * completion["usage"]["prompt_tokens"]
                + self.cost_batch[1] * completion["usage"]["completion_tokens"]
            ) / 1e6

        output_probs = [
            [(t2["token"], np.around(np.exp(t2["logprob"]), decimals=4)) for t2 in t1["top_logprobs"]]
            for t1 in completion["choices"][0]["logprobs"]["content"]
        ]
        statistics = dict(
            time=time_total,
            n_input_chars=(len(prompt["description"]), len(prompt["content"])),
            n_input_tokens=completion["usage"]["prompt_tokens"],
            n_output_chars=len(output),
            n_output_tokens=completion["usage"]["completion_tokens"],
            compl_finish_reason=output_finish_reason,
            compl_costs=output_costs,
            compl_probs=output_probs,
        )
        if verbose:
            print("===== START OF RESPONSE =====")
            print(output)
            print("===== END OF RESPONSE =====")
            print("===== START OF STATISTICS =====")
            print(f"finish reason:  {output_finish_reason}")
            print(f"costs:          ¢ {output_costs*100:f}")
            output_probs_str = "\n".join(
                " ".join(f"{str(prob):20}" for prob in probs)
                for probs in output_probs
            )
            print(f"probabilities:\n{output_probs_str}")
            print("===== END OF STATISTICS =====")

        # return response via coroutine
        yield output, statistics

    def create_batch(self, path_requests="batch_requests.jsonl", path_responses="batch_responses.jsonl", metadata={}):
        self.batch_mode = True
        self.batch_path_requests = path_requests
        self.batch_path_responses = path_responses
        self.batch_metadata = metadata

        self.batch_requests = {}
        self.batch_callbacks = {}
        self.batch_num_tokens = 0

    def submit_batch(self, verbose=False):
        # create batch request file
        os.makedirs(os.path.dirname(self.batch_path_requests), exist_ok=True)
        with open(self.batch_path_requests, "w") as file:
            for request in self.batch_requests.values():
                file.write(json.dumps(request) + "\n")
        if verbose:
            logger.info(f"Batch requests saved to \"{self.batch_path_requests}\".")

        # upload batch request file
        batch_file = self.client.files.create(
            file=open(self.batch_path_requests, "rb"),
            purpose="batch",
        )
        # create batch job
        batch_job = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata=self.batch_metadata,
        )
        if verbose:
            msg = f"Batch job created with id \"{batch_job.id}\". Waiting for it to complete..."
            msg += f"\nNumber of requests: {len(self.batch_requests)}"
            msg += f"\nNumber of tokens:   {self.batch_num_tokens}"
            logger.info(msg)

        # wait for batch job to complete
        delay = 15
        while batch_job.status in ["validating", "in_progress", "finalizing"]:
            time.sleep(delay)
            batch_job = self.client.batches.retrieve(batch_job.id)
        if batch_job.status != "completed":
            logger.warning(f"Batch job \"{batch_job.id}\" finished with status \"{batch_job.status}\".")

        # retrieve batch responses
        batch_responses = self.client.files.content(batch_job.output_file_id).content
        # save batch responses
        os.makedirs(os.path.dirname(self.batch_path_responses), exist_ok=True)
        with open(self.batch_path_responses, "wb") as f:
            f.write(batch_responses)
        if verbose:
            logger.info(f"Batch responses saved to \"{self.batch_path_responses}\".")

        # process batch responses
        results_batch = {}
        time_total = (batch_job.completed_at - batch_job.in_progress_at) / batch_job.request_counts.total
        with open(self.batch_path_responses, "r") as f:
            for line in f:
                response = json.loads(line.strip())
                response_id = response["custom_id"]
                completion = response["response"]["body"]
                # call coroutine
                response, statistics = self.batch_callbacks[response_id].send((completion, time_total))
                results_batch[response_id] = (response, statistics)
        if verbose:
            num_tokens_input = 0
            num_tokens_output = 0
            costs = 0
            for _, statistics in results_batch.values():
                num_tokens_input += statistics["n_input_tokens"]
                num_tokens_output += statistics["n_output_tokens"]
                costs += statistics["compl_costs"]
            msg = "Batch statistics:"
            msg += f"\nNumber of requests:      {batch_job.request_counts.completed} / {batch_job.request_counts.total}"
            msg += f"\nNumber of input tokens:  {num_tokens_input}"
            msg += f"\nNumber of output tokens: {num_tokens_output}"
            msg += f"\nCosts:                   $ {costs:f}"
            logger.info(msg)

        self.batch_mode = False
        self.batch_path_requests = None
        self.batch_path_responses = None
        self.batch_metadata = None

        self.batch_requests = {}
        self.batch_callbacks = {}
        self.batch_num_tokens = 0

        return results_batch

    # reference: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def num_tokens_from_messages(self, messages):
        if self.model_name_api in [
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            # new models
            "gpt-3.5-turbo-0125",
            "gpt-4-turbo-2024-04-09",
            "gpt-4o-2024-05-13",
            "gpt-4o-mini-2024-07-18",
        ]:
            tokens_per_message = 3
            tokens_per_name = 1
        elif self.model_name_api in [
            "gpt-3.5-turbo-0301",
        ]:
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        else:
            raise NotImplementedError(f"num_tokens_from_messages() is not implemented for model {self.model_name_api}.")

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(self.tokenizer.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens


MODELS = {
    "falcon-7b-instruct":       ModelHF("tiiuae/falcon-7b-instruct"),
    "falcon-40b-instruct":      ModelHF("tiiuae/falcon-40b-instruct"),
    "falcon-40b-instruct-8bit": ModelHF("tiiuae/falcon-40b-instruct", quantization_config=HF_MODEL_CONFIG_QUANT_8BIT),
    "falcon-180b-chat":         ModelHF("tiiuae/falcon-180b-chat"),
    "falcon-180b-chat-8bit":    ModelHF("tiiuae/falcon-180b-chat", quantization_config=HF_MODEL_CONFIG_QUANT_8BIT),

    "gemma-2b-it":    ModelHF("google/gemma-2b-it", hidden_activation="gelu_pytorch_tanh"),
    "gemma-7b-it":    ModelHF("google/gemma-7b-it", hidden_activation="gelu_pytorch_tanh"),
    "gemma1.1-2b-it": ModelHF("google/gemma-1.1-2b-it"),
    "gemma1.1-7b-it": ModelHF("google/gemma-1.1-7b-it"),

    "llama2-7b-chat":           ModelHF("meta-llama/Llama-2-7b-chat-hf"),
    "llama2-13b-chat":          ModelHF("meta-llama/Llama-2-13b-chat-hf"),
    "llama2-70b-chat":          ModelHF("meta-llama/Llama-2-70b-chat-hf"),
    "llama2-70b-chat-8bit":     ModelHF("meta-llama/Llama-2-70b-chat-hf", quantization_config=HF_MODEL_CONFIG_QUANT_8BIT),
    "llama3-8b-instruct":       ModelHF("meta-llama/Meta-Llama-3-8B-Instruct"),
    "llama3-70b-instruct":      ModelHF("meta-llama/Meta-Llama-3-70B-Instruct"),
    "llama3-70b-instruct-8bit": ModelHF("meta-llama/Meta-Llama-3-70B-Instruct", quantization_config=HF_MODEL_CONFIG_QUANT_8BIT),

    "mistral0.1-7b-instruct":         ModelHF("mistralai/Mistral-7B-Instruct-v0.1"),
    "mistral0.2-7b-instruct":         ModelHF("mistralai/Mistral-7B-Instruct-v0.2"),
    "mixtral0.1-8x7b-instruct":       ModelHF("mistralai/Mixtral-8x7B-Instruct-v0.1"),
    "mixtral0.1-8x7b-instruct-8bit":  ModelHF("mistralai/Mixtral-8x7B-Instruct-v0.1", quantization_config=HF_MODEL_CONFIG_QUANT_8BIT),
    "mixtral0.1-8x22b-instruct":      ModelHF("mistralai/Mixtral-8x22B-Instruct-v0.1"),
    "mixtral0.1-8x22b-instruct-8bit": ModelHF("mistralai/Mixtral-8x22B-Instruct-v0.1", quantization_config=HF_MODEL_CONFIG_QUANT_8BIT),

    "qwen1.5-7b-chat":        ModelHF("Qwen/Qwen1.5-7B-Chat"),
    "qwen1.5-14b-chat":       ModelHF("Qwen/Qwen1.5-14B-Chat"),
    "qwen1.5-32b-chat":       ModelHF("Qwen/Qwen1.5-32B-Chat"),
    "qwen1.5-72b-chat":       ModelHF("Qwen/Qwen1.5-72B-Chat"),
    "qwen1.5-72b-chat-8bit":  ModelHF("Qwen/Qwen1.5-72B-Chat", quantization_config=HF_MODEL_CONFIG_QUANT_8BIT),
    "qwen1.5-110b-chat":      ModelHF("Qwen/Qwen1.5-110B-Chat"),
    "qwen1.5-110b-chat-8bit": ModelHF("Qwen/Qwen1.5-110B-Chat", quantization_config=HF_MODEL_CONFIG_QUANT_8BIT),

    "gpt3.5-turbo": ModelAPI_OpenAI("gpt-3.5-turbo-0125", (0.5, 1.5), (0.25, 0.75)),
    "gpt4-turbo":   ModelAPI_OpenAI("gpt-4-turbo-2024-04-09", (10, 30), (5, 15)),
    "gpt4o-mini":   ModelAPI_OpenAI("gpt-4o-mini-2024-07-18", (0.15, 0.60), (0.075, 0.30)),
    "gpt4o":        ModelAPI_OpenAI("gpt-4o-2024-05-13", (5, 15), (2.5, 7.5)),
}
