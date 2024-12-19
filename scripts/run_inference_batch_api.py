import sys
sys.path.append("src")

import argparse
import json
import logging
import os

from tqdm.auto import tqdm

from benchmark import (
    BatchOverflowException,
    load_dataset,
    load_model,
    load_prompt_method,
)
from utils_ext.tools import setup_logging

logger = logging.getLogger(__name__)

def main(args):
    logger.info(f"Args: {args}")

    # load model once to save loading overhead
    model = load_model(args.model)
    if not hasattr(model, "create_batch") or not hasattr(model, "submit_batch"):
        raise ValueError(f"Model {model} does not support batch evaluation.")

    # create batch jobs
    for dataset_name in args.datasets:
        for method_name in args.methods:
            # create results folder
            path = f"{args.out}/{dataset_name}/{args.model}/{method_name}.json"
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # create batch requests file
            batch_index = 0
            path_batch_requests = lambda i: f"{args.out}/_batch_requests/{dataset_name}/{args.model}/{method_name}_{i}.jsonl"
            path_batch_responses = lambda i: f"{args.out}/_batch_responses/{dataset_name}/{args.model}/{method_name}_{i}.jsonl"

            # load dataset and prompt method
            dataset = load_dataset(dataset_name)
            prompt_method = load_prompt_method(method_name)

            # generate responses with batch requests
            results = []
            model.create_batch(
                path_requests=path_batch_requests(batch_index),
                path_responses=path_batch_responses(batch_index),
                metadata={
                    "out": args.out,
                    "dataset": dataset_name,
                    "model": args.model,
                    "method": method_name,
                    "batch_index": str(batch_index),
                },
            )
            batch_index += 1
            for prompt in tqdm(dataset):
                try:
                    prompt_method.generate_response(model, prompt)
                except BatchOverflowException:
                    print() # add newline because of tqdm

                    # submit batch and collect results
                    results_batch = model.submit_batch(verbose=True)
                    for id, (response, statistics) in results_batch.items():
                        results.append({
                            "id": id,
                            "responses": [response],
                            "statistics": [statistics],
                        })
                    # save results
                    with open(path, "w") as f:
                        json.dump(results, f, indent=4)

                    # retry generate for response
                    model.create_batch(
                        path_requests=path_batch_requests(batch_index),
                        path_responses=path_batch_responses(batch_index),
                        metadata={
                            "out": args.out,
                            "dataset": dataset_name,
                            "model": args.model,
                            "method": method_name,
                            "batch_index": str(batch_index),
                        },
                    )
                    batch_index += 1
                    prompt_method.generate_response(model, prompt)

            # submit batch and collect results
            results_batch = model.submit_batch(verbose=True)
            for id, (response, statistics) in results_batch.items():
                results.append({
                    "id": id,
                    "responses": [response],
                    "statistics": [statistics],
                })
            # save results
            with open(path, "w") as f:
                json.dump(results, f, indent=4)
            logger.info(f"Results saved to \"{path}\".")


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=[])
    parser.add_argument("--model")
    parser.add_argument("--methods", nargs="+", default=[])
    parser.add_argument("--out", default="output")
    args = parser.parse_args()

    main(args)
