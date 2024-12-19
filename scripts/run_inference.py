import sys
sys.path.append("src")

import argparse
import json
import logging
import os

from tqdm.auto import tqdm

from benchmark import load_dataset, load_model, load_prompt_method
from utils_ext.tools import setup_logging

logger = logging.getLogger(__name__)

def main(args):
    logger.info(f"Args: {args}")

    # load model once to save loading overhead
    model = load_model(args.model)

    for dataset_name in args.datasets:
        for method_name in args.methods:
            # create results folder
            path = f"{args.out}/{dataset_name}/{args.model}/{method_name}.json"
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # load dataset and prompt method
            dataset = load_dataset(dataset_name)
            prompt_method = load_prompt_method(method_name)

            # generate responses
            results = []
            for i, prompt in enumerate(tqdm(dataset)):
                responses, statistics = prompt_method.generate_response(model, prompt)
                results.append({
                    "id": prompt["id"],
                    "responses": responses,
                    "statistics": statistics,
                })

                # save results
                if i+1 % 1 == 100:
                    with open(path, "w") as f:
                        json.dump(results, f, indent=4)

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
