"""DC extension example: python examples/custom_method.py --data-root /path/to/data."""

import argparse
import json

import numpy as np

from ml_opf_bench import Prediction, create_method, load_dataset, predict_method, register_method, evaluate_predictions


class MeanDispatch:
    def fit(self, train, validation, *, seed):
        self.pg_mean = train.targets["pg_non_slack"].mean(axis=0)

    def predict(self, inputs):
        return Prediction(pg=np.tile(self.pg_mean, (len(inputs.indices), 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--case", default="case30")
    args = parser.parse_args()
    data = load_dataset(args.data_root, "dc", args.case)
    split = data.split()
    register_method("dc", "MEAN-DISPATCH", MeanDispatch)
    method = create_method("dc", "MEAN-DISPATCH")
    method.fit(split.train, split.validation, seed=42)
    prediction = predict_method(method, split.test.inputs)
    result = evaluate_predictions(split.test, prediction, name="MEAN-DISPATCH")
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
