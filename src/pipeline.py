from __future__ import annotations

import argparse
from src.config import ProjectConfig
from src.modeling.train_xgboost import train_xgboost


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=ProjectConfig().data_path)
    parser.add_argument("--model-path", default=ProjectConfig().model_path)
    args = parser.parse_args()

    metrics = train_xgboost(args.data_path, args.model_path)
    print("Training complete:", metrics)


if __name__ == "__main__":
    main()
