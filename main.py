import argparse
import logging
from core.config_loader import load_config
from core.data_processor import DataProcessor
from core.model_trainer import ModelTrainer
from core.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Generic Retail Forecasting Framework")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # 1. Process Data  (returns df + preprocessing metadata)
    processor = DataProcessor(config)
    df, preprocessing_meta = processor.run_pipeline()

    # 2. Train Models  (returns results + updates metadata with feature selection info)
    trainer = ModelTrainer(config)
    results, preprocessing_meta = trainer.run_training(df, preprocessing_meta)

    # 3. Evaluate + generate Explainable AI report
    evaluator = Evaluator(config)
    evaluator.evaluate(results, preprocessing_meta)

    logging.info("🚀  Pipeline Execution Complete!")


if __name__ == "__main__":
    main()
