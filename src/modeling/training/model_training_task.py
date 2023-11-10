import logging
import os
import time
from typing import List, Dict

import xgboost as xgb
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src import models_dir

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def model_training_task(experiment_id: str,
                        train_data: DataFrame,
                        predictors: List[str],
                        target_column: str,
                        batch_size: int = 1000,
                        params: Dict = None,
                        model_dir: str = models_dir) -> str:
    """
    Trains an XGBoost classifier incrementally on the training data.

    Args:
    experiment_id (str): Unique identifier for the experiment.
    train_data (DataFrame): DataFrame containing the training data.
    predictors (list): List of feature columns.
    target_column (str): The name of the target column.
    batch_size (int): Size of the batch for incremental training.
    params (dict): Dictionary containing XGBoost parameters.
    model_dir (str): Directory path to store the trained model.

    Returns:
    str: Path to the saved XGBoost model.

    Raises:
    ValueError: If parameters are missing or invalid.
    """
    if params is None:
        raise ValueError("XGBoost parameters are missing.")

    if not isinstance(train_data, DataFrame):
        raise ValueError("train_data should be a valid DataFrame.")

    if not predictors:
        raise ValueError("List of predictors is empty.")

    if target_column not in train_data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")

    if batch_size <= 0:
        raise ValueError("Batch size should be a positive integer.")

    logger.info(f"Experiment ID: {experiment_id} - Starting [model_training] task ...")
    start_time = time.time()
    xgb_model = None
    num_rows = train_data.count()
    num_batches = (num_rows + batch_size - 1) // batch_size
    splits = train_data.randomSplit([1.0] * num_batches, seed=42)
    total_batches = len(splits)
    batch_count = 0
    for i, batch_train_data in enumerate(splits):
        batch_start_time = time.time()
        current_batch = i + 1
        remaining_batches = total_batches - current_batch

        batch_size_current = batch_train_data.count()
        num_positive = batch_train_data.filter(F.col(target_column) == True).count()
        num_negative = batch_train_data.filter(F.col(target_column) == False).count()
        logger.info(f"Processing Batch {i + 1} / {total_batches} - Size: {batch_size_current}. Proportions in column '{target_column}' -> True: {num_positive}, False: {num_negative} ")

        batch_count += batch_size_current

        data_pandas = batch_train_data.toPandas()
        y_train = data_pandas[target_column].values
        X_train = data_pandas[predictors].values
        dtrain = xgb.DMatrix(data=X_train, label=y_train, nthread=1)

        if i == 0:
            xgb_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=10)
        else:
            xgb_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=10, xgb_model=xgb_model)

        batch_completion_time = time.time() - batch_start_time
        logger.info(f"Batch {i + 1} / {total_batches} trained. {remaining_batches} batches remaining. Batch Completion Time: {batch_completion_time / 60:.2f} minutes. Percentage of dataset processed: {(batch_count / num_rows) * 100:.2f}%.")

        # Release the memory
        batch_train_data.unpersist()
        del data_pandas, batch_train_data, dtrain, X_train, y_train

    end_time = time.time() - start_time
    logger.info(f"XGBoost batch model training complete. Took: {end_time / 60:.2f} minutes to train {total_batches} batch. Total number of rows processed: {batch_count} / {num_rows}.")

    model_path = os.path.join(model_dir, experiment_id, "xgb_trained.model")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    xgb_model.save_model(model_path)
    logger.info(f"XGBoost model saved to : {model_path}")
    logger.info(f"Experiment ID: {experiment_id} - [model_training] task done.")

    return model_path
