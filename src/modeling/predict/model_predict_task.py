import logging
import os
from typing import List

import xgboost as xgb
from pyspark.sql import DataFrame

from src import data_dir

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def model_predict_task(experiment_id: str,
                       prefix: str,
                       xgb_model_path: str,
                       data: DataFrame,
                       predictors: List[str],
                       target_column: str,
                       additional_columns: List[str],
                       batch_size: int = 1000,
                       data_dir: str = data_dir) -> str:
    """
    Generates predictions for the provided data using the trained XGBoost model.

    Args:
    experiment_id (str): Unique identifier for the experiment.
    prefix (str): Prefix for the output folder name.
    xgb_model_path (str): Path to the saved XGBoost model.
    data (DataFrame): DataFrame containing the data for prediction.
    predictors (list): List of feature columns.
    target_column (str): The name of the target column.
    additional_columns (list): List of additional columns to forward.
    batch_size (int): Size of the batch for prediction.
    data_dir (str): Directory path to store the predicted batch data.

    Returns:
    str: Path to the folder containing the predictions.
    """
    # Load the XGBoost model
    xgb_model = xgb.Booster()
    xgb_model.load_model(xgb_model_path)

    # Get the total number of rows and calculate the total number of batches
    logger.info(f"Experiment ID: {experiment_id} - Starting [model_predict] task ...")
    # Create the output folder for predictions
    output_folder = os.path.join(data_dir, f"batch_predictions", f"{experiment_id}", f"{prefix}")
    os.makedirs(output_folder, exist_ok=True)

    num_rows = data.count()
    num_batches = (num_rows + batch_size - 1) // batch_size
    splits = data.randomSplit([1.0] * num_batches, seed=42)
    total_batches = len(splits)
    batch_count = 0
    for i, batch_data in enumerate(splits):
        current_batch = i + 1
        remaining_batches = total_batches - current_batch

        # Select the data for the current batch
        batch_size_current = batch_data.count()
        batch_count += batch_size_current

        logger.info(f"Processing Batch {i + 1} / {total_batches} - Size: {batch_size_current}")

        # Convert the Spark DataFrame to a Pandas DataFrame
        data_pandas = batch_data.toPandas()

        # Create the DMatrix for prediction
        dmatrix_data = xgb.DMatrix(data_pandas[predictors])
        batch_predictions = xgb_model.predict(dmatrix_data)

        # Add predictions to the original DataFrame
        data_pandas['xgb.final_score'] = batch_predictions

        # Write the predictions to a Parquet file
        batch_file_path = os.path.join(output_folder, f"batch_{i}.parquet")
        data_pandas.to_parquet(batch_file_path, engine='pyarrow')

        logger.info(f"Batch {i + 1} / {total_batches} predictions completed. {remaining_batches} batches remaining. Percentage of dataset processed: {(batch_count / num_rows) * 100:.2f}%.")
        logger.info(f"Batch {i + 1} / {total_batches} predictions saved in {batch_file_path}")

        # Release the memory
        del batch_data, data_pandas, dmatrix_data, batch_predictions

    logger.info(f"Total number of rows processed: {batch_count} / {num_rows}.")
    logger.info(f"All batches predictions completed and saved in {output_folder}")
    logger.info(f"Experiment ID: {experiment_id} - [model_predict] task done.")
    return output_folder