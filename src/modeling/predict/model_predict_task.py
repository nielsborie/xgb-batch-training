import logging
import os
from typing import List

import xgboost as xgb
from pyspark.sql import DataFrame

from src.modeling.training.model_training_task import add_batch_column
from src import data_dir

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    num_rows = data.count()
    total_batches = num_rows // batch_size + 1
    logger.info(f"Experiment ID: {experiment_id} - Starting [model_predict] task ...")
    # Create the output folder for predictions
    output_folder = os.path.join(data_dir, f"batch_predictions", f"{experiment_id}", f"{prefix}")
    os.makedirs(output_folder, exist_ok=True)

    # Add a batch column if it doesn't exist
    data = add_batch_column(df=data, batch_size=batch_size)

    for i in range(total_batches):
        current_batch = i + 1
        remaining_batches = total_batches - current_batch

        # Select the data for the current batch
        batch_data = data.filter(data["batch"] == i).select(*additional_columns, target_column, *predictors)
        batch_size_current = batch_data.count()

        logger.debug(f"Processing Batch {i + 1} - Size: {batch_size_current}")

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

        logger.debug(f"Batch {i + 1} predictions completed. {remaining_batches} batches remaining. Batch saved at {batch_file_path}")

        # Release the memory
        del batch_data, data_pandas, dmatrix_data, batch_predictions

    logger.info(f"All batches predictions completed and saved in {output_folder}")
    logger.info(f"Experiment ID: {experiment_id} - [model_predict] task done.")
    return output_folder