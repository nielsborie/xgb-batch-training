import logging
from datetime import datetime
from datetime import timedelta
from typing import Tuple

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import BooleanType
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src import reports_dir

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_gantt(report_dir: str,
               experiment_id: str,
               start_dates: list,
               end_dates: list,
               sizes: list,
               periods: list) -> None:
    """
    Create a Gantt chart representing the duration of each period.

    Args:
        report_dir (str): The directory to save the chart in.
        experiment_id (str): The identifier of the experiment.
        start_dates (list): List of start dates for each period.
        end_dates (list): List of end dates for each period.
        sizes (list): List of sizes of each period.
        periods (list): List of names of the periods.

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis_date()

    colors = ['green', 'orange', 'pink']
    for i in range(len(start_dates)):
        plt.barh(y=periods[i], width=(end_dates[i] - start_dates[i]), left=start_dates[i], height=0.5, color=colors[i],
                 alpha=0.5)
        plt.text(start_dates[i], periods[i], f' Size: {sizes[i]} rows', va='center', ha='left')

    plt.xlabel('Time')
    plt.ylabel('Periods')
    plt.title(f'Gantt Chart for Experiment ID : {experiment_id}')

    experiment_dir = os.path.join(report_dir, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)
    plot_path = os.path.join(experiment_dir, f"gantt_plot.png")
    plt.savefig(plot_path)


def prepare_ml_experiment_task(data: DataFrame,
                               target_col: str,
                               datetime_col: str,
                               start_date: str = None,
                               end_date: str = None,
                               val_size_weeks: int = 4,
                               test_size_weeks: int = 4,
                               report_dir=reports_dir) -> Tuple[DataFrame, DataFrame, DataFrame, str]:
    """
    Create "train", "val", and "test" datasets based on the specified dates.

    Args:
    data (DataFrame): The DataFrame containing transaction data.
    target_col (str): Target column.
    datetime_col (str): Datetime column to use to split the dataset.
    start_date (str): Start date for the training set.
    end_date (str): End date for the test set.
    val_size_weeks (int): Size of the validation set in weeks.
    test_size_weeks (int): Size of the test set in weeks.

    Returns:
    train_data (DataFrame): Training dataset.
    val_data (DataFrame): Validation dataset.
    test_data (DataFrame): Test dataset.
    experiment_id (str): Unique identifier for the experiment.
    """

    # Generating experiment_id from current date and time
    experiment_id = datetime.now().strftime("%Y%m%d%H%M%S")

    logger.info(f"Experiment ID: {experiment_id} - Starting the [prepare_ml_experiment] task...")

    total_rows = data.count()
    if total_rows == 0:
        raise ValueError("No data found in the input dataset.")

    # If start_date and end_date are not provided, use the minimum and maximum datetime_col in the data
    if start_date is None:
        start_date = data.agg({datetime_col: "min"}).collect()[0][0]
        logger.info(f"No 'start_date' was provided using the minimum found in the dataset : {start_date}")

    if end_date is None:
        end_date = data.agg({datetime_col: "max"}).collect()[0][0]
        logger.info(f"No 'end_date' was provided using the maximum found in the dataset : {end_date}")

    # Convert start_date and end_date to date type
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    logger.info(f"Initial data - Start Date: {start_date}, End Date: {end_date}, Size: {data.count()} rows")

    # Calculate the start date for the validation and test sets
    test_start_date = end_date - timedelta(weeks=test_size_weeks)
    val_start_date = test_start_date - timedelta(weeks=val_size_weeks)

    data = data.withColumn(target_col, F.col(target_col).cast(BooleanType()))

    # Filter the data based on dates using PySpark
    train_data = data.filter((F.col(datetime_col) >= start_date) & (F.col(datetime_col) < val_start_date))
    val_data = data.filter((F.col(datetime_col) >= val_start_date) & (F.col(datetime_col) < test_start_date))
    test_data = data.filter(F.col(datetime_col) >= test_start_date)

    train_rows = train_data.count()
    val_rows = val_data.count()
    test_rows = test_data.count()
    # Check if any of the DataFrames is empty
    if train_rows == 0:
        raise ValueError("No data found in the train dataset. Check your parameters !")

    if val_rows == 0:
        raise ValueError("No data found in the validation dataset. Check your parameters !")

    if test_rows == 0:
        raise ValueError("No data found in the test dataset. Check your parameters !")
    start_dates = [start_date, val_start_date, test_start_date]
    end_dates = [val_start_date, test_start_date, end_date]
    sizes = [train_rows, val_rows, test_rows]
    periods = ["Train", "Validation", "Test"]
    plot_gantt(report_dir=report_dir, experiment_id=experiment_id, start_dates=start_dates,
               end_dates=end_dates, sizes=sizes, periods=periods)

    logger.info(f"Training Data - Start Date: {start_date}, End Date: {val_start_date}, Size: {train_rows} rows")
    logger.info(f"Validation Data - Start Date: {val_start_date}, End Date: {test_start_date}, Size: {val_rows} rows")
    logger.info(f"Test Data - Start Date: {test_start_date}, End Date: {end_date}, Size: {test_rows} rows")
    logger.info(f"Experiment ID: {experiment_id} - [prepare_ml_experiment] task done.")

    return train_data, val_data, test_data, experiment_id
