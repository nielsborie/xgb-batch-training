import logging
import os
from typing import Dict, Optional, List
from typing import TextIO

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score, roc_curve

logger = logging.getLogger(__name__)


def consolidate_reports(experiment_id: str, report_dir: str, target_col: str) -> Optional[str]:
    """
    Consolidates the reports generated for the validation and test datasets into a single combined report.

    Args:
        experiment_id (str): Unique identifier for the experiment.
        report_dir (str): Directory path to store the combined report.
        target_col (str): Target col name.

    Returns:
        str: Path to the combined report, or None if an error occurs during the consolidation process.
    """
    logger.info(f"Experiment ID: {experiment_id} - Starting [consolidate_report] task...")

    try:
        experiment_dir = os.path.join(report_dir, experiment_id)
        combined_report_path = os.path.join(experiment_dir, "combined_report.html")
        gantt_plot_path = os.path.relpath(os.path.join(experiment_dir, "gantt_plot.png"), experiment_dir)

        with open(combined_report_path, 'w') as combined_report:
            combined_report.write("<html><head>")
            combined_report.write("<style>")
            combined_report.write(get_css_style())
            combined_report.write("</style>")
            combined_report.write("</head><body>")

            write_gantt_section(combined_report=combined_report,
                                experiment_id=experiment_id,
                                gantt_plot_path=gantt_plot_path)
            write_section(combined_report=combined_report,
                          title="Validation Report",
                          experiment_dir=experiment_dir,
                          target_col=target_col,
                          prefix="val")
            write_section(combined_report=combined_report,
                          title="Test Report",
                          experiment_dir=experiment_dir,
                          target_col=target_col,
                          prefix="test")

            combined_report.write("</body></html>")

        logger.info(f"Reports consolidated successfully. Combined report saved at {combined_report_path}")
        logger.info(f"Experiment ID: {experiment_id} - [consolidate_report] task done.")
        return combined_report_path

    except Exception as e:
        logger.error(f"An error occurred during report consolidation: {e}")
        return None


def write_gantt_section(combined_report: TextIO, experiment_id: str, gantt_plot_path: str) -> None:
    """
    Write the gantt section in the combined report.

    Args:
        combined_report (TextIO): The file object for the combined report.
        experiment_id (str): Unique identifier for the experiment.
        gantt_plot_path (str): The relative path to the gantt plot.

    Returns:
        None
    """
    combined_report.write(f"<details>")
    combined_report.write(f"<summary>Experiment Details</summary>")
    combined_report.write("<div class='section'>")
    combined_report.write(f"<h2 style='text-align: center;'>Experiment ID : {experiment_id}</h2>")
    combined_report.write(f"<img src='{gantt_plot_path}'><br>")
    combined_report.write("</div>")
    combined_report.write("</details>")


def write_section(combined_report: TextIO, title: str, experiment_dir: str, prefix: str, target_col: str) -> None:
    """
    Write a section in the combined report.

    Args:
        combined_report (TextIO): The file object for the combined report.
        title (str): The title of the section.
        experiment_dir (str): The directory for the experiment.
        prefix (str): The prefix of the directory to use (should be something like "val" or "test").
        target_col (str): The target column name.

    Returns:
        None
    """
    combined_report.write(f"<details>")
    combined_report.write(f"<summary>{title}</summary>")
    combined_report.write("<div class='section'>")
    write_report(experiment_dir=experiment_dir, prefix=prefix, combined_report=combined_report, target_col=target_col)
    combined_report.write("</div>")
    combined_report.write("</details>")


def get_css_style() -> str:
    """
    Get the CSS style for the combined report.

    Returns:
        str: The CSS style for the combined report.
    """
    return """
        body {
            font-family: Arial, sans-serif;
            padding: 20px;
        }

        .section {
            border: 2px solid #ddd;
            padding: 10px;
            margin-bottom: 20px;
            background-color: #f9f9f9;
        }

        .section h2 {
            text-align: center;
            font-size: 24px;
            margin-bottom: 10px;
        }

        details {
            margin-bottom: 20px;
        }

        details summary {
            cursor: pointer;
            outline: none;
            font-size: 20px;
        }

        details summary::-webkit-details-marker {
            display: none;
        }

        details p {
            margin: 10px 0;
        }
    """


def calculate_metrics(y_true: pd.Series, y_score: pd.Series) -> Dict[str, Optional[float]]:
    """
    Calculate various evaluation metrics.

    Args:
        y_true (pd.Series): True labels.
        y_score (pd.Series): Predicted scores.

    Returns:
        Dict[str, float]: Dictionary containing calculated metrics.
    """
    metrics = {}
    try:
        metrics['logloss'] = log_loss(y_true, y_score)
    except Exception as e:
        logger.error(f"Failed to calculate log loss: {e}")
        metrics['logloss'] = None

    try:
        metrics['auc'] = roc_auc_score(y_true, y_score)
    except Exception as e:
        logger.error(f"Failed to calculate AUC: {e}")
        metrics['auc'] = None

    try:
        metrics['auprc'] = average_precision_score(y_true, y_score)
    except Exception as e:
        logger.error(f"Failed to calculate AUPRC: {e}")
        metrics['auprc'] = None

    return metrics


def generate_roc_curve(experiment_dir: str, fpr: List[float], tpr: List[float]) -> str:
    """
    Generate and save the ROC curve plot.

    Args:
        experiment_dir (str): Directory for the experiment.
        fpr (List[float]): List of false positive rates.
        tpr (List[float]): List of true positive rates.
        roc_plot_path (str): File path to save the ROC curve plot.

    Returns:
        None
    """
    roc_plot_path = os.path.join(experiment_dir, "roc_curve.png")
    plt.figure()
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(roc_plot_path)
    return roc_plot_path


def generate_histogram(data: pd.Series,
                       experiment_dir: str,
                       file_name: str,
                       label: str,
                       color: str,
                       title: str,
                       log_scale: bool = False) -> None:
    """
    Generate and save a histogram plot.

    Args:
        data (pd.Series): Data for the histogram.
        experiment_dir (str): Directory for the experiment.
        file_name (str): File name for the saved histogram.
        label (str): Label for the histogram.
        color (str): Color for the histogram.
        title (str): Title of the histogram plot.
        log_scale (bool): Flag to indicate if the y-axis should be in log scale.

    Returns:
        None
    """
    plt.figure()
    plt.hist(data, bins=50, alpha=0.7, color=color, label=label)
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.yscale('log' if log_scale else 'linear')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(experiment_dir, file_name))


def plot_target_distribution(target_counts: pd.Series, experiment_dir: str) -> str:
    """
    Plot and save the distribution of the 'target_col' column.

    Args:
        target_counts (pd.Series): Series containing the count of modalities in target column.
        experiment_dir (str): Directory for the experiment.

    Returns:
        str: Path to the saved plot.
    """
    counts_dict = target_counts.to_dict()
    if len(counts_dict) == 1:
        if True in counts_dict:
            counts_dict[False] = 0
        else:
            counts_dict[True] = 0

    sorted_counts = {k: counts_dict[k] for k in sorted(counts_dict)}
    labels = ['False', 'True']
    values = [sorted_counts[False], sorted_counts[True]]

    plt.figure()
    plt.bar(labels, values, color=['blue', 'red'])
    plt.yscale('log')
    plt.xlabel('target')
    plt.ylabel('Count (log scale)')
    plt.title('Distribution of target column')
    for i, value in enumerate(values):
        plt.text(i, value, str(value), ha='center', va='bottom', fontsize=12)
    plot_path = os.path.join(experiment_dir, "target_counts.png")
    plt.savefig(plot_path)
    return plot_path


def plot_histogram(data: pd.Series,
                   experiment_dir: str,
                   file_name: str,
                   label: str,
                   color: str,
                   title: str,
                   log_scale: bool = False) -> str:
    """
    Plot and save a histogram.

    Args:
        data (pd.Series): Data to be plotted.
        experiment_dir (str): Directory for the experiment.
        file_name (str): Name of the saved file.
        label (str): Label for the histogram.
        color (str): Color for the histogram.
        title (str): Title for the plot.
        log_scale (bool): Whether to use logarithmic scale.

    Returns:
        str: Path to the saved plot.
    """
    plt.figure()
    plt.hist(data, bins=50, alpha=0.7, color=color, label=label)
    if log_scale:
        plt.yscale('log')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend(loc='upper right')
    plot_path = os.path.join(experiment_dir, file_name)
    plt.savefig(plot_path)
    return plot_path


def write_report(experiment_dir: str, prefix: str, combined_report: TextIO, target_col: str) -> None:
    """
    Write the evaluation report.

    Args:
        experiment_dir (str): Report directory.
        prefix (str): Prefix of the directory to use (should be something like "val" or "test").
        combined_report (TextIO): File object for the combined report.
        target_col (str): The target column name.

    Returns:
        None
    """
    if prefix == "val":
        section_description = "Validation results for the model."
    elif prefix == "test":
        section_description = "Test results for the model."
    else:
        section_description = "Results for the model evaluation."

    combined_report.write(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px;'>")
    combined_report.write(
        f"<p>{section_description} This section presents the metrics obtained during the evaluation.</p>")

    # Reading metrics from parquet file
    metrics_df = pd.read_parquet(os.path.join(experiment_dir, prefix, 'metrics.parquet'))
    combined_report.write("<div style='text-align: center;'>")
    combined_report.write(metrics_df.to_html(index=False))
    combined_report.write("</div>")
    combined_report.write("<hr>")

    combined_report.write(f"<div style='display:flex;'>")

    # Plot ROC Curve
    combined_report.write(f"<div style='flex:50%; padding: 10px;'>")
    combined_report.write(f"<h3>ROC Curve</h3>")
    combined_report.write(f"<p>The Receiver Operating Characteristic (ROC) curve.</p>")
    combined_report.write(
        f"<img src='{os.path.relpath(os.path.join(experiment_dir, prefix, 'roc_curve.png'), experiment_dir)}' style='border: 1px solid #ddd; border-radius: 5px; max-width: 100%;'><br>")
    combined_report.write("</div>")

    # Plot Cardinality of target_col column
    combined_report.write(f"<div style='flex:50%; padding: 10px;'>")
    combined_report.write(f"<h3>Cardinality on '{target_col}'</h3>")
    combined_report.write(f"<p>The distribution of the '{target_col}' column.</p>")
    combined_report.write(
        f"<img src='{os.path.relpath(os.path.join(experiment_dir, prefix, 'target_counts.png'), experiment_dir)}' style='border: 1px solid #ddd; border-radius: 5px; max-width: 100%;'><br>")
    combined_report.write("</div>")

    combined_report.write("</div>")

    combined_report.write("<hr>")

    # Plot Distribution of Positive and Negative classes Scores
    combined_report.write(f"<div style='display:flex;'>")

    # Plot Distribution of Positive class Scores
    combined_report.write(f"<div style='flex:50%; padding: 10px;'>")
    combined_report.write(f"<h3>Positive Class Scores Distribution</h3>")
    combined_report.write(f"<p>The distribution for the positive class.</p>")
    combined_report.write(
        f"<img src='{os.path.relpath(os.path.join(experiment_dir, prefix, 'positive_distribution.png'), experiment_dir)}' style='border: 1px solid #ddd; border-radius: 5px; max-width: 100%;'><br>")
    combined_report.write("</div>")

    # Plot Distribution of Negative Class Scores
    combined_report.write(f"<div style='flex:50%; padding: 10px;'>")
    combined_report.write(f"<h3>Negative Class Scores Distribution</h3>")
    combined_report.write(f"<p>The distribution for the negative class.</p>")
    combined_report.write(
        f"<img src='{os.path.relpath(os.path.join(experiment_dir, prefix, 'negative_distribution.png'), experiment_dir)}' style='border: 1px solid #ddd; border-radius: 5px; max-width: 100%;'><br>")
    combined_report.write("</div>")

    combined_report.write("</div>")
    combined_report.write("</div>")


def model_evaluation_task(experiment_id: str,
                          report_dir: str,
                          prefix: str,
                          target_col: str,
                          predictions_data_path: str) -> str:
    """
    Evaluates the model's performance using various metrics and generates a report.

    Args:
        experiment_id (str): Unique identifier for the experiment.
        report_dir (str): Directory path to store the evaluation report.
        prefix (str): Prefix for the output folder name.
        target_col (str): The target column name.
        predictions_data_path (str): Path to the parquet predictions.

    Returns:
        str: Path to the directory containing all necessary files for generating the report.
    """
    logger.info(f"Experiment ID: {experiment_id} - Starting [model_evaluation] task...")

    predictions_data = pd.read_parquet(predictions_data_path)
    logger.info(f"Loaded '{prefix}' data successfully, input shape: {predictions_data.shape}")

    if predictions_data.empty:
        raise ValueError("The predictions_data DataFrame is empty.")

    experiment_dir = os.path.join(report_dir, experiment_id, prefix)
    os.makedirs(experiment_dir, exist_ok=True)

    y_true = predictions_data[target_col]
    y_score = predictions_data['xgb.final_score']

    metrics = calculate_metrics(y_true=y_true, y_score=y_score)

    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_score)
    generate_roc_curve(experiment_dir=experiment_dir, fpr=fpr, tpr=tpr)

    target_counts = predictions_data[target_col].value_counts()
    plot_target_distribution(target_counts=target_counts, experiment_dir=experiment_dir)

    positive = predictions_data.loc[predictions_data[target_col], 'xgb.final_score']
    negative = predictions_data.loc[~predictions_data[target_col], 'xgb.final_score']

    plot_histogram(data=positive,
                   experiment_dir=experiment_dir,
                   file_name="positive_distribution.png",
                   label="Positive Class Scores",
                   color='red',
                   title='Distribution of XGB Scores for Positive Class elements')
    plot_histogram(data=negative,
                   experiment_dir=experiment_dir,
                   file_name="negative_distribution.png",
                   label="Negative Class Scores",
                   color='blue',
                   title='Distribution of XGB Scores for Negative Class elements')

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_parquet(os.path.join(experiment_dir, 'metrics.parquet'), index=False)

    logger.info(f"Evaluation completed, saved evaluation metrics in {experiment_dir}")
    logger.info(f"Experiment ID: {experiment_id} - [model_evaluation] task done.")
    return experiment_dir
