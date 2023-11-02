import os
import tempfile
import unittest
from unittest import mock
from unittest.mock import patch, mock_open

import pandas as pd

from src.modeling.evaluation.model_evaluation_task import consolidate_reports, model_evaluation_task, calculate_metrics


class TestModelEvaluationTask(unittest.TestCase):

    @patch('src.modeling.evaluation.model_evaluation_task.pd.read_parquet')
    @patch('src.modeling.evaluation.model_evaluation_task.generate_roc_curve')
    @patch('src.modeling.evaluation.model_evaluation_task.plot_target_distribution')
    @patch('src.modeling.evaluation.model_evaluation_task.plot_histogram')
    @patch('src.modeling.evaluation.model_evaluation_task.calculate_metrics')
    def test_model_evaluation_task(self, mock_calculate_metrics, mock_plot_histogram, mock_plot_target_distribution,
                                   mock_generate_roc_curve, mock_read_parquet):
        # Create a temporary directory for the test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Given
            experiment_id = "test_experiment"
            report_dir = temp_dir
            prefix = "test"
            predictions_data_path = "path/to/the/.parquet"
            predictions_data = pd.DataFrame({
                'target_col': [False, True, False, True],
                'xgb.final_score': [0.2, 0.8, 0.4, 0.9]
            })
            mock_read_parquet.return_value = predictions_data
            mock_calculate_metrics.return_value = {'logloss': 0.815, 'auc': 1.0, 'auprc': 1.0}

            # When
            result_dir = model_evaluation_task(experiment_id=experiment_id,
                                               report_dir=report_dir,
                                               prefix=prefix,
                                               predictions_data_path=predictions_data_path,
                                               target_col="target_col")

            # Then
            # Check if the directory was created
            self.assertTrue(os.path.exists(result_dir))
            # Check if the metrics.parquet file was created
            self.assertTrue(os.path.exists(os.path.join(result_dir, 'metrics.parquet')))

            # Additional assertions
            mock_read_parquet.assert_called_once_with(predictions_data_path)
            mock_calculate_metrics.assert_called_once_with(y_true=predictions_data['target_col'],
                                                           y_score=predictions_data['xgb.final_score'])
            mock_generate_roc_curve.assert_called_once_with(experiment_dir=mock.ANY, fpr=mock.ANY, tpr=mock.ANY)
            mock_plot_target_distribution.assert_called_once_with(target_counts=mock.ANY, experiment_dir=mock.ANY)
            self.assertEqual(mock_plot_histogram.call_count, 2)

    def test_calculate_metrics(self):
        # Given
        y_true = pd.Series([False, True, False, True])
        y_score = pd.Series([0.2, 0.8, 0.4, 0.9])

        # When
        metrics = calculate_metrics(y_true=y_true, y_score=y_score)

        # Then
        # Check if the calculated logloss is as expected
        self.assertAlmostEqual(metrics['logloss'], 0.265, places=2)  # Expected logloss value
        # Check if the calculated AUC is as expected
        self.assertAlmostEqual(metrics['auc'], 1.0)  # Expected AUC value
        # Check if the calculated AUPRC is as expected
        self.assertAlmostEqual(metrics['auprc'], 1.0)  # Expected AUPRC value

    def test_model_evaluation_task_empty_data(self):
        # Create a temporary directory for the test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Given
            experiment_id = "test_experiment_empty"
            report_dir = temp_dir
            prefix = "test"
            predictions_data_path = os.path.join(temp_dir, "test_predictions_empty.parquet")
            # Create an empty DataFrame for predictions data
            predictions_data = pd.DataFrame({
                'target_col': [],
                'xgb.final_score': []
            })
            predictions_data.to_parquet(predictions_data_path)

            # When
            with self.assertRaises(ValueError) as context:
                result_dir = model_evaluation_task(experiment_id=experiment_id,
                                                   report_dir=report_dir,
                                                   prefix=prefix,
                                                   predictions_data_path=predictions_data_path,
                                                   target_col="target_col")

            # Then
            self.assertEqual("The predictions_data DataFrame is empty.", str(context.exception))


class TestConsolidateReports(unittest.TestCase):

    @patch('builtins.open', new_callable=mock_open, read_data="data")
    @patch('src.modeling.evaluation.model_evaluation_task.write_section')
    @patch('src.modeling.evaluation.model_evaluation_task.write_gantt_section')
    def test_consolidate_reports_success(self, mock_write_gantt_section, mock_write_section, mock_open):
        experiment_id = "exp_id_1"
        report_dir = "/path/to/reports"

        result = consolidate_reports(experiment_id=experiment_id, report_dir=report_dir, target_col="target_col")

        mock_open.assert_called_with('/path/to/reports/exp_id_1/combined_report.html', 'w')
        mock_write_gantt_section.assert_called_with(combined_report=mock_open(),
                                                    experiment_id=experiment_id,
                                                    gantt_plot_path="gantt_plot.png")
        mock_write_section.assert_any_call(combined_report=mock_open(),
                                           title="Validation Report",
                                           experiment_dir='/path/to/reports/exp_id_1',
                                           target_col="target_col",
                                           prefix="val")
        mock_write_section.assert_any_call(combined_report=mock_open(),
                                           title="Test Report",
                                           experiment_dir='/path/to/reports/exp_id_1',
                                           target_col="target_col",
                                           prefix="test")

        self.assertIsNotNone(result)

    @patch('builtins.open', side_effect=Exception("File not found"))
    def test_consolidate_reports_exception(self, mock_open):

        experiment_id = "exp_id_1"
        report_dir = "/path/to/reports"

        result = consolidate_reports(experiment_id=experiment_id, report_dir=report_dir, target_col="target")

        mock_open.assert_called_with('/path/to/reports/exp_id_1/combined_report.html', 'w')
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()