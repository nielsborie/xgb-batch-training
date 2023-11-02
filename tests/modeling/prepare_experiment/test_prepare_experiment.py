import os
import tempfile
import unittest
from datetime import datetime

from src.modeling.prepare_experiment.prepare_experiment_task import prepare_ml_experiment_task, plot_gantt
from tests import PySparkTestCase


class TestPrepareMLExperimentTask(PySparkTestCase):

    def get_data(self):
        return self.spark.createDataFrame([
            ("2023-01-01", False, "id1"),
            ("2023-01-02", False, "id2"),
            ("2023-02-03", True, "id3"),
            ("2023-03-04", True, "id4"),
        ], ["datetimeCol", "targetCol", "id"])

    def get_empty_data(self):
        return self.spark.createDataFrame([], "datetimeCol: string, targetCol: boolean, id: string")

    def test_prepare_ml_experiment_task_with_given_dates(self):
        # Given
        start_date = "2023-01-01"
        end_date = "2023-01-04"

        # When/Then
        with self.assertRaises(ValueError) as context:
            prepare_ml_experiment_task(data=self.get_data(),
                                       target_col="targetCol",
                                       datetime_col="datetimeCol",
                                       start_date=start_date,
                                       end_date=end_date,
                                       val_size_weeks=1,
                                       test_size_weeks=1,
                                       report_dir="some/path/to/report/dir")

        self.assertEqual("No data found in the train dataset. Check your parameters !", str(context.exception))

    def test_prepare_ml_experiment_task_with_empty_data(self):
        # Given
        empty_data = self.get_empty_data()

        # When/Then
        with self.assertRaises(ValueError) as context:
            prepare_ml_experiment_task(data=empty_data,
                                       target_col="targetCol",
                                       datetime_col="datetimeCol",
                                       val_size_weeks=1,
                                       test_size_weeks=1,
                                       report_dir="some/path/to/report/dir")

        self.assertEqual("No data found in the input dataset.", str(context.exception))

    def test_prepare_ml_experiment_task_with_no_start_date(self):
        # Given
        end_date = "2023-01-04"

        # When/Then
        with self.assertRaises(ValueError) as context:
            prepare_ml_experiment_task(data=self.get_data(),
                                       target_col="targetCol",
                                       datetime_col="datetimeCol",
                                       report_dir="some/path/to/report/dir",
                                       end_date=end_date,
                                       val_size_weeks=1,
                                       test_size_weeks=1)

        self.assertEqual("No data found in the train dataset. Check your parameters !", str(context.exception))

    def test_prepare_ml_experiment_task_with_no_end_date(self):
        # Given
        start_date = "2023-01-01"

        # When/Then
        with self.assertRaises(ValueError) as context:
            prepare_ml_experiment_task(data=self.get_data(),
                                       start_date=start_date,
                                       target_col="targetCol",
                                       datetime_col="datetimeCol",
                                       report_dir="some/path/to/report/dir",
                                       val_size_weeks=1,
                                       test_size_weeks=1)

        self.assertEqual("No data found in the validation dataset. Check your parameters !", str(context.exception))

    def test_prepare_ml_experiment_task_with_invalid_dates(self):
        # Given
        start_date = "2023-01-05"
        end_date = "2023-01-06"

        # When/Then
        with self.assertRaises(ValueError) as context:
            prepare_ml_experiment_task(data=self.get_data(),
                                       target_col="targetCol",
                                       datetime_col="datetimeCol",
                                       report_dir="some/path/to/report/dir",
                                       start_date=start_date,
                                       end_date=end_date,
                                       val_size_weeks=1,
                                       test_size_weeks=1)

        self.assertEqual("No data found in the train dataset. Check your parameters !", str(context.exception))

    def test_prepare_ml_experiment_task_with_zero_weeks(self):
        # Given
        start_date = "2023-01-01"
        end_date = "2023-01-04"

        # When/Then
        with self.assertRaises(ValueError) as context:
            prepare_ml_experiment_task(data=self.get_data(),
                                       target_col="targetCol",
                                       datetime_col="datetimeCol",
                                       report_dir="some/path/to/report/dir",
                                       start_date=start_date,
                                       end_date=end_date,
                                       val_size_weeks=0,
                                       test_size_weeks=0)

        self.assertEqual("No data found in the validation dataset. Check your parameters !", str(context.exception))

    def test_prepare_ml_experiment_task_with_good_data(self):
        # Given
        # When
        with tempfile.TemporaryDirectory() as temp_dir:
            train_data, val_data, test_data, experiment_id = prepare_ml_experiment_task(data=self.get_data(),
                                                                                        target_col="targetCol",
                                                                                        datetime_col="datetimeCol",
                                                                                        start_date=None,
                                                                                        end_date=None,
                                                                                        val_size_weeks=4,
                                                                                        test_size_weeks=4,
                                                                                        report_dir=temp_dir)

            # Then
            expected_train_transfer_ids = ["id1", "id2"]
            expected_val_transfer_ids = ["id3"]
            expected_test_transfer_ids = ["id4"]

            self.assertListEqual(train_data.select("id").rdd.flatMap(lambda x: x).collect(),
                                 expected_train_transfer_ids)
            self.assertListEqual(val_data.select("id").rdd.flatMap(lambda x: x).collect(),
                                 expected_val_transfer_ids)
            self.assertListEqual(test_data.select("id").rdd.flatMap(lambda x: x).collect(),
                                 expected_test_transfer_ids)

class TestPlotGantt(unittest.TestCase):

    def test_plot_gantt(self):
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            report_dir = os.path.join(temp_dir, "test_reports")
            experiment_id = datetime.now().strftime("%Y%m%d%H%M%S")
            start_dates = [datetime(2023, 1, 1), datetime(2023, 1, 5), datetime(2023, 1, 10)]
            end_dates = [datetime(2023, 1, 4), datetime(2023, 1, 9), datetime(2023, 1, 15)]
            sizes = [100, 150, 200]
            periods = ["Period 1", "Period 2", "Period 3"]

            # When
            plot_gantt(report_dir=report_dir,
                       experiment_id=experiment_id,
                       start_dates=start_dates,
                       end_dates=end_dates,
                       sizes=sizes,
                       periods=periods)

            # Then
            expected_file_path = os.path.join(report_dir, experiment_id, "gantt_plot.png")
            self.assertTrue(os.path.exists(expected_file_path))
