from unittest import mock
from unittest.mock import patch, Mock

import numpy as np

from src.modeling.training.model_training_task import model_training_task
from tests import PySparkTestCase


class TestModelTrainingTask(PySparkTestCase):
    def test_model_training_task_with_invalid_params(self):
        with self.assertRaises(ValueError):
            model_training_task("exp1", None, None, "target", -1, None, "models")

    def test_model_training_task_with_invalid_data(self):
        with self.assertRaises(ValueError):
            model_training_task("exp1", None, [], "target", 1000, None, "models")

    def test_model_training_task_with_missing_target_column(self):
        with self.assertRaises(ValueError):
            df = self.spark.createDataFrame([(1, "a"), (2, "b"), (3, "c")], ["transferId", "features"])
            model_training_task("exp1", df, ["features"], "label", 1000, None, "models")

    def test_model_training_task_incremental_training(self):
        # Create a sample DataFrame
        data = [("id1", 1, 0.1, False),
                ("id2", 2, 0.2, False),
                ("id3", 3, 0.3, False),
                ("id4", 4, 0.4, True),
                ("id5", 5, 0.5, False)]
        df = self.spark.createDataFrame(data, ["transferId", "feature1", "feature2", "target"])

        # Define the parameters
        params = {"max_depth": 3}

        # Call the function for incremental training
        model_path = model_training_task("exp1", df, ["feature1", "feature2"], "target", 2, params, "models")

        # Assert that the model path is not None
        self.assertTrue(model_path is not None)

    @patch('xgboost.train')
    @patch('xgboost.DMatrix')
    def test_model_training_task_with_mocked_data(self, mock_dmatrix, mock_train):
        # Given
        data = [("id1", 1, 0.1, False),
                ("id2", 2, 0.2, False),
                ("id3", 3, 0.3, False),
                ("id4", 4, 0.4, True),
                ("id5", 5, 0.5, False)]
        df = self.spark.createDataFrame(data, ["transferId", "feature1", "feature2", "target"])

        # Define the parameters
        params = {"max_depth": 3}

        # Mock the xgb.train method
        mock_xgb_model = Mock()
        mock_xgb_model.save_model.return_value = None
        mock_train.return_value = mock_xgb_model

        # When
        model_training_task("exp1", df, ["feature1", "feature2"], "target", 2, params, "models")

        # Then
        # Verify that xgb_model.save_model is called
        mock_xgb_model.save_model.assert_called_once_with("models/exp1/xgb_trained.model")

        # Verify that each row is used exactly once
        rows_used = set()
        for call_args in mock_dmatrix.call_args_list:
            _, kwargs = call_args
            X_train = kwargs['data']
            rows_used.update(set(X_train[:, 0]))  # Assuming 'feature1' is in the DataFrame

        # Verify that each row is used exactly once
        expected_rows_used = set(df.select('feature1').rdd.flatMap(lambda x: x).collect())
        self.assertEqual(expected_rows_used, rows_used)

        # Verify that xgb_model.save_model is called
        mock_xgb_model.save_model.assert_called_once_with("models/exp1/xgb_trained.model")
