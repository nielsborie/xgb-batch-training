from src.modeling.training.model_training_task import model_training_task, add_batch_column
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


class TestAddBatchColumn(PySparkTestCase):

    def test_add_batch_column_with_larger_dataset(self):
        # Given
        data = [(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e"), (6, "f")]
        df = self.spark.createDataFrame(data, ["id", "features"])
        batch_size = 2

        # When
        df_with_batch = add_batch_column(df, batch_size)

        # Then
        self.assertEqual(df_with_batch.filter(df_with_batch["batch"] == 0).count(), 2)
        self.assertEqual(df_with_batch.filter(df_with_batch["batch"] == 1).count(), 2)
        self.assertEqual(df_with_batch.filter(df_with_batch["batch"] == 2).count(), 2)

    def test_add_batch_column_with_smaller_dataset(self):
        # Given
        data_small = [(1, "a"), (2, "b")]
        df_small = self.spark.createDataFrame(data_small, ["id", "features"])
        batch_size_small = 5

        # When
        df_with_batch_small = add_batch_column(df_small, batch_size_small)

        # Then
        self.assertEqual(df_with_batch_small.filter(df_with_batch_small["batch"] == 0).count(), 2)

    def test_add_batch_column_with_uneven_dataset(self):
        # Given
        data_uneven = [(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e"), (6, "f"), (7, "g")]
        df_uneven = self.spark.createDataFrame(data_uneven, ["id", "features"])
        batch_size_uneven = 3

        # When
        df_with_batch_uneven = add_batch_column(df_uneven, batch_size_uneven)

        # Then
        self.assertEqual(df_with_batch_uneven.filter(df_with_batch_uneven["batch"] == 0).count(), 3)
        self.assertEqual(df_with_batch_uneven.filter(df_with_batch_uneven["batch"] == 1).count(), 3)
        self.assertEqual(df_with_batch_uneven.filter(df_with_batch_uneven["batch"] == 2).count(), 1)