import os

from xgboost.core import XGBoostError

from src.modeling.predict.model_predict_task import model_predict_task
from tests import PySparkTestCase, resources_path, test_data_dir


class TestModelPredictTask(PySparkTestCase):
    xgb_model_path = os.path.join(resources_path, "models", "exp1", "xgb_trained.model")

    def test_model_predict_task_with_invalid_xgb_model_path(self):
        with self.assertRaises(XGBoostError):
            data = self.spark.createDataFrame([(1, 0.1, '2023-01-01', 0)], ["id", "someValue", "datetimeCol", "target"])
            model_predict_task("exp1", "prefix", "invalid_model_path", data, ["someValue"], "target", 2, "data")

    def test_model_predict_task_with_invalid_data(self):
        with self.assertRaises(ValueError):
            model_predict_task("exp1", "prefix", "xgb_model_path", None, ["someValue"], "target", 2, "data")

    def test_model_predict_task_with_missing_target_column(self):
        with self.assertRaises(ValueError):
            data = self.spark.createDataFrame([(1, 0.1, '2023-01-01')], ["id", "someValue", "datetimeCol"])
            model_predict_task("exp1", "prefix", "xgb_model_path", data, ["someValue"], "label", 2, "data")
            model_predict_task(experiment_id="exp1",
                               prefix="prefix",
                               xgb_model_path=self.xgb_model_path,
                               data=df,
                               predictors=["feature1", "feature2"],
                               target_column="target",
                               batch_size=2,
                               additional_columns=["id", "someValue", "datetimeCol"],
                               data_dir="data")

    def test_model_predict_task_predictions(self):
        # Given
        data = [("id1", 100, "2022-10-10", 1, 0.1, 0),
                ("id2", 100, "2022-10-10", 2, 0.2, 1),
                ("id3", 100, "2022-10-10", 3, 0.3, 0),
                ("id4", 100, "2022-10-11", 4, 0.4, 1),
                ("id5", 100, "2022-10-12", 5, 0.5, 0)]
        df = self.spark.createDataFrame(data, ["id", "someValue", "datetimeCol", "feature1", "feature2", "target"])

        # When
        output_folder = model_predict_task(experiment_id="exp1",
                                           prefix="prefix",
                                           xgb_model_path=self.xgb_model_path,
                                           data=df,
                                           predictors=["feature1", "feature2"],
                                           target_column="target",
                                           batch_size=2,
                                           additional_columns=["id", "someValue", "datetimeCol"],
                                           data_dir=test_data_dir)

        # Then
        self.assertTrue(os.path.exists(output_folder))
