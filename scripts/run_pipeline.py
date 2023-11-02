import logging

from pyspark import SparkConf
from pyspark.sql import SparkSession

from data_generator import generate_random_dataset
from scripts import reports_dir, data_dir, models_dir
from src.modeling.evaluation.model_evaluation_task import model_evaluation_task, consolidate_reports
from src.modeling.predict.model_predict_task import model_predict_task
from src.modeling.prepare_experiment.prepare_experiment_task import prepare_ml_experiment_task
from src.modeling.training.model_training_task import model_training_task

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    conf = (SparkConf()
            .set("spark.driver.memory", "30g")
            .set("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED")
            .set('spark.sql.caseSensitive', "true"))

    spark = (SparkSession.builder
             .appName("example")
             .master("local[3]")
             .config(conf=conf)
             .getOrCreate())

    spark.sparkContext.setLogLevel("ERROR")

    data_path = generate_random_dataset()

    df = spark.read.parquet(data_path)

    train_data, val_data, test_data, experiment_id = prepare_ml_experiment_task(data=df,
                                                                                target_col="targetCol",
                                                                                datetime_col="datetimeCol",
                                                                                start_date="2023-01-01",
                                                                                end_date="2023-12-31",
                                                                                val_size_weeks=4,
                                                                                test_size_weeks=8,
                                                                                report_dir=reports_dir)
    params = {
        'max_depth': 1,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'logloss',
        'lambda': 1,
        'alpha': 0,
        'scale_pos_weight': 1,
        'gamma': 0,
        'min_child_weight': 1,
        'nthread': 6
    }

    model_path = model_training_task(experiment_id=experiment_id,
                                     train_data=train_data,
                                     predictors=["feature1", "feature2"],
                                     target_column="targetCol",
                                     batch_size=5000,
                                     params=params,
                                     model_dir=models_dir
                                     )

    train_data.unpersist()
    test_predictions_path = model_predict_task(experiment_id=experiment_id,
                                               data_dir=data_dir,
                                               prefix="test",
                                               batch_size=5000,
                                               xgb_model_path=model_path,
                                               data=test_data,
                                               predictors=["feature1", "feature2"],
                                               additional_columns=["otherValue", "id", "datetimeCol"],
                                               target_column="targetCol"
                                               )
    test_data.unpersist()
    val_predictions_path = model_predict_task(experiment_id=experiment_id,
                                              data_dir=data_dir,
                                              prefix="val",
                                              batch_size=5000,
                                              xgb_model_path=model_path,
                                              data=val_data,
                                              predictors=["feature1", "feature2"],
                                              additional_columns=["otherValue", "id", "datetimeCol"],
                                              target_column="targetCol"
                                              )
    val_data.unpersist()

    # Generate report for validation data
    val_report_dir = model_evaluation_task(experiment_id=experiment_id,
                                           report_dir=reports_dir,
                                           prefix="val",
                                           target_col="targetCol",
                                           predictions_data_path=val_predictions_path)

    # Generate report for test data
    test_report_dir = model_evaluation_task(experiment_id=experiment_id,
                                            report_dir=reports_dir,
                                            prefix="test",
                                            target_col="targetCol",
                                            predictions_data_path=test_predictions_path)

    # Consolidate the reports
    combined_report_path = consolidate_reports(experiment_id=experiment_id, report_dir=reports_dir, target_col="targetCol")
    spark.stop()
