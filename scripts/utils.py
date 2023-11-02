from pyspark.sql import SparkSession

from src.modeling.training.model_training_task import model_training_task


def get_dummy_trained_model():
    spark = (SparkSession
             .builder
             .master("local[1]")
             .appName("PySpark unit test")
             .getOrCreate())
    # Create a sample DataFrame
    data = [("id1", 1, 0.1, False),
            ("id2", 2, 0.2, False),
            ("id3", 3, 0.3, False),
            ("id4", 4, 0.4, True),
            ("id5", 5, 0.5, False)]
    df = spark.createDataFrame(data, ["id", "feature1", "feature2", "target"])

    # Define the parameters
    params = {"max_depth": 3}

    # Call the function for incremental training
    model_path = model_training_task("exp1", df, ["feature1", "feature2"], "target", 2, params, "./models")
    print(f"Created and saved a dummy model in {model_path}")
    spark.stop()

if __name__ == "__main__":
    get_dummy_trained_model()