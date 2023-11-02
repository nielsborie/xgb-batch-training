import os.path
import uuid
import logging
from datetime import datetime, timedelta
from random import randint, uniform
from pyspark.sql import functions as F

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, FloatType, DateType, IntegerType, BooleanType

from scripts import data_dir

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_random_dataset() -> str:
    @udf(returnType=StringType())
    def generate_uuid():
        return str(uuid.uuid4())

    @udf(returnType=FloatType())
    def generate_value(i):
        return round(10 + (990 * float(i % 10) / 10), 2)

    @udf(returnType=DateType())
    def generate_date(i):
        return datetime(2023, 1, 1) + timedelta(days=i % 365)

    @udf(returnType=FloatType())
    def generate_feature1():
        return uniform(1, 1000)

    @udf(returnType=IntegerType())
    def generate_feature2():
        return randint(3, 777)

    @udf(returnType=BooleanType())
    def generate_target_col(i):
        return bool(i % int(1 / ratio) == 0)

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

    data_size = 100000
    ratio = 0.001

    df = (spark.range(data_size)
          .withColumn("reference", generate_uuid())
          .withColumn("otherValue", generate_value("id"))
          .withColumn("datetimeCol", generate_date("id"))
          .withColumn("feature1", generate_feature1())
          .withColumn("feature2", generate_feature2())
          .withColumn("targetCol", generate_target_col("id")))

    df.show(5)
    logger.info(f"Generated randomly a dataset of shape: ({df.count(), len(df.columns)})")
    experiment_id = datetime.now().strftime("%Y%m%d%H%M%S")
    output_data_path = os.path.join(data_dir, "random_data", experiment_id)
    df.write.parquet(output_data_path)
    logger.info(f"Random dataset saved in : {output_data_path}")
    return output_data_path
