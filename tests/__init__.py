import datetime as dt
import os.path
import unittest
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

path = Path(__file__).parent
resources_path = os.path.join(path, "resources")
test_data_dir = os.path.join(resources_path, "data")
test_reports_dir = os.path.join(resources_path, "reports")
test_models_dir = os.path.join(resources_path, "models")

class PySparkTestCase(unittest.TestCase):
    """Set-up of global test SparkSession"""

    @classmethod
    def setUpClass(cls):
        cls.spark = (SparkSession
                     .builder
                     .master("local[1]")
                     .appName("PySpark unit test")
                     .getOrCreate())

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def assert_schema_equals(self, df1: DataFrame, df2: DataFrame, check_nullable=True):
        field_list = lambda fields: (fields.name, fields.dataType, fields.nullable)
        fields1 = [*map(field_list, df1.schema.fields)]
        fields2 = [*map(field_list, df2.schema.fields)]
        if check_nullable:
            res = set(fields1) == set(fields2)
        else:
            res = set([field[:-1] for field in fields1]) == set([field[:-1] for field in fields2])
        return self.assertTrue(res)

    def assert_data_equals(self, df1: DataFrame, df2: DataFrame):
        data1 = df1.collect()
        data2 = df2.collect()
        return self.assertTrue(set(data1) == set(data2))

    @staticmethod
    def _to_datetime(datetime_as_str: str):
        return dt.datetime.strptime(datetime_as_str, '%Y-%m-%d %H:%M:%S')

    @staticmethod
    def _to_date(datetime_as_str: str):
        return dt.datetime.strptime(datetime_as_str, '%Y-%m-%d').date()
