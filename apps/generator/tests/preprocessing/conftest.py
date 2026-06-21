import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder.master("local[1]")
        .appName("genpm-preprocessing-tests")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        # Disable the RAPIDS GPU plugin: it is installed on the shared cluster but
        # does not support local-mode Spark 3.5.2 and would crash SparkContext init.
        .config("spark.plugins", "")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
