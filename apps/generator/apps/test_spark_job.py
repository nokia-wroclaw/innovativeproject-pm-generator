import os


from pyspark.sql import SparkSession

from genpm.utils.spark_session import minio_spark_conf


def main():
    print("Start")

    bucket_name = os.environ.get("S3_BUCKET", "test-bucket")

    builder = SparkSession.builder.appName("AirflowTestJob")
    for conf, val in minio_spark_conf().items():
        builder = builder.config(conf, val)
    spark = builder.getOrCreate()

    data = [
        ("Anna", "IT", 15000),
        ("Jan", "HR", 9000),
        ("Tomasz", "IT", 18000),
        ("Kasia", "Marketing", 12000),
    ]
    df = spark.createDataFrame(data, ["Name", "Department", "Salary"])

    print("Original DataFrame:")
    df.show()

    it_df = df.filter(df.Department == "IT")

    print("Filtered DataFrame:")
    it_df.show()

    # Construct the s3a:// path and write data
    output_path = f"s3a://{bucket_name}/test_output/it_department"
    print(f"Writing data to: {output_path}")

    # Save in Parquet format with overwrite mode (will replace files if they already exist)
    it_df.write.mode("overwrite").parquet(output_path)

    # --- NEW SECTION: PARQUET READ TEST ---
    read_path = (
        f"s3a://{bucket_name}/dummy/dummy/fbe3cdab-52d6-45f5-a98e-195d51349e8d_dummy.parquet"
    )
    print(f"\nAttempting to read file from: {read_path}")

    try:
        # Read data from the Parquet file
        dummy_df = spark.read.parquet(read_path)
        print("Successfully read the file! Here is the schema and data:")
        dummy_df.printSchema()
        dummy_df.show()
    except Exception as e:
        print(f"Failed to read the file. It might not exist or another error occurred: {e}")
    # --------------------------------------

    print("Finished successfully")
    spark.stop()


if __name__ == "__main__":
    main()
