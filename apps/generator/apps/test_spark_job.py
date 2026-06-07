import os

from pyspark.sql import SparkSession


def main():
    print("Start")

    # Fetch MinIO configuration from environment variables defined in docker-compose
    s3_endpoint = os.environ.get("S3_URL", "http://minio:9000")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "your_default_access_key")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "your_default_secret_key")
    bucket_name = os.environ.get("S3_BUCKET", "test-bucket")

    # Initialize Spark session with appropriate parameters for MinIO
    spark = (
        SparkSession.builder.appName("AirflowTestJob")
        .config("spark.hadoop.fs.s3a.endpoint", s3_endpoint)
        .config("spark.hadoop.fs.s3a.access.key", access_key)
        .config("spark.hadoop.fs.s3a.secret.key", secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .getOrCreate()
    )

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
