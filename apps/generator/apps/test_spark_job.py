from pyspark.sql import SparkSession


def main():
    print("start")
    spark = SparkSession.builder \
        .appName("AirflowTestJob") \
        .getOrCreate()

    data = [
        ("Anna", "IT", 15000),
        ("Jan", "HR", 9000),
        ("Tomasz", "IT", 18000),
        ("Kasia", "Marketing", 12000)
    ]
    df = spark.createDataFrame(data, ["Imie", "Dzial", "Pensja"])

    df.show()

    it_df = df.filter(df.Dzial == "IT")

    print("\nFiltered")
    it_df.show()

    print("Finished")

    spark.stop()


if __name__ == "__main__":
    main()