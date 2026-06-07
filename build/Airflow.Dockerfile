FROM apache/airflow:3.2.0
USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         openjdk-17-jre-headless \
         curl \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Must match build/Spark.Dockerfile (PySpark 3.13 on driver and executors).
ENV SPARK_VERSION=3.5.2
ENV HADOOP_VERSION=3

ENV HADOOP_JAR_VERSION=3.3.4
ENV AWS_SDK_VERSION=1.12.262

RUN curl -O https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
  && tar xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz -C /opt/ \
  && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
  && ln -s /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark \
  && curl -sL https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_JAR_VERSION}/hadoop-aws-${HADOOP_JAR_VERSION}.jar -o /opt/spark/jars/hadoop-aws-${HADOOP_JAR_VERSION}.jar \
  && curl -sL https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/${AWS_SDK_VERSION}/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar -o /opt/spark/jars/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar \
  && chown -R airflow:root /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} \
  && chmod -R 755 /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} \
  && chmod +x /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}/bin/*

RUN apt-get update && apt-get install -y procps \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install --no-cache-dir \
    numpy pandas scipy plotly boto3 pyarrow \
    && python -c "import numpy"

ENV GENPM_PYSPARK_PYTHON=/usr/python/bin/python3.13
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV SPARK_HOME=/opt/spark
ENV PYSPARK_PYTHON=/usr/python/bin/python3.13
ENV PYSPARK_DRIVER_PYTHON=/usr/python/bin/python3.13
