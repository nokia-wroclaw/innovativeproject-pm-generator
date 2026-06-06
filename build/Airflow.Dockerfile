FROM apache/airflow:3.2.0
USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         openjdk-17-jre-headless \
         curl \
         procps \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

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

RUN apt-get update && apt-get install -y python3.11 python3.11-venv python3.11-dev \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Dedicated venv for Spark jobs (system python3.11 has no pip on Debian).
RUN python3.11 -m venv /opt/genpm-venv \
  && /opt/genpm-venv/bin/pip install --no-cache-dir --upgrade pip \
  && /opt/genpm-venv/bin/pip install --no-cache-dir \
    pyspark==3.5.2 \
    python-dotenv \
    pyarrow \
    ruptures \
    scipy \
    pyyaml \
    pandas \
  && chown -R airflow:root /opt/genpm-venv \
  && mkdir -p /tmp/genpm-spark-checkpoints \
  && chown airflow:root /tmp/genpm-spark-checkpoints

USER airflow
RUN pip install --no-cache-dir \
    pyspark==3.5.2 \
    python-dotenv \
    pyarrow \
    ruptures \
    scipy \
    pyyaml \
    pandas
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV SPARK_HOME=/opt/spark
ENV PYTHONPATH=/opt/genpm/generator
ENV PYSPARK_PYTHON=/opt/genpm-venv/bin/python
ENV PYSPARK_DRIVER_PYTHON=/opt/genpm-venv/bin/python
ENV GENPM_SPARK_CHECKPOINT_DIR=/tmp/genpm-spark-checkpoints