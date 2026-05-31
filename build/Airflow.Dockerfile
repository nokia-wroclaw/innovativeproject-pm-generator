FROM apache/airflow:3.2.0
USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         openjdk-17-jre-headless \
         curl \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
ENV SPARK_VERSION=3.5.2
ENV HADOOP_VERSION=3
RUN curl -O https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
  && tar xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz -C /opt/ \
  && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
  && ln -s /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark \
  && chown -R airflow:root /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} \
  && chmod -R 755 /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} \
  && chmod +x /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}/bin/*

RUN apt-get update && apt-get install -y python3.11 python3.11-venv python3.11-dev \
  && apt-get clean && rm -rf /var/lib/apt/lists/*


USER airflow
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV SPARK_HOME=/opt/spark