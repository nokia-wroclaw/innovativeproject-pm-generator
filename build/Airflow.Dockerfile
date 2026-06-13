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

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Must match build/Spark.Dockerfile (PySpark 3.12 on driver and executors).
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

# genpm is installed as a package; runtime mount overlays /opt/airflow/generator for dev.
COPY apps/generator /opt/airflow/generator
RUN mkdir -p /opt/airflow/artifacts \
    && python3 -c "import shutil; shutil.make_archive('/opt/airflow/artifacts/genpm', 'zip', '/opt/airflow/generator', 'genpm')" \
    && chown -R airflow:root /opt/airflow/generator /opt/airflow/artifacts

# Venv must be created as the airflow runtime user — root-owned uv Python is not executable
# by AIRFLOW_UID (50000), which causes "Permission denied" in Spark PythonRunner.
USER airflow
RUN uv venv /opt/airflow/genpm-venv --python 3.12 \
    && uv pip install --python /opt/airflow/genpm-venv/bin/python --no-cache \
        numpy pandas scipy plotly boto3 pyarrow python-dotenv setuptools \
    && uv pip install --python /opt/airflow/genpm-venv/bin/python --no-cache \
        -e /opt/airflow/generator --no-deps \
    && PYTHONPATH="/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.7-src.zip" \
        /opt/airflow/genpm-venv/bin/python -c "import distutils; import genpm.utils.spark_session"

ENV GENPM_PYSPARK_PYTHON=/opt/airflow/genpm-venv/bin/python
ENV GENPM_PY_FILES=/opt/airflow/artifacts/genpm.zip
ENV GENPM_GENERATOR_ROOT=/opt/airflow/generator
# Make the (runtime-mounted) genpm package importable by the scheduler / dag-processor / worker
# python at DAG parse time. The build-time genpm.zip is only a fallback; DAGs rebuild it from the
# live mount at submit time so driver and cluster executors run identical code.
ENV PYTHONPATH=/opt/airflow/generator
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV SPARK_HOME=/opt/spark
ENV PYSPARK_DRIVER_PYTHON=/opt/airflow/genpm-venv/bin/python
