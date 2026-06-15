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

# genpm source is kept in the image for: building the wheel, DAG-parse-time imports (via PYTHONPATH),
# and the SparkSubmit launcher scripts under apps/. The dev compose mounts over it for live edits.
COPY apps/generator /opt/airflow/generator
RUN chown -R airflow:root /opt/airflow/generator

# Venv must be created as the airflow runtime user — root-owned uv Python is not executable
# by AIRFLOW_UID (50000), which causes "Permission denied" in Spark PythonRunner.
USER airflow
# Build genpm as a pinned wheel and install it WITH its dependencies into the Spark driver venv.
# We install the wheel's full dependency graph (not a hand-curated list) so the version pins from
# apps/generator/pyproject.toml apply — otherwise bare `pip install numpy pandas ...` pulls latest
# (pandas 3.x / numpy 2.4) which breaks tsgm/statsmodels. pyspark is also pulled in but is shadowed
# at runtime by SPARK_HOME on PYTHONPATH (see genpm.utils.spark_bootstrap). setuptools provides
# distutils on Python 3.12.
RUN uv venv /opt/airflow/genpm-venv --python 3.12 \
    && uv build --wheel --out-dir /opt/airflow/artifacts /opt/airflow/generator \
    && uv pip install --python /opt/airflow/genpm-venv/bin/python --no-cache \
        setuptools /opt/airflow/artifacts/genpm_generator-*.whl \
    && PYTHONPATH="/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.7-src.zip" \
        /opt/airflow/genpm-venv/bin/python -c "import distutils; import genpm.utils.spark_session" \
    && /opt/airflow/genpm-venv/bin/python -c "import genpm.modelling.generate_s3"

ENV GENPM_PYSPARK_PYTHON=/opt/airflow/genpm-venv/bin/python
ENV GENPM_GENERATOR_ROOT=/opt/airflow/generator
# Make the genpm package importable by the scheduler / dag-processor / worker python at DAG parse
# time. /opt/airflow/generator holds the COPYed source in prod and the live mount in dev.
# GENPM_PY_FILES is intentionally unset: prod uses the installed wheel; the dev compose sets it to
# turn on the --py-files live-code override for executors.
ENV PYTHONPATH=/opt/airflow/generator
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV SPARK_HOME=/opt/spark
ENV PYSPARK_DRIVER_PYTHON=/opt/airflow/genpm-venv/bin/python
