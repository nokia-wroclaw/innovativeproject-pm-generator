FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ENV RAPIDS_VERSION=24.08.0
ENV JAVA_VERSION=17
# Must match build/Airflow.Dockerfile (driver/executor serialization).
ENV SPARK_VERSION=3.5.2
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/home/spark

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

ARG DEVCONTAINER

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python-is-python3 \
    openjdk-${JAVA_VERSION}-jre-headless \
    curl \
    wget \
    vim \
    sudo \
    whois \
    ca-certificates-java \
    procps \
    nvidia-utils-525 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH="${SPARK_HOME}/python:${SPARK_HOME}/python/lib/py4j-0.10.9.7-src.zip:${PYTHONPATH}"
ENV PATH="${SPARK_HOME}/bin:${SPARK_HOME}/python:${PATH}"

RUN SPARK_DOWNLOAD_URL="https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" \
    && echo "Downloading from: ${SPARK_DOWNLOAD_URL}" \
    && wget --verbose -O apache-spark.tgz "${SPARK_DOWNLOAD_URL}" \
    && mkdir -p /home/spark \
    && tar -xf apache-spark.tgz -C /home/spark --strip-components=1 \
    && rm apache-spark.tgz

RUN wget --verbose -O ${SPARK_HOME}/jars/rapids-4-spark_2.12-${RAPIDS_VERSION}.jar \
    "https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/${RAPIDS_VERSION}/rapids-4-spark_2.12-${RAPIDS_VERSION}.jar"
ENV HADOOP_JAR_VERSION=3.3.4
ENV AWS_SDK_VERSION=1.12.262

RUN wget --quiet -O ${SPARK_HOME}/jars/hadoop-aws-${HADOOP_JAR_VERSION}.jar https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_JAR_VERSION}/hadoop-aws-${HADOOP_JAR_VERSION}.jar \
    && wget --quiet -O ${SPARK_HOME}/jars/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/${AWS_SDK_VERSION}/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar

ARG USERNAME=hostuser
ARG USER_UID=1000
ARG USER_GID=1000

ENV APP_HOME=/home/${USERNAME}/app

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

RUN chown -R $USERNAME:$USERNAME ${SPARK_HOME} \
    && mkdir -p ${SPARK_HOME}/logs ${SPARK_HOME}/event_logs \
    && chown -R $USERNAME:$USERNAME ${SPARK_HOME}/event_logs ${SPARK_HOME}/logs \
    && chmod -R 0777 ${SPARK_HOME}/event_logs ${SPARK_HOME}/logs

RUN echo "spark.eventLog.enabled true" >> $SPARK_HOME/conf/spark-defaults.conf \
    && echo "spark.eventLog.dir file://${SPARK_HOME}/event_logs" >> $SPARK_HOME/conf/spark-defaults.conf \
    && echo "spark.history.fs.logDirectory file://${SPARK_HOME}/event_logs" >> $SPARK_HOME/conf/spark-defaults.conf \
    && echo "spark.plugins com.nvidia.spark.SQLPlugin" >> $SPARK_HOME/conf/spark-defaults.conf \
    && echo "spark.sql.execution.arrow.pyspark.enabled true" >> $SPARK_HOME/conf/spark-defaults.conf

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY build/entrypoint.sh /home/spark/entrypoint.sh
RUN chmod +x /home/spark/entrypoint.sh

USER $USERNAME
WORKDIR $APP_HOME

COPY --chown=$USERNAME:$USERNAME pyproject.toml uv.lock ./
COPY --chown=$USERNAME:$USERNAME apps/generator/ ./apps/generator/
COPY --chown=$USERNAME:$USERNAME apps/backend/pyproject.toml ./apps/backend/pyproject.toml

# PySpark requires the same minor Python on driver (Airflow) and executors (this venv).
# uv sync installs the heavy deps; then overlay the pinned genpm wheel (--no-deps) so executors run
# the exact same versioned genpm as the Airflow driver (no editable/snapshot skew).
RUN uv python install 3.12 \
    && uv sync --frozen --python 3.12 \
    && uv build --wheel --out-dir /tmp/genpm-dist ./apps/generator \
    && uv pip install --python $APP_HOME/.venv/bin/python --no-deps /tmp/genpm-dist/genpm_generator-*.whl \
    && $APP_HOME/.venv/bin/python -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version" \
    && $APP_HOME/.venv/bin/python -c "import genpm.preprocessing.__main__, genpm.raw_vis.__main__"
RUN $APP_HOME/.venv/bin/python -m ipykernel install --prefix=$APP_HOME/.venv --name=spark-env --display-name "Python (Spark Project)"

ENV PATH="$APP_HOME/.venv/bin:$PATH"
EXPOSE 4040 4041 7077 18080 8888
ENTRYPOINT ["/home/spark/entrypoint.sh"]
