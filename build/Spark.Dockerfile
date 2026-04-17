FROM python:3.11-slim-bookworm

ENV JAVA_VERSION=17
ENV SPARK_VERSION=3.5.2
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/home/spark

ARG DEVCONTAINER

RUN apt-get update && apt-get install -y \
    openjdk-${JAVA_VERSION}-jre-headless \
    curl \
    wget \
    vim \
    sudo \
    whois \
    ca-certificates-java \
    procps \
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


ARG USERNAME=sparkuser

ARG SPARK_GROUP=sparkusers
ARG SPARK_GROUP_GID=1005
ENV APP_HOME=/home/${USERNAME}/app

RUN groupadd --gid 1000 $USERNAME \
    && groupadd --gid $SPARK_GROUP_GID $SPARK_GROUP \
    && useradd --uid 1000 --gid 1000 -m -s /bin/bash -G $SPARK_GROUP $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

RUN chown -R $USERNAME:$USERNAME ${SPARK_HOME} \
    && mkdir -p ${SPARK_HOME}/logs ${SPARK_HOME}/event_logs \
    && chown -R $USERNAME:$USERNAME ${SPARK_HOME}/event_logs ${SPARK_HOME}/logs \
    && chmod -R 0777 ${SPARK_HOME}/event_logs ${SPARK_HOME}/logs

RUN echo "spark.eventLog.enabled true" >> $SPARK_HOME/conf/spark-defaults.conf \
    && echo "spark.eventLog.dir file://${SPARK_HOME}/event_logs" >> $SPARK_HOME/conf/spark-defaults.conf \
    && echo "spark.history.fs.logDirectory file://${SPARK_HOME}/event_logs" >> $SPARK_HOME/conf/spark-defaults.conf

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY build/entrypoint.sh /home/spark/entrypoint.sh
RUN chmod +x /home/spark/entrypoint.sh

USER $USERNAME
WORKDIR $APP_HOME

COPY --chown=$USERNAME:$USERNAME pyproject.toml uv.lock ./

COPY --chown=$USERNAME:$USERNAME apps/generator/ ./apps/generator/

RUN uv sync --frozen
RUN $APP_HOME/.venv/bin/python -m ipykernel install --prefix=$APP_HOME/.venv --name=spark-env --display-name "Python (Spark Project)"

ENV PATH="$APP_HOME/.venv/bin:$PATH"
EXPOSE 4040 4041 18080 8888
ENTRYPOINT ["/home/spark/entrypoint.sh"]
