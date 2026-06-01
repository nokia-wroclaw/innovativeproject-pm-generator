import getpass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# that's a more robust approach
USER = getpass.getuser()
# USER = os.getenv("USER")

SHARED_DIR_PATH = Path(f"/home/{USER}/app/apps/apps/generator/data/shared_dir")
RAW_DATASET_PATH = SHARED_DIR_PATH / "eda_data/raw_pm_data"

SPARK_CHECKPOINT_PATH = SHARED_DIR_PATH / "tmp" / "checkpoints"

# /home/sparkuser/app/apps/apps/generator/data/shared_dir/tmp/checkpoints

SPARK_CONFIGS = {
    # ============================================================
    # 1. FULL RESOURCES
    # Uses maximum safe capacity of the VM (30 cores, 110GB RAM).
    # Leaves 2 cores and 16GB RAM for the OS to prevent freezing.
    # Best for: Massive jobs, full pipeline runs.
    # ============================================================
    "FULL_RESOURCES": {
        "spark.master": "local[30]",
        "spark.driver.memory": "110g",
        "spark.memory.fraction": "0.80",
        "spark.memory.storageFraction": "0.30",
        "spark.sql.shuffle.partitions": "512",
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.sql.files.maxPartitionBytes": "256MB",
        "spark.sql.autoBroadcastJoinThreshold": "100MB",
        "spark.driver.extraJavaOptions": "-XX:+UseG1GC",
        "spark.sql.execution.arrow.pyspark.enabled": "true",

        # Disable RAPIDS
        "spark.plugins": "",
        "spark.rapids.sql.enabled": "false",
        "spark.kryo.registrator": "",
    },
    # ============================================================
    # 2. ONLY HALF
    # Uses ~50% of the VM (16 cores, 55GB RAM).
    # Best for: Development, experimentation, or running two
    # moderate workloads concurrently.
    # ============================================================
    "HALF_SAFE": {
        "spark.master": "local[16]",
        "spark.driver.memory": "55g",
        "spark.memory.fraction": "0.75",
        "spark.memory.storageFraction": "0.30",
        "spark.sql.shuffle.partitions": "256",
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.sql.files.maxPartitionBytes": "128MB",
        "spark.sql.autoBroadcastJoinThreshold": "50MB",
        "spark.driver.extraJavaOptions": "-XX:+UseG1GC",
        "spark.sql.execution.arrow.pyspark.enabled": "true",

        # Disable RAPIDS
        "spark.plugins": "",
        "spark.rapids.sql.enabled": "false",
        "spark.kryo.registrator": "",
    },
    # ============================================================
    # 3. STANDARD (A FIFTH)
    # Uses ~20% of the VM (6 cores, 22GB RAM).
    # Best for: Small daily tasks, data exploration, light ETL,
    # or allowing multiple users on the same VM.
    # ============================================================
    "STANDARD_FIFTH": {
        "spark.master": "local[6]",
        "spark.driver.memory": "22g",
        "spark.memory.fraction": "0.70",
        "spark.memory.storageFraction": "0.30",
        "spark.sql.shuffle.partitions": "128",
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.sql.files.maxPartitionBytes": "128MB",
        "spark.sql.autoBroadcastJoinThreshold": "25MB",
        "spark.driver.extraJavaOptions": "-XX:+UseG1GC",
        "spark.sql.execution.arrow.pyspark.enabled": "true",
    },
    # ============================================================
    # 4. WINDOW HEAVY (Full Resources Tuned)
    # Scaled to your 126GB/32C VM with high execution memory
    # and massive shuffle partitions to reduce sort chunk sizes.
    # ============================================================
    "WINDOW_HEAVY": {
        "spark.master": "local[30]",
        "spark.driver.memory": "110g",
        "spark.memory.fraction": "0.85",  # Prioritize execution over cache
        "spark.memory.storageFraction": "0.20",
        "spark.sql.shuffle.partitions": "1024",  # Very high to prevent OOM during sorts
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.sql.files.maxPartitionBytes": "128MB",
        "spark.driver.extraJavaOptions": "-XX:+UseG1GC",
        "spark.sql.execution.arrow.pyspark.enabled": "true",

        # Disable RAPIDS
        "spark.plugins": "",
        "spark.rapids.sql.enabled": "false",
        "spark.kryo.registrator": "",
    },
    # ============================================================
    # 5. JOIN HEAVY (Full Resources Tuned)
    # High broadcast thresholds to bypass shuffles entirely
    # when joining dimensional tables to massive fact tables.
    # ============================================================
    "JOIN_HEAVY": {
        "spark.master": "local[30]",
        "spark.driver.memory": "110g",
        "spark.memory.fraction": "0.80",
        "spark.memory.storageFraction": "0.25",
        "spark.sql.shuffle.partitions": "768",
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.sql.autoBroadcastJoinThreshold": "250MB",  # Aggressive broadcasting
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.sql.files.maxPartitionBytes": "256MB",
        "spark.driver.extraJavaOptions": "-XX:+UseG1GC",
        "spark.sql.execution.arrow.pyspark.enabled": "true",

        # Disable RAPIDS
        "spark.plugins": "",
        "spark.rapids.sql.enabled": "false",
        "spark.kryo.registrator": "",
    },
    # ============================================================
    # 6. AGGREGATION HEAVY (Full Resources Tuned)
    # Balanced memory execution and high shuffles to handle
    # massive groupBys, rollups, and skewness.
    # ============================================================
    "AGG_HEAVY": {
        "spark.master": "local[30]",
        "spark.driver.memory": "110g",
        "spark.memory.fraction": "0.82",
        "spark.memory.storageFraction": "0.20",
        "spark.sql.shuffle.partitions": "640",
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.sql.files.maxPartitionBytes": "256MB",
        "spark.driver.extraJavaOptions": "-XX:+UseG1GC",
        "spark.sql.execution.arrow.pyspark.enabled": "true",

        # Disable RAPIDS
        "spark.plugins": "",
        "spark.rapids.sql.enabled": "false",
        "spark.kryo.registrator": "",
    },
    "RAPIDS": {
        "spark.master": "local[16]",
        "spark.driver.memory": "48g",
        "spark.memory.fraction": "0.75",
        "spark.memory.storageFraction": "0.30",
        "spark.sql.shuffle.partitions": "256",

        # RAPIDS plugin
        "spark.plugins": "com.nvidia.spark.SQLPlugin",
        "spark.rapids.sql.enabled": "true",

        # Kryo
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.kryo.registrator": "com.nvidia.spark.rapids.GpuKryoRegistrator",

        # AQE
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.sql.execution.arrow.pyspark.enabled": "true",

        # Setting up GPU
        "spark.rapids.sql.concurrentGpuTasks": "6",
        "spark.rapids.memory.pinnedPool.size": "8g",
        "spark.rapids.sql.explain": "NOT_ON_GPU",

        "spark.rapids.sql.batchSizeBytes": "536870912",

        "spark.rapids.memory.gpu.pooling.enabled": "true",
        "spark.rapids.memory.gpu.allocFraction": "0.90",
        "spark.rapids.memory.gpu.reserve": "2g",
    }
}

# Grouping KPIs based on agg character
MEAN_LIKE_UNITS = ["%", "bit/s", "kbit/s", "Mbit/s", "ms", "#/s", "#/h"]
VOLUME_UNITS = ["#"]

MIN_KEYWORDS = ["min", "minimal", "minimum"]

MAX_KEYWORDS = ["max", "maximal", "maximum"]

AVG_KEYWORDS = ["avg", "average"]

# RATIOS: telecom KPI acronyms / categories that are usually percentages
RATIO_KEYWORDS = [
    "cssr",  # Call Setup Success Rate
    "hosr",  # Handover Success Rate
    "asr",  # Answer Seizure Ratio
    "ccr",  # Call Completion Ratio / related completion KPIs
    "dcr",  # Drop Call Ratio
    "bler",  # Block Error Rate
    "fer",  # Frame Error Rate
    "per",  # Packet Error Rate
    "availability",
    "accessibility",
    "retainability",
    "integrity",
    "utilization",
    "Average Time",
    "Average Duration",
]

# MEAN-LIKE: speed / quality / radio level / delay measurements
MEAN_LIKE_KEYWORDS = [
    "throughput",
    "latency",
    "jitter",
    "rtt",
    "rssi",
    "rsrp",
    "rsrq",
    "sinr",
    "snr",
    "mos",
]

# VOLUME: additive traffic / count style telecom nouns
VOLUME_KEYWORDS = [
    "erlang",
    "mou",
    "bytes",
    "octets",
    "attempts",
    "packets",
    "Total Time",
    "Total Duration",
    "volume",
]

MAX_IMPUTABLE_GAP = 6
