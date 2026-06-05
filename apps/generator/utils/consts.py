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


# ADDITIONAL SPARK CONFIGS:
SPARK_CONFIGS = {
    # ============================================================
    # FULL MACHINE UTILIZATION
    # Heavy joins, windows, massive groupBy, feature engineering
    # Uses almost entire VM
    #
    # Best for:
    # - huge shuffles
    # - heavy window operations
    # - repeated joins
    # - large aggregations
    # - long lineage pipelines
    #
    # Risk:
    # - can pressure OS if workload spikes hard
    # ============================================================
    "FULL_HEAVY": {
        # ----------------------------
        # CPU
        # ----------------------------
        "spark.master": "local[30]",  # leave 1-2 cores for OS
        # ----------------------------
        # MEMORY
        # ----------------------------
        "spark.driver.memory": "90g",
        # Fraction of heap used for execution + storage
        # Higher = better for shuffles/windows/groupBy
        "spark.memory.fraction": "0.80",
        # Fraction reserved for cached data
        # Lower = more room for execution/shuffles
        "spark.memory.storageFraction": "0.30",
        # ----------------------------
        # SHUFFLES
        # ----------------------------
        # IMPORTANT:
        # Higher values help:
        # - joins
        # - groupBy
        # - windows
        # - repartition-heavy jobs
        #
        # Lower values help:
        # - tiny datasets
        # - avoid scheduler overhead
        #
        # For 32 cores:
        # 512 is a strong heavy-workload value
        "spark.sql.shuffle.partitions": "512",
        # Adaptive query execution
        "spark.sql.adaptive.enabled": "true",
        # Dynamically reduce tiny partitions
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        # Helps skewed joins enormously
        "spark.sql.adaptive.skewJoin.enabled": "true",
        # ----------------------------
        # SERIALIZATION
        # ----------------------------
        # MUCH better than Java serializer
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        # ----------------------------
        # FILE SCAN TUNING
        # ----------------------------
        # Larger values:
        # - fewer tasks
        # - larger memory pressure
        #
        # Smaller values:
        # - more tasks
        # - safer memory
        #
        # 256MB good for large VM
        "spark.sql.files.maxPartitionBytes": "256MB",
        # ----------------------------
        # WINDOW/JOIN OPTIMIZATION
        # ----------------------------
        # Better broadcast handling
        "spark.sql.autoBroadcastJoinThreshold": "100MB",
        # ----------------------------
        # JVM GC
        # ----------------------------
        # Important for large heaps
        "spark.driver.extraJavaOptions": "-XX:+UseG1GC",
        # ----------------------------
        # ARROW
        # ----------------------------
        # Faster pandas conversion
        "spark.sql.execution.arrow.pyspark.enabled": "true",

        # Disable RAPIDS
        "spark.plugins": "",
        "spark.rapids.sql.enabled": "false",
        "spark.kryo.registrator": "",
    },
    # ============================================================
    # HALF MACHINE UTILIZATION
    #
    # Safer for:
    # - development
    # - experimentation
    # - unstable workloads
    # - avoiding OOM
    #
    # Leaves significant headroom
    # ============================================================
    "HALF_SAFE": {
        "spark.master": "local[16]",
        "spark.driver.memory": "48g",
        "spark.memory.fraction": "0.75",
        "spark.memory.storageFraction": "0.30",
        # Lower shuffle count
        # good for medium workloads
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
    # WINDOW HEAVY
    #
    # Best for:
    # - row_number
    # - lag/lead
    # - rolling windows
    # - ordered TS operations
    #
    # Window operations are:
    # - sort heavy
    # - shuffle heavy
    # - memory heavy
    #
    # More partitions reduces OOM risk
    # ============================================================
    "WINDOW_HEAVY": {
        "spark.master": "local[30]",
        "spark.driver.memory": "96g",
        # More execution memory
        "spark.memory.fraction": "0.85",
        "spark.memory.storageFraction": "0.20",
        # IMPORTANT:
        # More partitions => smaller sort chunks
        # huge help for windows
        "spark.sql.shuffle.partitions": "768",
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        # Smaller partition files reduce sort memory
        "spark.sql.files.maxPartitionBytes": "128MB",
        "spark.driver.extraJavaOptions": "-XX:+UseG1GC",
        "spark.sql.execution.arrow.pyspark.enabled": "true",

        # Disable RAPIDS
        "spark.plugins": "",
        "spark.rapids.sql.enabled": "false",
        "spark.kryo.registrator": "",
    },
    # ============================================================
    # JOIN HEAVY
    #
    # Best for:
    # - many joins
    # - repeated joins
    # - dimensional enrichment
    # - star-schema workloads
    #
    # Join operations are:
    # - shuffle heavy
    # - skew sensitive
    # ============================================================
    "JOIN_HEAVY": {
        "spark.master": "local[30]",
        "spark.driver.memory": "90g",
        "spark.memory.fraction": "0.80",
        "spark.memory.storageFraction": "0.25",
        # Large shuffle partition count
        "spark.sql.shuffle.partitions": "768",
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        # IMPORTANT:
        # Larger broadcast threshold can massively
        # speed up dimensional joins
        "spark.sql.autoBroadcastJoinThreshold": "250MB",
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
    # AGGREGATION HEAVY
    #
    # Best for:
    # - groupBy
    # - rollups
    # - KPI aggregation
    # - cube operations
    #
    # Aggregations are:
    # - shuffle heavy
    # - skew sensitive
    # ============================================================
    "AGG_HEAVY": {
        "spark.master": "local[30]",
        "spark.driver.memory": "90g",
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
