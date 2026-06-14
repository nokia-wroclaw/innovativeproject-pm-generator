# Background Script Runners

Two scripts for running Python jobs inside the Spark container **without keeping a terminal open**.

## Prerequisites

1. The Spark container must be running — start it with:
   ```bash
   ./tools/bin/start_env.sh
   ```
2. Docker must be installed and your user must have permission to run `docker exec`.
3. The Python file you want to run must exist inside the container at:
   ```
   /home/<USER>/app/apps/generator/<your_file.py>
   ```
   (this maps to `apps/generator/` in the repo via the volume mount)

---

## Scripts

### `run_training_bg.sh` — run a training job in the background

```bash
./tools/bin/run_training_bg.sh -f genpm/modelling/run_training.py
```

### `run_check_bg.sh` — run a check/validation job in the background

```bash
./tools/bin/run_check_bg.sh -f genpm/preprocessing/run.py
```

Both scripts accept the same flag:

| Flag | Description |
|------|-------------|
| `-f` / `--file` | Python file to run, **relative to `apps/generator`** inside the container |
| `-h` / `--help` | Show usage |

---

## Monitoring a running job

Both scripts print the log file path on startup. Use these commands to monitor:

```bash
# stream live logs
docker exec <USER>-genpm-spark tail -f /tmp/training_<timestamp>.log

# check if still running
docker exec <USER>-genpm-spark pgrep -af python3

# stop the job
docker exec <USER>-genpm-spark pkill -f <your_file.py>
```

Replace `<USER>` with your Linux username (e.g. `miklep2163`).
