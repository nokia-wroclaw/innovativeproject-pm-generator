# PM Generator
Performance Management Generator project

## How to set up profile on VM:
sudo useradd -m -s /bin/bash \<login from PWR AD> <br>
sudo usermod -aG docker \<profile name> <br>
sudo chown -R \<profile name>:<profile name> /home/\<profile name>/.docker <br>

## How to route a docker prots through OUR port on vm
run `./bin/start_env.sh` and paste the command filled with your ports


### Dependencies
Backend FastAPI uses uv.

### How to add new dependencies to the project
# Runtime dependency:
docker compose exec fastapi sh -c "cd /app && uv add <package>"

# Dev dependency:
docker compose exec fastapi sh -c "cd /app && uv add --dev <package>"


### Ruff and MyPy scripts
Use simple scripts from bin:
- ./bin/be_ruff.sh            # lint
- ./bin/be_ruff.sh --fix      # lint + autofix
- ./bin/be_mypy.sh
- ./bin/be_verify.sh          # ruff + mypy
