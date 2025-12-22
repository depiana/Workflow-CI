FROM continuumio/miniconda3

WORKDIR /app

COPY MLProject/ .

RUN conda env create -f conda.yaml
SHELL ["conda", "run", "-n", "workflow_ci_env", "/bin/bash", "-c"]

ENTRYPOINT ["conda", "run", "-n", "workflow_ci_env", "python", "modelling.py"]