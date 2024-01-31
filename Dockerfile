FROM continuumio/miniconda3:latest
ARG DIR_WORK=/work
WORKDIR ${DIR_WORK}
ARG BASE_ENV_YML=base_environment.yml
COPY ${BASE_ENV_YML} ${DIR_WORK}/
RUN conda update --name base -c defaults conda && \
    conda env create --file ${BASE_ENV_YML}
ARG DIR_CONDA=/opt/conda
ARG VENV=stats
ENV PATH ${DIR_CONDA}/envs/${VENV}/bin:$PATH
ARG CONDA_REQ_TXT=conda_requirements.txt
COPY ${CONDA_REQ_TXT} ${DIR_WORK}/
RUN conda install --name ${VENV} -c conda-forge --file ${CONDA_REQ_TXT}
ARG PIP_REQ_TXT=pip_requirements.txt
COPY ${PIP_REQ_TXT} ${DIR_WORK}/
SHELL ["conda", "run", "--name", "stats", "/bin/bash", "-c"]
RUN pip install --upgrade pip && \
    pip install -r ${PIP_REQ_TXT}
ADD . ${DIR_WORK}/
