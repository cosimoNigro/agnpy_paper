FROM condaforge/mambaforge

ARG DEBIAN_FRONTEND=noninteractive
ARG CONDA_ENV=agnpy-paper

RUN apt-get update -y
RUN apt-get install -y build-essential
# latex is needed to properly render plot labels
RUN apt-get install -y texlive-full

ADD . /root/agnpy_paper
WORKDIR /root/agnpy_paper

# install the main environment
RUN mamba env create -f agnpy_paper.yml

# back to the workdir with the paper
WORKDIR /root/agnpy_paper

# activate the environment by default
ENV PATH /opt/conda/envs/$CONDA_ENV/bin:$PATH
CMD ["/bin/bash"]
