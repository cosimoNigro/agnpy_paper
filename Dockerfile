FROM condaforge/mambaforge

ARG DEBIAN_FRONTEND=noninteractive
ARG CONDA_ENV=agnpy-paper

RUN apt-get update -y
RUN apt-get install -y build-essential
# latex is needed to properly render plot labels
RUN apt-get install -y dvipng texlive-latex-extra texlive-fonts-recommended cm-super

ADD . /root/agnpy_paper
WORKDIR /root/agnpy_paper

# install the main environment
RUN mamba env create -f agnpy_paper.yml
RUN echo "source activate agnpy-paper" >> ~/.bashrc
ENV PATH /opt/conda/envs/$CONDA_ENV/bin:$PATH

# define work dir
WORKDIR /root/agnpy_paper
ENV DOCKER_INSIDE "True"

# activate the environment by default
CMD ["/bin/bash"]
