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

# initialise conda and the environment
RUN conda init bash

# install the specific tag of jetset used for the comparison plots
WORKDIR /root
RUN echo "conda activate agnpy-paper" > ~/.bashrc
RUN pip install wget
RUN git clone https://github.com/andreatramacere/jetset-installer.git && \
    cd jetset-installer/ && \
    python jetset_installer.py 1.2.0rc10

# note jetset uses the latest iminuit while gammapy-0.18.2 uses iminuit<2, downgrade it
RUN conda install iminuit=1.5.4

# back to the workdir with the paper
WORKDIR /root/agnpy_paper

# activate the environment by default
ENV PATH /opt/conda/envs/$CONDA_ENV/bin:$PATH
CMD /bin/bash -c "source activate $CONDA_ENV"
