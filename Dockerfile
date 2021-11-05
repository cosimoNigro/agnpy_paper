FROM condaforge/mambaforge

RUN apt-get update -y
RUN apt-get install -y build-essential

ADD . /root/agnpy_paper

WORKDIR /root/agnpy_paper

# install the main environment
RUN mamba env create -f agnpy_paper.yml
# install the specific tag of jetset used for the comparison plots
WORKDIR /root
RUN git clone https://github.com/andreatramacere/jetset-installer.git && \
    cd jetset-installer/ && \
    python jetset_installer.py 1.2.0rc10

# back to the workdir with the paper
WORKDIR /root/agnpy_paper
# activate the environment by default
ENV PATH="/opt/conda/envs/agnpy_paper/bin:$PATH"
RUN echo "source activate agnpy-paper" >> ~/.bashrc
