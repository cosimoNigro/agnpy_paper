FROM condaforge/mambaforge

RUN apt-get update -y
RUN apt-get install -y build-essential

ADD . /root/agnpy_paper

WORKDIR /root/agnpy_paper

RUN mamba env create -f agnpy_paper.yml
ENV PATH="/opt/conda/envs/agnpy_paper/bin:$PATH"
RUN echo "source activate agnpy_paper" >> ~/.bashrc
