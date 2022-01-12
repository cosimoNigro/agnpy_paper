# `agnpy`: an open-source python package modelling the radiative processes of jetted active galactic nuclei

This repository contains the scripts to generate the figures included in the paper 
"`agnpy`: an open-source python package modelling the radiative processes of jetted active galactic nuclei".

This repository is also archived in zenodo [![DOI](https://zenodo.org/badge/318151275.svg)](https://zenodo.org/badge/latestdoi/318151275)

## Content of the repository
The repository contains the following files:
```
.
├── README.md
├── agnpy_paper.yml
├── Dockerfile
├── figure_2_synchrotron_example.py
├── figure_3_u_targets_example.py
├── figure_4_absorption_example.py
├── figure_5_Mrk421_fit_gammapy.py
├── figure_5_Mrk421_fit_sherpa.py
├── figure_6_PKS1510-089_fit_gammapy.py
├── figure_6_PKS1510-089_fit_sherpa.py
├── figure_7_ssc_validation.py
├── figure_8_ec_disk_validation.py
├── figure_9_ec_blr_validation.py
├── figure_10_ec_dt_validation.py
├── figure_11_ec_cmb_validation.py
├── figure_12_tau_blr_validation.py
├── figure_13_tau_dt_validation.py
├── figure_appendix_C_sed_resolution.py
└── utils.py

```

### scripts
Each python script is named after the figure it produces in the paper, followed by a brief description of the computation performed.
Figure 5 and 6, representing the fit of two blazars multi-wavelength SEDs, can be produced either using sherpa or gammapy fitting routines.
`utils.py` contains some utility function for timing and reproduction of the reference results. 

### yaml environment
The `agnpy_paper.yml` file can be used to set-up a `conda` environment containing the same dependencies used in the paper.
It is suggested to create a new environment via
```shell
conda env create -f agnpy_paper.yml
```
and activate it 
```shell
source activate agnpy_paper
```
after activating the environment, each script can be executed via the command line with
```shell
python figure_2_synchrotron_example.py
```

## Docker container
To conserve the exact computational environment of the paper we created a Docker container.    
The container is [available on Dockerhub](https://hub.docker.com/r/cosimonigro/agnpy_paper).

### build the container yourself
After you have installed docker, to build the container using the `Dockerfile` in this repository type:
```
docker build -t agnpy_paper .
```

### run the container
To run the container:
```
docker run -it --rm agnpy_paper
```
the container shell will open with the `agnpy_paper` conda environment activated and within the `agnpy_paper` repository: users can directly run any of the scripts.

The scripts do not display any of the plots, but save them in a `figures` directory that can be generated by any of the scripts.
The easiest solution to visualise the figures produced by the scripts is to create a local `figures` repository and mount it in the container:
```
mkdir figures
docker run -it --rm -v $PWD/figures:/root/agnpy_paper/figures agnpy_paper
```
and then visualise the figures from the local filesysytem.
