## `agnpy`: an open-source python package modelling the radiative processes of jetted active galactic nuclei

This repository contains the scripts to generate the figures included in the paper 
"`agnpy`: an open-source python package modelling the radiative processes of jetted active galactic nuclei".

This repository is also archived in zenodo [![DOI](https://zenodo.org/badge/318151275.svg)](https://zenodo.org/badge/latestdoi/318151275)

## content of the repository
The repository contains the following files
```
.
├── agnpy_paper.yml
├── figure_2_synchrotron_example.py
├── figure_3_ec_example.py
├── figure_4_u_targets_example.py
├── figure_5_absorption_example.py
├── figure_6_Mrk421_fit_gammapy.py
├── figure_6_Mrk421_fit_sherpa.py
├── figure_7_PKS1510-089_fit_gammapy.py
├── figure_7_PKS1510-089_fit_sherpa.py
├── figure_8_ssc_validation.py
├── figure_9_ec_disk_validation.py
├── figure_10_ec_blr_validation.py
├── figure_11_ec_dt_validation.py
├── figure_12_ec_cmb_validation.py
├── figure_13_tau_blr_validation.py
├── figure_14_tau_dt_validation.py
├── appendix_B_absorption_resolution.py
├── appendix_B_sed_resolution.py
├── README.md
└── utils.py

```

### scripts
Each python script is named after the figure it produces in the paper, followed by a brief description of the computation performed.
Figure 6 and 7, representing the fit of two blazars multi-wavelength SEDs, can be produced either using sherpa or gammapy fitting routines.
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
after activating the environment each script can be executed via the command line with
```shell
python figure_2_synchrotron_example.py
```

## Docker container
To conserve the exact workflow of the paper we create a Docker container

TODO: make the docker container

