# Treasure Hunt Analyses

Analysis repository for Treasure Hunt.

[![Paper](https://img.shields.io/badge/DOI-10.1002/hipo.23539-informational.svg)](https://doi.org/10.1002/hipo.23539)

## Overview

This repository analyses data from the Treasure Hunt task, a spatial-navigation memory task, with data from implanted microwires.

## Reference

This project is described in the following paper:

    Donoghue T, Cao R, Han CZ, Holman CM, Brandmeir NJ, Wang S, Jacobs J (2023). Single neurons in the human medial 
    temporal lobe flexibly shift representations across spatial and memory tasks. Hippocampus. DOI: 10.1002/hipo.23539

Direct Link: https://doi.org/10.1002/hipo.23539

## Requirements

This repository requires Python >= 3.7.

As well as typical scientific Python packages, dependencies include:
- [pynwb](https://github.com/NeurodataWithoutBorders/pynwb)
- [convnwb](https://github.com/JacobsSU/convnwb)
- [spiketools](https://github.com/spiketools/spiketools)

The full list of dependencies is listed in `requirements.txt`.

## Repository Layout

This repository is set up in the following way:
- `code/` contains custom code and utilities for doing the analyses
- `notebooks/` contains notebooks for exploring analyses
- `scripts/` contains stand alone scripts

## Data

The datasets analyzed in this project are from human subjects with implanted microwires.

Data notes:
- Datasets for this project are organized into the [NWB](https://www.nwb.org/) format.
- Basic preprocessing and data conversion is done in the [ConvertTH](https://github.com/JacobsSU/ConvertTH) repository.
- Spike sorting, to isolate putative single-neurons, has been performed on this data.
