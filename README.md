# Treasure Hunt Analyses

Analysis repository for Treasure Hunt.

## Overview

This repository analyses data from the Treasure Hunt task, a spatial-navigation memory task, with data from implanted microwires.

## Requirements

This repository requires Python >= 3.7.

As well as typical scientific Python packages, dependencies include:
- [pynwb](https://github.com/NeurodataWithoutBorders/pynwb)
- [convnwb](https://github.com/JacobsSU/convnwb)
- [spiketools](https://github.com/spiketools/spiketools)
- [pingouin](https://github.com/raphaelvallat/pingouin)

The full list of dependencies is listed in `requirements.txt`.

## Repository Layout

This repository is set up in the following way:
- `code/` contains custom code and utilities for doing the analyses
- `figures/` contains saved output figures
- `notebooks/` contains notebooks for exploring analyses
- `scripts/` contains stand alone scripts

## Data

The datasets analyzed in this project are from human subjects with implanted microwires.

Data notes:
- Datasets for this project are organized into the [NWB](https://www.nwb.org/) format.
- Basic preprocessing and data conversion is done in the [ConvertTH](https://github.com/JacobsSU/ConvertTH) repository.
- Spike sorting, to isolate putative single-neurons, has been performed on this data.
