# Treasure Hunt Analyses

Analysis repository for Treasure Hunt.

[![Paper](https://img.shields.io/badge/paper-10.1002/hipo.23539-informational.svg)](https://doi.org/10.1002/hipo.23539)
[![Data](https://img.shields.io/badge/data-ConvertTH-lightgrey)](https://github.com/HSUPipeline/ConvertTH)
[![Template](https://img.shields.io/badge/template-HSUPipeline/AnalyzeTEMPLATE-yellow.svg)](https://github.com/HSUPipeline/AnalyzeTEMPLATE)

## Overview

This repository analyses data from the Treasure Hunt (TH) task, a spatial-navigation memory task, with data from implanted microwires.

## Project Guide

You can step through all the analyses and results in this project by stepping through the notebooks folders.

Note that some parts of the project, including running analyses across the whole dataset,
are done by stand alone scripts, which are available in the `scripts` folder.

## Reference

The project is described in the following paper:

    Donoghue, T., Cao, R., Han, C. Z., Holman, C. M., Brandmeir, N. J., Wang, S., & Jacobs, J. (2023). Single
    neurons in the human medial temporal lobe flexibly shift representations across spatial and memory tasks.
    Hippocampus, 33(5), 600–615. DOI: 10.1002/hipo.23539

Direct Link: https://doi.org/10.1002/hipo.23539

## Requirements

This repository requires Python >= 3.7.

As well as typical scientific Python packages, dependencies include:
- [pynwb](https://github.com/NeurodataWithoutBorders/pynwb)
- [convnwb](https://github.com/HSUPipeline/convnwb)
- [spiketools](https://github.com/spiketools/spiketools)

The full list of dependencies is listed in `requirements.txt`.

## Repository Layout

This repository is set up in the following way:
- `code/` contains custom code and utilities for doing the analyses
- `notebooks_overlap/` contains notebooks that step through analyses combining both tasks
- `notebooks_th/` contains notebooks that step through analyses of the TH task
- `scripts/` contains stand alone scripts

## Data

The datasets analyzed in this project are from human subjects with implanted microwires.

Data notes:
- Datasets for this project are organized into the [NWB](https://www.nwb.org/) format.
- Basic preprocessing and data conversion is done in the [ConvertTH](https://github.com/HSUPipeline/ConvertTH) repository.
- Spike sorting, to isolate putative single-neurons, has been performed on this data.
