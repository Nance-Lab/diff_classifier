---
title: 'diff\_classifier: Parallelization of multi-particle tracking video analyses'
tags:
- Python
- multi-particle tracking
- Amazon Web Services
- parallelization
authors:
- name: Chad Curtis
  orcid: 0000-0001-6312-392X
  affiliation: "1"
- name: Ariel Rokem
  orcid: 0000-0003-0679-1985
  affiliation: "2"
- name: Elizabeth Nance
  orcid: 0000-0001-7167-7068
  affiliation: "1"
affiliations:
- name: Department of Chemical Engineering, University of Washington
  index: 1
- name: eScience Institute, University of Washington
  index: 2
date: 20 August 2018
bibliography: paper.bib
---

# Summary

The [diff_classifier](https://github.com/ccurtis7/diff_classifier) package seeks
to address the issue of scale-up in multi-particle tracking (MPT) analyses via a
parallelization approach. MPT is a powerful analytical tool that has been used
in fields ranging from aeronautics to oceanography [@Pulford:2005] allowing
researchers to collect spatial and velocity information of moving objects from
video datasets. Examples include:

* Tracking tracers in ocean currents to study fluid flow
* Tracking molecular motors (e.g. myosin, kinesin) to assess motile activity
* Measuring intracellular trafficking by tracking membrane vesicles
* Assessing microrheological properties by tracking nanoparticle movement.

While a variety of tracking algorithms are available to researchers
[@Chenouard:2014], a common problem is that data analysis usually depends on the
use of graphical user interfaces, and relies on human input for accurate
tracking. For example, particle detection often relies on the selection of a
quality threshold, a numerical quantity distinguishing between “real” particles
and “fake” particles [@Tineves:2017]. If this threshold is too high, false
positive trajectories result in skewed MSD profiles, and in extreme cases, cause
the code to crash due to a lack of convergence in the particle linking step. If
the threshold is too low, trajectories will be cut short resulting in a bias
towards short fast-moving trajectories and could result in empty datasets
[@Wang:2015].

Due to variations in experimental conditions and image quality, user-selected
tracking parameters can vary widely from video to video. As parameter selection
can also vary from user to user, this also brings up the issue of
reproducibility. diff_classifier addresses these issues with regression
tools to predict input tracking parameters and parallelized script-based
implementations in Amazon Web Services (AWS), using the Simple Storage
Service (S3) and Batch for data storage and computing, respectively, and
relying on the Cloudknot software library for automating these
interactions [@cloudknot]. By manually tracking a small subset of the entire
video dataset to be analyzed (5-10 videos per experiment), users can predict
tracking parameters based on intensity distributions of input images. This can
simultaneously reduce time-to-first-result in MPT workflows and provide
reproducible MPT results.

diff_classifier also includes downstream MPT analysis tools including mean
squared displacement and feature calculations, visualization tools, and a
principle component analysis implementation. MPT is commonly used to calculate
and report ensemble-averaged diffusion coefficients of nanoparticles and other
objects. We sought to expand the power of MPT analyses by changing the unit of
analysis to individual particle trajectories. By including a variety of features
(e.g. aspect ratio, boundedness, fractal dimension), with trajectory-level
resolution, users can implement a range of data science analysis techniques to
their MPT datasets.

The source code for diff_classifier has been archived to Zenodo with the
linked DOI: [@zenodo]


# Acknowledgements

The authors would like to thank the eScience Institute for the resources
and expertise provided through the Incubator Program that made
diff_classifier possible. The University of Washington eScience Institute
is supported through a grant from the Gordon & Betty Moore Foundation and
the Alfred P. Sloan Foundation. The authors would also like to thank
funding from the National Institute of General Medical Sciences 1R35
GM124677-01 (E. Nance).

# References
