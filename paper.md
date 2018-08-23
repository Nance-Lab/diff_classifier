---
title: 'diff_classifier: Parallelization of multi-particle tracking videos'
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

Multi-particle tracking (MPT) is a powerful analytical tool that has been used
in fields ranging from aeronautics to oceanography to biomedical engineering
[@Pulford:2005]. However, MPT methods currently resemble more an art than a
science. MPT can be broken down into two essential components: feature detection
and frame linking. Different tracking algorithms rely on a variety of methods
for both steps. A recent competition open to the particle tracking community
challenged participants to independently apply their self-developed methods and
algorithms to a shared dataset capturing a variety of scenarios, e.g. Brownian
motion similar to vesicles in the cytoplasm and directed motion such as
microtubule transport [@Chenouard:2014]. It was found that no method performed
near perfectly, and factors impacting image quality during acquisition (e.g.
non-uniform background, polydispersity, and photobleaching) all hampered
accuracy.

One aspect of particle tracking that has yet to be fully solved is the element
of bias introduction by human selection of tracking parameters. Most tracking
packages require some human input in both particle detection and linking steps,
and these vary depending on image quality, particle properties, and acquisition
setup. Not only does this introduce potential sources of error, but it can also
significantly slow down analysis workflows if human input is required for each
video. Users must choose between potential error in results by using constant
parameters across all videos or high time-to-first-results that scale with the
number of videos to be analyzed.

In this paper, we present ``diff_classifier`` as a toolset to address the
scale-up of nanoparticle tracking analysis using the combination of
parallelization techniques, regression to predict tracking parameters, and
downstream analysis and visualization tools to minimize time-to-first-result.

The source code for ``diff_classifier`` has been archived to Zenodo with the
linked DOI: [@zenodo]

# Acknowledgements

The authors would like to thank the eScience Institute for the resources and
expertise provided through the Incubator Program that made ``diff_classifier``
possible.

# References
