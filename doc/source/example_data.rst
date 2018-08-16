.. example_data-label:

Example data
============

We have provided sample datasets for analysis with diff_classifier at the UW 
ResearchWorks Archive. These datasets were published in conjunction with the publication
"Colloidal stability as a determinant of nanoparticle behavior in the brain." These
videos show diffusion of 100nm polystyrene latex particles in 0.4% agarose gels with
varying calcium concentrations in ACSF.

Downloading example data
------------------------

The files can be downloaded with the command:

.. code-block::

  wget https://digital.lib.washington.edu/researchworks/bitstream/handle/1773/41669/0mM.7z?sequence=4&isAllowed=y

The raw files are compressed 7Z files, and must be unzipped to obtain the tiff files
for analysis. This can be done with the pzip package:

.. code-block::

  sudo apt-get install p7zip
  7za e myfiles.7z