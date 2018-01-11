# diff-classifier

This project aims to take existing analyses of nanoparticle diffusion that I have worked on previously and expand the analysis to additional features.  This project was done in conjunction with the eScience Institute at the University of Washington.

# Objective 1

Develop a package to extract trajectory features from high-speed video of
nanoparticles in the brain and to classify trajectories based on their
diffusive behavior.  This package will be implemented primarily with Python,
but will call on ImageJ packages including TrackMate and TrajClassifier.  We
will use this package to analyze existing datasets of nanoparticle diffusion in
the healthy developing brain.  We will automate this process to minimize
computation time and increase reproducibility, and we will implement a data
storage protocol for datasets in the future.

# Objective 2

Create regional maps of diffusive behavior based on the trajectory features.  
Features calculated in Objective 1, including efficiency, asymmetry, kurtosis,
and trappedness, will be visualized spatially to examine regional differences
in tissue structure.  Differences will be analyze between regions in the brain
including the cortex, hippocampus, and hypothalamus.  
