# Introduction

When implementing the Fiji package with Jython, I needed to figure out how option in the GUI interface in Fiji
corresponded to the variables and actions in the python script.  This is a working attempt at doing so. It
appears that different steps in the GUI are organized into separate packages corresponding to folders in the
trackmate package available on GitHub (https://github.com/fiji/TrackMate/releases/tag/TrackMate_-3.5.3). 

# Detection
First, the user must import the desired detection method, referred to as a Factory in the code. I identified
the following options:

- BlockLogDetectorFactory
- DogDetectorFactory
- DownsampleLogDetectorFactory
- LogDetectorFactory
- ManualDetectorFactory
- SpotDetectorFactory

Note that all these options are selected in the GUI interface when prepping spot detection.  Downstream
steps in detection will depend on the method selected, so I will continue with the LogDetector method,
since the example script uses this method.

Examining the LogDetectorFactory java file, it appears that the different user inputs required for analysis
are contained in the Detector Keys objects.  The following objects must be defined:

- KEY__DO_MEDIAN_FILTERING
- KEY_DO_SUBPIXEL_LOCALIZATION
- KEY_RADIUS
- KEY_TARGET_CHANNEL
- KEY_THRESHOLD

# Feature Filter
This was the one I wanted to be sure to get into the nitty-gritty, because the example script only includes
a single feature filter, quality.  What if you want to include multiple features?  These features are
included in individual Factory files in the features/spot folder.

**Intensity Analyzer Factory**
- MEAN_INTENSITY
- MEDIAN_INTENSITY
- MIN_INTENSITY
- MAX_INTENSITY
- TOTAL_INTENSITY
- STANDARD_DEVIATION

**Morphology Analyzer Factory**
- ELLIPSOIDFIT_SEMIAXISLENGTH_C
- ELLIPSOIDFIT_SEMIAXISLENGTH_B
- ELLIPSOIDFIT_SEMIAXISLENGTH_A
- ELLIPSOIDFIT_AXISPHI_C
- ELLIPSOIDFIT_AXISPHI_B
- ELLIPSOIDFIT_AXISPHI_A
- ELLIPSOIDFIT_AXISTHETA_C
- ELLIPSOIDFIT_AXISTHETA_B
- ELLIPSOIDFIT_AXISTHETA_A

**Spot Radius Estimator Factory**
- ESTIMATED_DIAMETER

# Tracking
