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

# Spot Feature Filter
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

**Spot Contrast Analyzer Factory**
- CONTRAST

**Spot Contrast and SNR Analyzer Factory**
- CONTRAST
- SNR (Signal/Noise Ratio)

# Tracking
Similar to the detection step, all the different tracking methods are contained in Factory files.  I identified the following methods currently available:

- SimpleSparseLAPTrackerFactory
- SarseLAPTrackerFactory
- FastLAPTrackerFactory
- LAPTrackerFactory
- SimpleFastLAPTrackerFactory
- KalmanTrackerFactory
- NearestNeighborTrackerFactory
- ManualTrackerFactory
- SpotTrackerFactory

All parameters for LAP Tracker-based methods are contained in the LAPTrackerFactor file:

- KEY_ALLOW_GAP_CLOSING
- KEY_ALLOW_TRACK_MERGING
- KEY_ALLOW_TRACK_SPLITTING
- KEY_ALTERNATIVE_LINKING_COST_FACTOR
- KEY_BLOCKING_VALUE
- KEY_CUTOFF_PERCENTILE
- KEY_GAP_CLOSING_FEATURE_PENALTIES
- KEY_GAP_CLOSING_MAX_DISTANCE
- KEY_GAP_CLOSING_MAX_FRAME_GAP
- KEY_LINKING_FEATURE_PENALTIES
- KEY_LINKING_MAX_DISTANCE
- KEY_MERGING_FEATURE_PENALTIES
- KEY_MERGING_MAX_DISTANCE
- KEY_SPLITTING_FEATURE_PENALTIES
- KEY_SPLITTING_MAX_DISTANCE

# Track Feature Filter
After tracking has been completed, the user can filter trajectories based on the properties of the trajectories themselves, not individual spots.  These features are again contained in Factory files in features/track folder.

**Track Spot Quality Feature Analyzer**
- TRACK_MEAN_QUALITY
- TRACK_MAX_QUALITY
- TRACK_MIN_QUALITY
- TRACK_MEDIAN_QUALITY
- TRACK_STD_QUALITY

**Track Speed Statistics Analyzer**
- Velocity
- TRACK_MEAN_SPEED
- TRACK_MAX_SPEED
- TRACK_MIN_SPEED
- TRACK_MEDIAN_SPEED
- TRACK_STD_SPEED

**Track Location Analyzer**
- Track Location
- TRACK_X_LOCATION
- TRACK_Y_LOCATION
- TRACK_Z_LOCATION

**Track Index Analyzer**
- TRACK_INDEX
- TRACK_ID

**Track Duration Analyzer**
- Track duration
- TRACK_DURATION
- TRACK_START
- TRACK_STOP
- TRACK_DISPLACEMENT

**Track Branching Analyzer**
- Branching analyzer
- NUMBER_GAPS
- LONGEST_GAP
- NUMBER_SPLITS
- NUMBER_MERGES
- NUMBER_COMPLEX
- NUMBER_SPOTS

# Edge Features
There aren't any examples including filtering based on edge features, and I don't remember seeing these in the GUI.  However, I will include them here.  Maybe it would be easy to include them in a script, if desired.  I'm not sure what an edge is really atm.

**Edge Velocity Analyzer**
- Edge velocity
- VELOCITY
- DISPLACEMENT

**Edge Time Location Analyzer**
- Edge mean location
- EDGE_TIME
- EDGE_X_LOCATION
- EDGE_Y_LOCATION
- EDGE_Z_LOCATION

**Edge Target Analyzer**
- Edge target
- SPOT_SOURCE_ID
- SPOT_TARGET_ID
- LINK_COST

Finally, I was able to locate the export as xml file in the actions directory, ExportTracksToXML.java


