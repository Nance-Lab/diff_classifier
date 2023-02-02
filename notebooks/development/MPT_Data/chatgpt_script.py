import random
from pyimagej import ImageJ
from pyimagej.ijroi import IJRoi, SpotFilter

ij = ImageJ()
imp = ij.py.open("/Users/nelsschimek/Documents/nancelab/diff_classifier/notebooks/development/MPT_Data/TIFs/P10F_NT_10DIV_40nm_slice_2_midbrain_vid_2.tif")

# Create a new instance of the Settings class
settings = ij.py.run_macro("new Settings()")

# Generate random settings
settings.detector_factory = ij.py.run_macro("new DogDetectorFactory()")
settings.tracker_factory = ij.py.run_macro("new LAPTrackerFactory()")
settings.radius = random.uniform(1, 10)
settings.threshold = random.uniform(0, 1)
settings.min_spot_size = random.randint(1, 10)
settings.max_spot_size = random.randint(20, 100)

# Create an instance of the TrackMate class
trackmate = ij.py.run_macro("new TrackMate(imp, settings)")

# Create a new instance of the SpotFilter class
quality_filter = SpotFilter("Quality", "above", random.uniform(0, 1))

# Add the filter to the TrackMate instance
trackmate.add_filter(quality_filter)

# Execute the plugin
trackmate.exec()

# Get the model
model = trackmate.get_model()

# Export data to a CSV file
model.to_csv("path/to/output.csv")
