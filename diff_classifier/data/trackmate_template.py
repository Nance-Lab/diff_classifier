from ij import IJ, ImagePlus, ImageStack
import fiji.plugin.trackmate.Settings as Settings
import fiji.plugin.trackmate.Model as Model
import fiji.plugin.trackmate.SelectionModel as SelectionModel
import fiji.plugin.trackmate.TrackMate as TrackMate
import fiji.plugin.trackmate.Logger as Logger
import fiji.plugin.trackmate.detection.DetectorKeys as DetectorKeys
import fiji.plugin.trackmate.detection.DogDetectorFactory as DogDetectorFactory
import fiji.plugin.trackmate.tracking.sparselap.SparseLAPTrackerFactory as SparseLAPTrackerFactory
import fiji.plugin.trackmate.tracking.LAPUtils as LAPUtils
import fiji.plugin.trackmate.visualization.hyperstack.HyperStackDisplayer as HyperStackDisplayer
import fiji.plugin.trackmate.features.FeatureFilter as FeatureFilter
import fiji.plugin.trackmate.features.FeatureAnalyzer as FeatureAnalyzer
import fiji.plugin.trackmate.features.spot.SpotContrastAndSNRAnalyzerFactory as SpotContrastAndSNRAnalyzerFactory
import fiji.plugin.trackmate.action.ExportStatsToIJAction as ExportStatsToIJAction
import fiji.plugin.trackmate.io.TmXmlReader as TmXmlReader
import fiji.plugin.trackmate.action.ExportTracksToXML as ExportTracksToXML
import fiji.plugin.trackmate.io.TmXmlWriter as TmXmlWriter
import fiji.plugin.trackmate.features.ModelFeatureUpdater as ModelFeatureUpdater
import fiji.plugin.trackmate.features.SpotFeatureCalculator as SpotFeatureCalculator
import fiji.plugin.trackmate.features.spot.SpotContrastAndSNRAnalyzer as SpotContrastAndSNRAnalyzer
import fiji.plugin.trackmate.features.spot.SpotIntensityAnalyzerFactory as SpotIntensityAnalyzerFactory
import fiji.plugin.trackmate.features.track.TrackSpeedStatisticsAnalyzer as TrackSpeedStatisticsAnalyzer
import fiji.plugin.trackmate.features.track.TrackDurationAnalyzer as TrackDurationAnalyzer
import fiji.plugin.trackmate.util.TMUtils as TMUtils


# Get currently selected image
#imp = WindowManager.getCurrentImage()
imp = IJ.openImage({target_file})
#imp = IJ.openImage('/home/ubuntu/data/RED_nPEG_37C_pH72_S1_1_1_2.tif')
#imp.show()


#-------------------------
# Instantiate model object
#-------------------------

model = Model()

# Set logger
model.setLogger(Logger.IJ_LOGGER)

#------------------------
# Prepare settings object
#------------------------

settings = Settings()
settings.setFrom(imp)

# Configure detector
settings.detectorFactory = DogDetectorFactory()
settings.detectorSettings = {{
    DetectorKeys.KEY_DO_SUBPIXEL_LOCALIZATION : True,
    DetectorKeys.KEY_RADIUS : 2.5,
    DetectorKeys.KEY_TARGET_CHANNEL : 1,
    DetectorKeys.KEY_THRESHOLD : 5.,
    DetectorKeys.KEY_DO_MEDIAN_FILTERING : False,
}}

# Configure tracker
settings.trackerFactory = SparseLAPTrackerFactory()
settings.trackerSettings = LAPUtils.getDefaultLAPSettingsMap()
settings.trackerSettings['LINKING_MAX_DISTANCE'] = 10.0
settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE']=10.0
settings.trackerSettings['MAX_FRAME_GAP']= 3

# Add the analyzers for some spot features.
# You need to configure TrackMate with analyzers that will generate
# the data you need.
# Here we just add two analyzers for spot, one that computes generic
# pixel intensity statistics (mean, max, etc...) and one that computes
# an estimate of each spot's SNR.
# The trick here is that the second one requires the first one to be in
# place. Be aware of this kind of gotchas, and read the docs.
settings.addSpotAnalyzerFactory(SpotIntensityAnalyzerFactory())
settings.addSpotAnalyzerFactory(SpotContrastAndSNRAnalyzerFactory())

filter2 = FeatureFilter('QUALITY', 10, True)
settings.addSpotFilter(filter2)
filter3 = FeatureFilter('MEDIAN_INTENSITY', 10, True)
settings.addSpotFilter(filter3)
filter4 = FeatureFilter('SNR', 0.5, True)
settings.addSpotFilter(filter4)


# Add an analyzer for some track features, such as the track mean speed.
settings.addTrackAnalyzer(TrackSpeedStatisticsAnalyzer())
settings.addTrackAnalyzer(TrackDurationAnalyzer())

filter5 = FeatureFilter('TRACK_DISPLACEMENT', 5, True)
settings.addTrackFilter(filter5)

settings.initialSpotFilterValue = 1

print(str(settings))

#----------------------
# Instantiate trackmate
#----------------------

trackmate = TrackMate(model, settings)

#------------
# Execute all
#------------


ok = trackmate.checkInput()
if not ok:
    sys.exit(str(trackmate.getErrorMessage()))

ok = trackmate.process()
if not ok:
    sys.exit(str(trackmate.getErrorMessage()))



#----------------
# Display results
#----------------

#model.getLogger().log('Found ' + str(model.getTrackModel().nTracks(True)) + ' tracks.')

selectionModel = SelectionModel(model)
#displayer =  HyperStackDisplayer(model, selectionModel, imp)
#displayer.render()
#displayer.refresh()

# The feature model, that stores edge and track features.
fm = model.getFeatureModel()

model.getLogger().log('Data starts here.')
model.getLogger().log('Track_ID' +','+ 'Spot_ID' +','+ 'Frame' +','+ 'X' +','+ 'Y' +','+ 'Quality' +','+ 'SN_Ratio' +','+ 'Mean_Intensity')
for id in model.getTrackModel().trackIDs(True):

    # Fetch the track feature from the feature model.
    #v = fm.getTrackFeature(id, 'TRACK_MEAN_SPEED')
    #dur = fm.getTrackFeature(id, 'TRACK_DURATION')
    #model.getLogger().log('')
    #model.getLogger().log('Track ' + str(id) + ': mean velocity = ' + str(v) + ' ' + model.getSpaceUnits() + '/' + model.getTimeUnits())
    #model.getLogger().log('Track ' + str(id) + ': duration = ' + str(dur) + ' ' + model.getTimeUnits())
    track = model.getTrackModel().trackSpots(id)
    for spot in track:
        sid = spot.ID()
        # Fetch spot features directly from spot.
        x=spot.getFeature('POSITION_X')
        y=spot.getFeature('POSITION_Y')
        t=spot.getFeature('FRAME')
        q=spot.getFeature('QUALITY')
        snr=spot.getFeature('SNR')
        mean=spot.getFeature('MEAN_INTENSITY')
        model.getLogger().log(str(id) +','+ str(sid) +','+ str(t) +','+ str(x) +','+ str(y) +','+ str(q) +','+ str(snr) +','+ str(mean))
