input =  'H:/Tracking_Videos/Gel_Studies/11_15_17_Gel_Study_37C_72pH/10mM/redo/crop/'
output = 'H:/Tracking_Videos/Gel_Studies/11_15_17_Gel_Study_37C_72pH/10mM/redo/crop/Output/'
filename = 'RED_PEG_37C_pH72_S1_1_'
extension = '.tif'
videos = 9
numvideos = 5
slices = 4

abso = 2
rad = 4
disp = 6
lin = 2

function find_trajectories(input, output, filename, extension) {
	open(input + filename +  extension);
	run("Mean...", "radius=2 stack");
	run("Duplicate...", "duplicate");
	run("Mean...", "radius=20 stack");
	imageCalculator("Subtract create stack", filename + extension, filename + '-1' + extension);
	saveAs('Tiff', output + filename);
	run("Particle Tracker 2D/3D", "radius=rad cutoff=0 per/abs=abso link=lin displacement=disp dynamics=Brownian");
	while (nImages > 0) {
		selectImage(nImages);
		close();
	}
}

filename = 'RED_PEG_37C_pH72_S'
for (i = 1; i<slices+1; i++){
  for (j = 1; j<numvideos+1; j++){
	for (k = 1; k < videos + 1;  k++){
	  find_trajectories(input, output, filename + i + '_' + j + '_' + k, extension);}
  }
}