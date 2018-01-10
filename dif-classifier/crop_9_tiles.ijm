input =  'H:\\Tracking_Videos\\Gel_Studies\\11_15_17_Gel_Study_37C_72pH\\10mM\\redo\\'
output = 'H:\\Tracking_Videos\\Gel_Studies\\11_15_17_Gel_Study_37C_72pH\\10mM\\redo\\crop\\'
filename = 'RED_nPEG_37C_pH72_S'
extension = '.tif'
videos = 5
slices = 4



function crop_small(input, output, filename, extension) {
	run("Bio-Formats Importer", "open=" + input + filename + '.tif'  + " color_mode=Default view=Hyperstack stack_order=XYCZT");
	//open(input + filename +  extension);
	master_height = getHeight();
	master_width = getWidth();

	crop_dimension = 512;
	x_offset = 10;
	y_offset = 10;
	num_crops = 0;
	count = 0;

	if (master_height > master_width) {
    	num_crops = floor((master_width-2*x_offset)/crop_dimension);
	}
	else {
    	num_crops = floor((master_height-2*y_offset)/crop_dimension);
	}

	x_free_space = (master_width-2*x_offset)-(num_crops*crop_dimension);
	y_free_space = (master_height-2*y_offset)-(num_crops*crop_dimension);
	x_spacing = x_free_space/(num_crops+1);
	y_spacing = y_free_space/(num_crops+1);

	for (i = 1; i < num_crops + 1; i++) {
    	  for (j = 1; j < num_crops + 1; j++) {
        		count = count + 1;
        		run("Duplicate...", "title=NewStack duplicate");
        		makeRectangle(x_offset+(j-1)*(crop_dimension)+j*x_spacing, y_offset+(i-1)*(crop_dimension)+i*y_spacing, crop_dimension, crop_dimension);
        		roiManager("Add");
        		roiManager("Select", (count-1));
        		roiManager("Rename", "re_crop_" + count);
        		run("Crop");
        		saveAs("Tiff", output + filename + "_" + count);
        		close();
        		selectWindow(filename + extension);
        		//run("Bio-Formats Importer", "open=" + input + filename + '.tif'  + " color_mode=Default view=Hyperstack stack_order=XYCZT");
    	  }
	}
	while (nImages>0){ 
          selectImage(nImages); 
          close(); 
      }
	
}



//for (i = 1; i<slices; i++){
//  for (j = 4; j<videos; j++){
//	crop_small(input, output, filename + i + '_' + j, extension);
//  }
//}

//filename = 'RED_PEG_37C_pH72_S'
//for (i = 1; i<slices+1; i++){
//  for (j = 1; j<videos+1; j++){
//	crop_small(input, output, filename + i + '_' + j, extension);
//  }
//}

input =  'H:\\Tracking_Videos\\Gel_Studies\\11_15_17_Gel_Study_37C_72pH\\10mM\\redo\\'
output = 'H:\\Tracking_Videos\\Gel_Studies\\11_15_17_Gel_Study_37C_72pH\\10mM\\redo\\crop\\'
filename = 'RED_nPEG_37C_pH72_S1_5';
crop_small(input, output, filename, extension);
filename = 'RED_nPEG_37C_pH72_S2_5';
crop_small(input, output, filename, extension);
filename = 'RED_nPEG_37C_pH72_S3_5';
crop_small(input, output, filename, extension);
filename = 'RED_nPEG_37C_pH72_S4_4';
crop_small(input, output, filename, extension);
filename = 'RED_nPEG_37C_pH72_S4_5';
crop_small(input, output, filename, extension);
