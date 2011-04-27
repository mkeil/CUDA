#include "ppm_writer.h"
#include <fstream>



void write_ppm(rgb* pixelarray, int width, int height, char* filename) {
    
	std::ofstream f;
    f.open(filename);
    
	// std::cout << "Datei schreiben " << filename << std::endl;
	
	f << "P3" << std::endl;
	f << width << " " << height << std::endl;
	f << 255 << std::endl;
	
	
	
	int x,y,index;
	for (y = height-1; y >= 0 ; y --) {
		for (x = 0; x < width; x ++) {
			index = y * width + x;	
			f << pixelarray[index] << std::endl;
		}
	}
	f.close();
	
}

