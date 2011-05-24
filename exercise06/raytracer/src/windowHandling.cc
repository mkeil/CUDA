#include <GL/glut.h>

#include "windowHandling.h"

float* picData;
int picWidth;
int picHeight;

void imageOnWindow (rgb* pixelarray, int width, int height,int argc, char **argv) {

	// Initialisieren
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB |GLUT_SINGLE);
	glutInitWindowPosition(1,1);
	glutInitWindowSize(width,height);
	glutCreateWindow("Parallele Algorithmen mit CUDA.");
	
	
	
	picData = (float*) malloc(3 * sizeof(float) * width * height);
	picWidth = width;
	picHeight = height;
	
	int index; 
	int l = 0; 
	
	for (int y = height; y > 0; y--) {
		for (int x = 0; x < width; x++) {
			index = y * width + x;
			picData[3 * l] = pixelarray[index].r / 255.0;
			picData[3 * l + 1] = pixelarray[index].g / 255.0;
			picData[3 * l + 2] = pixelarray[index].b / 255.0;
			l ++; 
		}		
	}
	glutDisplayFunc(drawImage); 
	
	glutMainLoop();
}

void drawImage () {
	glDrawPixels(picWidth,picHeight, GL_RGB, GL_FLOAT, picData);
}
 
