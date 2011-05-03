#pragma once

#include <vector>
#include <iostream>

#include "copyToDevice.h"

void sToDevice(const scene& s) {
	std::cout << "scene2Device gestartet" << std::endl;
	size_t primitivSize = sizeof(s.contentPrimitives);
	
	triangle* triangleDevPnt; 
	triangle* triangleHostPnt = (triangle*) &s.contentPrimitives.contentTriangles[0];
	
	cudaError_t error = cudaMalloc(&triangleDevPnt, primitivSize);
	
	error = cudaMemcpy(triangleDevPnt, triangleHostPnt, primitivSize, cudaMemcpyHostToDevice);
	
	
}