#include <glog/logging.h>
#include <cmath>
#include <math.h>

#include <iostream>
#include <ostream>
#include <iterator>
#include <stdlib.h>

#include "raytracer.h"

//WE ASSUME LEFT-HANDED ORIENTATION (left hand rule)...
__host__ __device__ point cross(const point& p1, const point& p2) {
    point tmp;
	tmp.x = p1.y * p2.z - p1.z * p2.y;
	tmp.y = p1.z * p2.x - p1.x * p2.z;
	tmp.z = p1.x * p2.y - p1.y * p2.x;
	return tmp;
}

__host__ __device__ float dot(const point& p1, const point& p2) {
	return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
}

__host__ __device__ float norm(const point& p) { //Euclidean norm
    float sum_powers = pow (p.x,2) + pow (p.y,2) + pow (p.z,2);
	return sqrt(sum_powers);
}

__host__ __device__ void normalize(point& p) {
    p = (1 / norm(p)) * p;
	
}

__host__ __device__ point operator+(const point& left, const point& right) {
    point tmp;
	tmp.x = left.x + right.x;
	tmp.y = left.y + right.y;
	tmp.z = left.z + right.z;
	return tmp;
}

__host__ __device__ point operator-(const point& left, const point& right) {
    point tmp;
	tmp.x = left.x - right.x;
	tmp.y = left.y - right.y;
	tmp.z = left.z - right.z;
	return tmp;
}

__host__ __device__ point operator*(const float& scalar, const point& p) {
	point tmp;
	tmp.x = scalar * p.x; 
	tmp.y = scalar * p.y;
	tmp.z = scalar * p.z;
	return tmp; 
}

__host__ __device__ rgb operator*(const float& scalar, const rgb& c) {
	rgb tmp;
	tmp.r = scalar * c.r; 
	tmp.g = scalar * c.g;
	tmp.b = scalar * c.b;
	return tmp; 
}



__host__ __device__ point operator*(const point& p, const float& scalar) {
	return scalar *p;
}

__host__ __device__ void calcTriangleNormal (const triangle& tri,point& triangleNormal) {
	point dirVec1 = tri.a - tri.b;
	point dirVec2 = tri.a - tri.c;
	triangleNormal = cross(dirVec1, dirVec2);
	normalize(triangleNormal);
}

__host__ __device__ bool intersecRayPlane (const ray& r, const triangle& tri, point& intersection) {
	// Ebene aus Dreieck erstellen E: planeNormal * X + planeSkalar = 0
	// Strahl: X = r.location + parameterT * r.direction; 
	
	const float epsilon = 0.000001;
	
	point planeNormal;
	
	calcTriangleNormal (tri,planeNormal);

	float planeSkalar = dot(tri.a-r.location,planeNormal);
	float denominator = dot(planeNormal, r.direction);
	
	/*
	std::cout << "Strahl" << r << std::endl;
	std::cout << "Dreieck" << tri << std::endl;	
	std::cout << "planeNormal " << planeNormal << std::endl;
	*/
 /*
	std::cout << "Strahl" << r << std::endl;
	std::cout << "Dreieck" << tri << std::endl;	
	std::cout << "planeSkalar " << planeSkalar << std::endl;
	std::cout << "planeNormal " << planeNormal << std::endl;
	std::cout << "denominator" << denominator << std::endl;
	 */
	
	if (denominator > -epsilon && denominator < epsilon) {
		// Strahl und Ebene parallel
		// std::cout << "Strahl und ebene parallel" << std::endl;
		return false; 
	} else {
		float numerator = planeSkalar;
		float parameterT = numerator / denominator;
		// std::cout << "numerator " << numerator << std::endl;
		// std::cout << "parameterT " << parameterT << std::endl;
		if (parameterT < 0) {
			// Schnittpunkt liegt in entgegengesetzter Richtung des Strahls
			// std::cout << "Schnittpunkt liegt in entgegengesetzter Richtung des Strahls. ParameterT: " << parameterT << std::endl;
			// intersection = r.location + parameterT * r.direction;  
			// std::cout << "Schnittpunkt würde bei: " << intersection << " liegen" << std::endl;
			return false;
		} else {
			 
			intersection = r.location + parameterT * r.direction; 
			// std::cout << "Strahl schneidet DreichsEbene bei: " << intersection << std::endl;
			return true;
		}
	}
}
__host__ __device__ void calcBaryCoor (const triangle& tri, const point& intersection, float& beta, float& gamma) {
	// Berechnung aus http://www.uninformativ.de/bin/RaytracingSchnitttests-76a577a-CC-BY.pdf
	
	point a = tri.a - intersection;
	point b = tri.a - tri.b;
	point c = tri.a - tri.c;
	float commonDenominator = dot(c,c) * dot(b,b) - dot (c,b) * dot (b,c);
	float numeratorBeta = dot(a,b) * dot (c,c) - dot (a,c) * dot (c,b);
	
	float numeratorGamma = dot (a,c) * dot (b,b) - dot (a,b) * dot (b,c);
	
	beta = numeratorBeta / commonDenominator;
	gamma = numeratorGamma / commonDenominator; 

}


__host__ __device__ bool intersect(const ray& r, const triangle& tri, point& intersection) {
	// Vorgehen nach http://www.cs.princeton.edu/courses/archive/fall00/cs426/lectures/raycast/raycast.pdf
	bool intersecPlan = intersecRayPlane (r, tri, intersection);
	
	if (!intersecPlan) {
		return false;
	} else {
		// Testen ob Schnittpunkt in innerhalb des Dreiecks
		
		float beta;
		float gamma;
		calcBaryCoor (tri, intersection, beta, gamma);

		if ( (beta >= 0) && (gamma >= 0) && (1 - beta - gamma >= 0)) {
			

			return true;
		} else {

			return false; 
		}
			
	}
}


__host__ __device__ void initial_ray(const camera& c, int height, int width, int x, int y, ray& r) {
	// 3D Erweiterung von http://www.cs.jhu.edu/~misha/Spring11/06.pdf
	
	point xVektor = cross(c.up, c.direction);
	point yVektor = c.up;
	
	normalize(xVektor);
	normalize(yVektor);
	point imageCenter = c.location + c.distance * c.direction;
	
	point halfHeightSpan = c.distance * tan (c.vertical_angle * 3.1415 / 360 ) * yVektor; 
	// point top = imageCenter + halfHeightSpan;
	// point bodom = imageCenter - halfHeightSpan;
	
	point halfWidthSpan = c.distance * tan (c.horizontal_angle * 3.1415 / 360) * xVektor;
	// point right = imageCenter + halfWidthSpan;
	point upperLeft = imageCenter - halfWidthSpan + halfHeightSpan;
	
	point imagePlanePoint = upperLeft +((halfWidthSpan * (2.0 / width)) * (x + 0.5)) - ((halfHeightSpan * (2.0 / height)) * (y + 0.5));
	
	r.direction = imagePlanePoint - c.location;
	normalize(r.direction);
	
	r.location = imagePlanePoint;
	
}




__host__ __device__ float calcAngle (const point& a, const point& b) {
	float angle = dot (a,b) / (norm(a) * norm (b)); 
	return angle;
}

__host__ __device__ rgb flatShadingColor (const ray& r, const triangle& nearest_triangle) {
	point triangleNormal; 
	calcTriangleNormal (nearest_triangle,triangleNormal);
	float angle = calcAngle (r.direction, triangleNormal); 
	float ratio = fabs(angle);
	rgb shadingColor = ratio * nearest_triangle.color; // hier ggf. einen Operator überschreiben
	// std::cout << "DreiecksNormale: " << triangleNormal << " r.dir : " <<  r.direction << std::endl;
	return shadingColor ;
}
struct parameters {
	rgb background;
    camera cam;
	int width;
	int height;
	int nrTriangle;
	int nrLights;
};

typedef struct parameters parameters; 

__global__ void calcPoint (parameters* para, triangle* triPnt, point* lightsPnt, rgb* image, int x, int y) {
			#ifndef NO_CUDA
				x = threadIdx.x + blockIdx.x * blockDim.x;
				y = threadIdx.y + blockIdx.y * blockDim.y;
			#endif
			if (x < para->width && y < para->height) {
			
			
			ray r;
			point ins;
			rgb color; 
			
			int index = y * para->width + x;
			int l;
			point nearestIntersection; 
			
			nearestIntersection.x = 999;
			nearestIntersection.y = 999;
			nearestIntersection.z = 999;
			
			
			initial_ray(para->cam, para->height, para->width,  x, y, r);
			
			color = para->background;
			for (l = 0; l < para->nrTriangle; l++){
				
				if (intersect(r, triPnt[l], ins)) {	
					if (norm (ins) < norm(nearestIntersection)) {
						
						nearestIntersection = ins;
						// color = s.contentPrimitives.contentTriangles[l].color; 
						color = flatShadingColor (r, triPnt[l]); 
					}
				} 
			}
			
			// std::cout << "Index: " << index << " Farbe: " << color << std::endl;
			// std::cout << "*******" << std::endl;
			image[index] = color; 
			}
}


void render_image(const scene& s, const int& height, const int& width, rgb* image) {
	
	int size = s.contentPrimitives.contentTriangles.size();
	triangle* triangles = (triangle*)malloc(size*sizeof(triangle));
	for(int i = 0;i < size;i++)
		triangles[i] = s.contentPrimitives.contentTriangles[i];
	
	int lsize = s.contentLights.sceneLights.size();
	point* lights = (point*)malloc(lsize*sizeof(point));
	for(int i = 0;i < lsize;i++)
		lights[i] = s.contentLights.sceneLights[i];
	
	
	// paras kopieren
	parameters para;
	para.background = s.background;
    para.cam = s.contetCamera;
	para.width = width;
	para.height = height;
	para.nrTriangle = size;
	para.nrLights = lsize;
	
	
	#if NO_CUDA
	
	for (int y = 0; y < height; y ++) {
		for (int x = 0; x < width; x ++) {	
			calcPoint(&para, triangles, lights,image, x, y);
		}
	}
	// */
	#else
	
	parameters* devPtnPara;
	cudaError_t error = cudaMalloc (&devPtnPara,sizeof(parameters));
	CHECK_EQ (error,cudaSuccess) << cudaGetErrorString(error);
	CHECK_NOTNULL (devPtnPara);
	
	error = cudaMemcpy (devPtnPara,&para,sizeof(para),cudaMemcpyHostToDevice); 
	CHECK_EQ (error,cudaSuccess) << cudaGetErrorString(error);
	
	
	// triangles kopieren
	
	triangle* devPtnTri;
	error = cudaMalloc (&devPtnTri,size * sizeof(triangle));
	CHECK_EQ (error,cudaSuccess) << cudaGetErrorString(error);
	CHECK_NOTNULL (devPtnTri);
	
	error = cudaMemcpy (devPtnTri,triangles,size * sizeof(triangle),cudaMemcpyHostToDevice); 
	CHECK_EQ (error,cudaSuccess) << cudaGetErrorString(error);
	// lights kopieren
	point* devPtnLight = NULL;
	if(lsize > 0){
	error = cudaMalloc (&devPtnLight,lsize * sizeof(point));
	CHECK_EQ (error,cudaSuccess) << cudaGetErrorString(error);
	CHECK_NOTNULL (devPtnLight);
	
	error = cudaMemcpy (devPtnLight,lights,lsize * sizeof(point),cudaMemcpyHostToDevice); 
	CHECK_EQ (error,cudaSuccess) << cudaGetErrorString(error);
	}
	// image alloc
	rgb* devPtnImage;
	error = cudaMalloc (&devPtnImage,width * height * sizeof(rgb));
	CHECK_EQ (error,cudaSuccess) << cudaGetErrorString(error);
	CHECK_NOTNULL (devPtnImage);
	// kernel aufrufen
	int value = 16;
	dim3 Blocksize (value,value,1);
	dim3 Gridsize ((width + value - 1) / value,(height + value - 1) / value,1);
	calcPoint<<<Gridsize,Blocksize>>> (devPtnPara, devPtnTri, devPtnLight, devPtnImage,0,0);
	error = cudaGetLastError ();
	CHECK_EQ (error,cudaSuccess) << cudaGetErrorString(error);
	error = cudaThreadSynchronize();
	CHECK_EQ (error,cudaSuccess) << cudaGetErrorString(error);
	// zurückkopieren
	error = cudaMemcpy (image, devPtnImage, width * height * sizeof(rgb),cudaMemcpyDeviceToHost);
	CHECK_EQ (error,cudaSuccess) << cudaGetErrorString(error);
	
	
		
#endif
}