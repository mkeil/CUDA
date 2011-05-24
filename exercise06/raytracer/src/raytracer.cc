#include <glog/logging.h>
#include <cmath>
#include <math.h>

#include <sys/time.h>


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

__host__ __device__ double norm(const point& p) { //Euclidean norm
    double sum_powers = pow (p.x,2) + pow (p.y,2) + pow (p.z,2);
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

__host__ __device__ bool operator==(const point& p1, const point& p2) {
	
	if ( (p1.x == p2.x) && (p1.y == p2.y)  && ( p1.z == p2.z)) {
		return true; 
	}	else {
		return false;
	}	
}

__host__ __device__ bool operator==(const triangle& t1, const triangle& t2) {
	
	if ((t1.a == t2.a) && (t1.b == t2.b) & (t1.c == t2.c)) {
		return true;
	} else {
		return false;
	}
}

__host__ __device__ void calcTriangleNormal (const triangle& tri,point& triangleNormal) {
	point dirVec1 = tri.a - tri.b;
	point dirVec2 = tri.a - tri.c;
	normalize (dirVec1);
	normalize (dirVec2);
	triangleNormal = cross(dirVec1, dirVec2);
	normalize(triangleNormal);
}

__host__ __device__ bool intersectNew(const ray& r, const triangle& tri, point& intersection) {
	const float epsilon = 0.000001;
	// find vectors ...
	point edge1 = tri.b - tri.a;
	point edge2 = tri.c - tri.a;
	
	// begin calculation determinat
	point pvec = cross(r.direction ,edge2);
	
	float det = dot(edge1,pvec);
	
	if ((det > -epsilon) && (det <epsilon)) {
		return false;
	} else {
		float inv_det = 1.0 / det;
		
		// calculate distance
		point tvec = r.location - tri.a;
		
		// calculate U parameter
		float u = dot(tvec,pvec) * inv_det;
		
		if ((u < 0.0) || (u > 1.0)) {
			return false;
		} else {
			// prepare to test V paramater
			point qvec = cross (tvec, edge1);
			
			// calculate V parameter
			float v = dot(r.direction,qvec) * inv_det;
			
			if ((v < 0.0) || (u + v > 1.0)) {
				return false;
			} else {
				float t = dot(edge2,qvec) * inv_det;
				intersection = r.location + t * r.direction;
				return true;
			}
		}
	}	
}


__host__ __device__ void initial_ray(const camera& c, int height, int width, int x, int y, ray& r) {
	// 3D Erweiterung von http://www.cs.jhu.edu/~misha/Spring11/06.pdf
	
	double pi = 3.1415926;
	point xVektor = cross(c.up, c.direction);
	point yVektor = c.up;
	
	normalize(xVektor);
	normalize(yVektor);
	point imageCenter = c.location + c.distance * c.direction;
	
	point halfHeightSpan = c.distance * tan (c.vertical_angle * pi / 360 ) * yVektor; 
	point halfWidthSpan = c.distance * tan (c.horizontal_angle * pi / 360) * xVektor;

	point upperLeft = imageCenter - halfWidthSpan + halfHeightSpan;
	
	point imagePlanePoint = upperLeft +((halfWidthSpan * (2.0 / width)) * (x + 0.5)) - ((halfHeightSpan * (2.0 / height)) * (y + 0.5));
	
	r.direction = imagePlanePoint - c.location;
	normalize(r.direction);
	
	r.location = imagePlanePoint;
	
}




__host__ __device__ float calcAngle (const point& a, const point& b) {
	float angle = dot (a,b) / ( norm(a) * norm (b) ); 
	return angle;
}

__host__ __device__ rgb flatShadingColor (const ray& r, const triangle& nearest_triangle) {
	point triangleNormal; 
	
	calcTriangleNormal (nearest_triangle,triangleNormal);
	float angle = calcAngle (triangleNormal,r.direction); 
	float ratio = fabs(angle);
	rgb shadingColor = ratio * nearest_triangle.color; // hier ggf. einen Operator überschreiben
	// rgb shadingColor = nearest_triangle.color; 
	
	return shadingColor ;
}

__host__ __device__ float disToLight (const point& p, point light) {
	point disVec = p - light;
	return norm (disVec);
}


__host__ __device__ bool pointInShadow (point* light, const point& intersecPoint,triangle* triPnt, int nrTri, const triangle& ignoreTriangle) {
	
	ray rayToLight;
	rayToLight.location = intersecPoint;
	rayToLight.direction = *light - intersecPoint;;
	normalize(rayToLight.direction);
	 
	// liegt ein anderes Dreieck im Lichtstahl ?
	point newIntersection;
	for (int l = 0; l < nrTri; l++){
		bool intersecRayToLight = intersectNew(rayToLight, triPnt[l], newIntersection);
		if (intersecRayToLight && (disToLight (newIntersection, *light) < disToLight(intersecPoint, *light) )) {	
			if ( !(triPnt[l] == ignoreTriangle)) { // test ob der Schnittpunkt zum ursprünglichen Dreieck gehört?
				return true;
			}
		}
	}
	return false; 
}


__host__ __device__ rgb lReflectanceColor (const ray& r,const triangle& nearest_triangle,point* light, const point& intersecPoint,triangle* triPnt, int nrTri) {
	rgb reflectedColor;
	bool shadow = pointInShadow(light, intersecPoint,triPnt,nrTri, nearest_triangle);
	rgb black; black.r = 0; black.g = 0; black.b = 0;
	
	if (!shadow)  {
		point triangleNormal; 
		calcTriangleNormal (nearest_triangle,triangleNormal);
		point reflectanceDir = *light - intersecPoint;
		normalize(reflectanceDir);
		float angle = calcAngle (reflectanceDir, triangleNormal); 
		float ratio = fabs(angle);
		reflectedColor = ratio * nearest_triangle.color;
	 } else {
		 reflectedColor = black;
	 }
	return reflectedColor ;
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

__host__ __device__ float disToCam (const point& p, const camera& c) {
	point disVec = p - c.location;
	return norm (disVec);
}

__global__ void calcPoint (parameters* para, triangle* triPnt, point* lightsPnt, rgb* image, int x, int y) {
			#ifndef NO_CUDA
				x = threadIdx.x + blockIdx.x * blockDim.x;
				y = threadIdx.y + blockIdx.y * blockDim.y;
			#endif
			if (x < para->width && y < para->height) {
				
				ray r; point intersectPoint; rgb color; 
				
				bool hit = false; 
				point nearestIntersection; nearestIntersection.x = 9999; nearestIntersection.y = 9999; nearestIntersection.z = 9999;
				ray hitingRay; 
				triangle hitingTriangle;
				
				
				initial_ray(para->cam, para->height, para->width,  x, y, r);
				
				for (int l = 0; l < para->nrTriangle; l++){
					if (intersectNew(r, triPnt[l], intersectPoint)) {	
					
						if (disToCam (intersectPoint, para->cam) < disToCam (nearestIntersection, para->cam)) {						 
							hit = true; 
							nearestIntersection = intersectPoint;
							hitingRay = r;
							hitingTriangle = triPnt[l];
						}
					} 
				}
				if (hit) {
					color = lReflectanceColor (hitingRay,hitingTriangle,lightsPnt,nearestIntersection,triPnt,para->nrTriangle);
					// color = flatShadingColor (hitingRay, hitingTriangle); 
					// color = hitingTriangle.color;
					// std::cout << "getroffenes Dreick: " << hitingTriangleNr << std::endl;
				}
				else {
					color = para->background;
				}
				
				int index = y * para->width + x;
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
	
	float elapsedTime; 
	
	#if NO_CUDA
		timeval start, end, diff;
		gettimeofday(&start, 0);
 
		for (int y = 0; y < height; y ++) {
			for (int x = 0; x < width; x ++) {	
				calcPoint(&para, triangles, lights,image, x, y);
			}
		}
		
		gettimeofday(&end, 0);
		
		diff.tv_sec = end.tv_sec - start.tv_sec; 
		diff.tv_usec = end.tv_usec - start.tv_usec; 
		
		
		
		elapsedTime = diff.tv_sec * 1000.0 + diff.tv_usec / 1000.0;

		
	// */
	#else
		cudaEvent_t startRender, stopRender; 
		
		
		cudaEventCreate(&startRender);  cudaEventCreate(&stopRender); 
		
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

		
		cudaEventRecord(startRender, 0);
		int value = 16;
		dim3 Blocksize (value,value,1);
		dim3 Gridsize ((width + value - 1) / value,(height + value - 1) / value,1);
		calcPoint<<<Gridsize,Blocksize>>> (devPtnPara, devPtnTri, devPtnLight, devPtnImage,0,0);
		error = cudaGetLastError ();
		CHECK_EQ (error,cudaSuccess) << cudaGetErrorString(error);
		error = cudaThreadSynchronize();
		CHECK_EQ (error,cudaSuccess) << cudaGetErrorString(error);
		cudaEventRecord(stopRender, 0);
		
		cudaEventSynchronize(stopRender);
		// float elapsedTime; 
		cudaEventElapsedTime(&elapsedTime, startRender, stopRender);
		cudaEventDestroy(startRender); 
		cudaEventDestroy(stopRender); 
				
		// zurückkopieren
		error = cudaMemcpy (image, devPtnImage, width * height * sizeof(rgb),cudaMemcpyDeviceToHost);
		CHECK_EQ (error,cudaSuccess) << cudaGetErrorString(error);
		
	#endif
	float rayPerMs = (width * height) / elapsedTime;
	std::cout << "Für das Rendern auf wurden " << elapsedTime << " ms gebraucht." << std::endl;
	std::cout << "pro ms wurden " << rayPerMs << " Rays initialisiert." << std::endl;
}