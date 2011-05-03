#include <glog/logging.h>
#include <cmath>

#include <iostream>
#include <ostream>
#include <iterator>


#include "raytracer.h"

//WE ASSUME LEFT-HANDED ORIENTATION (left hand rule)...
point cross(const point& p1, const point& p2) {
    point tmp;
	tmp.x = p1.y * p2.z - p1.z * p2.y;
	tmp.y = p1.z * p2.x - p1.x * p2.z;
	tmp.z = p1.x * p2.y - p1.y * p2.x;
	return tmp;
}

float dot(const point& p1, const point& p2) {
	return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
}

float norm(const point& p) { //Euclidean norm
    float sum_powers = pow (p.x,2) + pow (p.y,2) + pow (p.z,2);
	return sqrt(sum_powers);
}

void normalize(point& p) {
    p = (1 / norm(p)) * p;
}

point operator+(const point& left, const point& right) {
    point tmp;
	tmp.x = left.x + right.x;
	tmp.y = left.y + right.y;
	tmp.z = left.z + right.z;
	return tmp;
}

point operator-(const point& left, const point& right) {
    point tmp;
	tmp.x = left.x - right.x;
	tmp.y = left.y - right.y;
	tmp.z = left.z - right.z;
	return tmp;
}

point operator*(const float& scalar, const point& p) {
	point tmp;
	tmp.x = scalar * p.x; 
	tmp.y = scalar * p.y;
	tmp.z = scalar * p.z;
	return tmp; 
}

bool intersect(const ray& r, const triangle& tri, point& intersection) {
    // Vorgehen nach http://wwwcg.in.tum.de/Teaching/SS2007/Proseminar/Workouts/data/Florian_Ferstl.pdf
	// Hilfsvariablen berechnen
	point d = r.direction;
	point o = r.location;
	
	normalize (d); 
	
	point e1 = tri.b - tri.a;
	point e2 = tri.c - tri.a;
	
	// erstes Kreuzpordukt und erste Prüfung
	point p = cross (d,e2);
	float D = dot (p, e1);
	
	if (D == 0) {
		return false; // Strahl parallel zur Dreiecksebene
	} else {
		float D_inv = 1 / D;
		point t = o - tri.a;
		float alpha = dot (p,t) * D_inv;
		if ( (alpha < 0) | (alpha > 1) ) {
			return false; // Strahl kann dreiecksebene nichtmehr schneiden
		} else {
			// zweites kreuzpordukt und zweite Prüfung
			point q = cross (t, e1);
			float beta = dot (q,d) * D_inv;
			if ( (beta < 0) | ( (beta + alpha) > 1)) {
				return false; 
			} else {
				float lamda = dot (q, e2) * D_inv;
				if (lamda > 0) {
					intersection = o + lamda * d; 
					return true;
				} else {
					return false;
				}
			}
		}
	}
}


void initial_ray(const camera& c, int height, int width, int x, int y, ray& r) {
	// 3D Erweiterung von http://www.cs.jhu.edu/~misha/Spring11/06.pdf
	
	point xVektor = cross(c.up, c.direction);
	point yVektor = c.up;
	
	normalize(xVektor);
	normalize(yVektor);
	
	point imageCenter = c.location + c.distance * c.direction;
	
	point halfHeightSpan = c.distance * tan (c.vertical_angle / 2) * yVektor; 
	point top = imageCenter + halfHeightSpan;
	point bodom = imageCenter - halfHeightSpan;
	
	point halfWidthSpan = c.distance * tan (c.horizontal_angle / 2) * xVektor;
	point right = imageCenter + halfWidthSpan;
	point left = imageCenter - halfWidthSpan;
	
	point imagePlanePoint; 
	
	imagePlanePoint.z = 1; // später noch genauer festlegen
	
	imagePlanePoint.y = bodom.y + ((y + 0.5) / height) * (top.y - bodom.y);
	
	imagePlanePoint.x = left.x + ((x + 0.5) / width) * (right.x - left.x);
	
	
	
	
	r.direction = imagePlanePoint - c.location;
	
	r.location = c.location;
	
	/*
	std::cout << "imageCenter " << imageCenter << std::endl;
	std::cout << "top: " << top << " bodom: " << bodom << std::endl;
	std::cout << "left: " << left << " right " << right << std::endl;
	std::cout << "Punkt auf Bildebene: " << imagePlanePoint << std::endl;
	std::cout << "Strahl: " << r << std::endl;
	*/
}

void render_image(const scene& s, const int& height, const int& width, rgb* image) {
	
	int x,y,index;
	ray r;
	point ins;
	rgb color; 
	
	point nearestIntersection; 
	
	/*
	initial_ray(s.contetCamera, height, width, 9, 9, r);
	if (intersect(r, s.contentPrimitives.contentTriangles[0], ins)) {				
		std::cout << "Treffer bei: " << ins << std::endl;
	} 
	*/  
	
	
	//  /*
	for (y = 0; y < height; y ++) {
		for (x = 0; x < width; x ++) {	
			index = y * width + x;
			int unsigned l;
			nearestIntersection.x = 999;
			nearestIntersection.y = 999;
			nearestIntersection.z = 999;
			color = s.background;
			
			// std::cout << "Koordinaten: " << x << "," << y << std::endl;
			
			initial_ray(s.contetCamera, height, width,  x, y, r);
			
			// std::cout << "Strahl: " << r << std::endl;
			for (l = 0; l < s.contentPrimitives.contentTriangles.size(); l++){
				// std::cout << "zutestendes Dreieck:" << s.contentPrimitives.contentTriangles[l] << std::endl;
				if (intersect(r, s.contentPrimitives.contentTriangles[l], ins)) {	
					if (norm (ins) < norm(nearestIntersection)) {
						color = s.contentPrimitives.contentTriangles[l].color;	
						nearestIntersection = ins;
					//	std::cout << "Treffer bei: " << ins << std::endl;
					}
				} 
			}
			// std::cout << "Index: " << index << " Farbe: " << color << std::endl;
			// std::cout << "*******" << std::endl;
			image[index] = color; 
		}
	}
	// */
	
	
	
}

