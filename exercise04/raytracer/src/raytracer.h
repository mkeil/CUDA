#pragma once

#include "types.h"

// vector operations
point cross(const point& p1, const point& p2);
float dot(const point& p1, const point& p2);
float norm(const point& p);
void normalize(point& p);
point operator+(const point& left, const point& right);
point operator*(const float& scalar, const point& p);

rgb operator*(const float& scalar, const rgb& c);

// intersection
bool intersect(const ray& r, const triangle& t, point& intersection);
bool intersecRayPlane (const ray& r, const triangle& tri, point& intersection);


// initial rays
// void initial_ray(const camera& c, int x, int y, ray& r);
void initial_ray(const camera& c, int height, int width, int x, int y, ray& r);


void calcTriangleNormal (const triangle& tri,point& triangleNormal);

float calcAngle (const point& a, const point& b);

rgb flatShadingColor (const ray& r, const triangle& nearest_triangle);

void render_image(const scene& s, const int& height, const int& width, rgb* image);

void copySceneToDevice(const scene& s);