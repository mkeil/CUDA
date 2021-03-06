#pragma once

#include <vector>
#include <iostream>

// define structs
struct point
{
    float x;
    float y;
    float z;
};

struct rgb
{
    int r;
    int g;
    int b;
};

struct triangle
{
    point a;
    point b;
    point c;
    rgb color;
};

struct primitives
{
    // Annahme: ich weiß welche Arten von primitives in der Szene möglich sind, Bsp. sphere contentSpheres; etc.
    std::vector <triangle> contentTriangles;
};

struct camera
{
    point location;
    point direction;
    point up;
	float distance;
	float horizontal_angle;
	float vertical_angle;
};

struct ray
{
	point location;
	point direction;
};

struct scene
{
    rgb background;
    primitives contentPrimitives;
    camera contetCamera;
};

// define types
typedef struct point point;
typedef struct rgb rgb;
typedef struct triangle triangle;
typedef struct primitives primitives;
typedef struct camera camera;
typedef struct ray ray;
typedef struct scene scene;

// define stream output for types
std::ostream& operator <<(std::ostream& s, const point& p);
std::ostream& operator <<(std::ostream& s, const rgb& r);
std::ostream& operator <<(std::ostream& s, const triangle& t);
std::ostream& operator <<(std::ostream& s, const camera& c);
std::ostream& operator <<(std::ostream& s, const ray& r);

