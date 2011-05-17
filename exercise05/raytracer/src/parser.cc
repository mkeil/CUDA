#include <glog/logging.h>

#include "parser.h"

#include <iostream>
#include <fstream>

void operator >>(const YAML::Node& node, point& v) {
   // printf("Punkte: \n");
   node[0] >> v.x;
   node[1] >> v.y;
   node[2] >> v.z;
   
   // printf("%f, %f, %f \n",v.x,v.y,v.z);
}

void operator >>(const YAML::Node& node, rgb& r) {
	// printf("Farben: \n");
	node[0] >> r.r;
	node[1] >> r.g;
	node[2] >> r.b;
	
	// printf("%d, %d, %d \n",r.r,r.g,r.b);
}

void operator >>(const YAML::Node& node, triangle& t) {
	// printf("Dreieck wird eingelesen \n");
	node[0] >> t.a;
	node[1] >> t.b;
	node[2] >> t.c;
}

void operator >>(const YAML::Node& node, primitives& p) {
	for(unsigned i=0;i<node.size();i++) {
		triangle t;
		node[i]["triangle"] >>  t; 
		node[i]["color"] >> t.color;
		p.contentTriangles.push_back(t);
	}
}

void operator >>(const YAML::Node& node, lights& l) {
	for(unsigned i=0;i<node.size();i++) {
		point p;
		node[i] >> p;
		l.sceneLights.push_back(p);
   }
}


void operator >>(const YAML::Node& node, camera& c) {
    node["location"] >> c.location;
	node["direction"] >> c.direction;
	node["up"] >> c.up;    
	node["distance"] >> c.distance;
	node["horizontal_angle"] >> c.horizontal_angle;
	node["vertical_angle"] >> c.vertical_angle;
	
}

void parse_scene(const char* filename, scene& s) {
    std::ifstream fin(filename);
    YAML::Parser parser(fin);
	YAML::Node doc;
    parser.GetNextDocument(doc);
	
	find_camera(doc, s.contetCamera);
	find_background(doc, s.background);
	find_primitives(doc, s.contentPrimitives);
	find_lights(doc,s.contentLights);
	
}

void find_lights(const YAML::Node& doc, lights& l) {
	l.sceneLights.clear();
	if (doc.FindValue("lights")) {
		doc["lights"] >> l;
	}    
}

void find_camera(const YAML::Node& doc, camera& c) {
	doc["camera"] >> c;
}

void find_background(const YAML::Node& doc, rgb& b) {
    doc["background"] >> b;
}	
	
void find_primitives(const YAML::Node& doc, primitives& p) {
	doc["primitives"] >> p;
}


