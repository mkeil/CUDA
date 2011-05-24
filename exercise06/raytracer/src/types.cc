#include "types.h"

std::ostream& operator <<(std::ostream& s, const point& p) {
    // TODO
    // fill me

	s << p.x << " " << p.y << " " << p.z;
	return s;
}

std::ostream& operator <<(std::ostream& s, const rgb& r) {
    s << r.r << " " << r.g << " " << r.b;
	return s;
}

std::ostream& operator <<(std::ostream& s, const triangle& t) {
	
	s << "A:" << t.a << " B:" << t.b << " C:" << t.c << " Farbe: " << t.color;
	return s;
}

std::ostream& operator <<(std::ostream& s, const camera& c) {
    s << "loc: " << c.location << "dir: " << c.direction << "dis: " << c.distance << "h. angle: " << c.horizontal_angle << "v. angle:" << c.vertical_angle << std::endl;
	return s;
}

std::ostream& operator <<(std::ostream& s, const ray& r) {
	s << " loc: "<< r.location << " dir: " << r.direction;
	return s;
}

