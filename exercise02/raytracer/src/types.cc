#include "types.h"

std::ostream& operator <<(std::ostream& s, const point& p) {
    // TODO
    // fill me

	s << p.x << " " << p.y << " " << p.z;
	return s;
}

std::ostream& operator <<(std::ostream& s, const rgb& r) {
    s << r.r << " " << r.g << " " << r.b<< std::endl;
	return s;
}

std::ostream& operator <<(std::ostream& s, const triangle& t) {
	
	s << "A:" << t.a << "B:" << t.b << "C:" << t.c << std::endl;
	return s;
}

std::ostream& operator <<(std::ostream& s, const camera& c) {
    // TODO
    // fill me
}

std::ostream& operator <<(std::ostream& s, const ray& r) {
	s << r.location << "\n" << r.direction<< std::endl;
	return s;
}

