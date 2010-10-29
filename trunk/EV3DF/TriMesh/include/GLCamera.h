#ifndef GLCAMERA_H
#define GLCAMERA_H
/*
Szymon Rusinkiewicz
Princeton University

GLCamera.h
Manages OpenGL camera and trackball/arcball interaction
*/

#include "TriMeshLib.h"
#include "Vec.h"
#include "XForm.h"
#include "timestamp.h"

namespace Mouse {
	enum button { NONE, ROTATE, MOVEXY, MOVEZ, WHEELUP, WHEELDOWN, LIGHT };
};

class TRIMESHDLL GLCamera {
private:
	int lastmousex, lastmousey;
	Mouse::button lastb;
	timestamp last_time;

	vec lightdir;

	bool dospin;
	point spincenter;
	vec spinaxis;
	float spinspeed;

	float fov, pixscale;
	mutable float surface_depth;
	float click_depth;
	float tb_screen_x, tb_screen_y, tb_screen_size;
	bool read_depth(int x, int y, point &p) const;

	void startspin();
	vec mouse2tb(float x, float y);
	void rotate(int mousex, int mousey, xform &xf);
	void movexy(int mousex, int mousey, xform &xf);
	void movez(int mousex, int mousey, xform &xf);
	void wheel(Mouse::button updown, xform &xf);
	void relight(int mousex, int mousey);

public:
	void set_fov(float _fov) { fov = _fov; }
	void mouse(int mousex, int mousey, Mouse::button b,
		   const point &scene_center, float scene_size,
		   xform &xf);
	point mouse_click(int mousex, int mousey,
			 const point &scene_center, float scene_size);
	void stopspin() { dospin = false; }
	bool autospin(xform &xf);
	void setupGL(const point &scene_center, float scene_size) const;
	GLCamera(float _fov = 0.7f) : lastb(Mouse::NONE), dospin(false),
				      spinspeed(0), fov(_fov),
				      surface_depth(0), click_depth(0)
	{
		lightdir[0] = lightdir[1] = 0; lightdir[2] = 1;
		last_time = now();
	}
};

#endif
