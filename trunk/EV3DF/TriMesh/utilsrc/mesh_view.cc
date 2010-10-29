/*
Szymon Rusinkiewicz
Princeton University

mesh_view.cc
Simple viewer
*/

#include <stdio.h>
#include <stdlib.h>
#include "TriMesh.h"
#include "XForm.h"
#include "GLCamera.h"
#include <GL/glut.h>
#include <string>
using std::string;


// Globals
TriMesh *themesh;
GLCamera camera;
xform xf;
char *xffilename;
bool draw_edges = false;
bool draw_curv = false;
bool draw_2side = false;
bool shiny = true;
bool lit = true;
float clen;


// Signal a redraw
void need_redraw()
{
	glutPostRedisplay();
}


// Clear the screen
void cls()
{
	glDisable(GL_DITHER);
	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_NORMALIZE);
	glDisable(GL_LIGHTING);
	glDisable(GL_NORMALIZE);
	glDisable(GL_COLOR_MATERIAL);
	glClearColor(0.08, 0.08, 0.08, 0);
	glClearDepth(1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

}


// Set up lights and materials
void setup_lighting()
{
	if (!lit || themesh->normals.empty()) {
		glDisable(GL_LIGHTING);
		glColor3f(0.85, 0.85, 0.85);
		return;
	}

	GLfloat mat_diffuse[4] = { 1, 1, 1, 1 };
	GLfloat mat_specular[4] = { 0.18, 0.18, 0.18, 0.18 };
	if (!shiny) {
		mat_specular[0] = mat_specular[1] =
		mat_specular[2] = mat_specular[3] = 0.0f;
	}
	GLfloat mat_shininess[] = { 64 };
	GLfloat global_ambient[] = { 0.02, 0.02, 0.05, 0.05 };
	GLfloat light0_ambient[] = { 0, 0, 0, 0 };
	GLfloat light0_diffuse[] = { 0.85, 0.85, 0.8, 0.85 };
	GLfloat light1_diffuse[] = { -0.01, -0.01, -0.03, -0.03 };
	GLfloat light0_specular[] = { 0.85, 0.85, 0.85, 0.85 };
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, mat_diffuse);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light0_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light0_specular);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, light1_diffuse);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, global_ambient);
	glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_FALSE);
	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, draw_2side);
	if (!themesh->colors.empty()) {
		glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
		glEnable(GL_COLOR_MATERIAL);
	}
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHT1);
	glEnable(GL_NORMALIZE);
}


// Draw triangle strips.  They are stored as length followed by values.
void draw_tstrips()
{
	const int *t = &themesh->tstrips[0];
	const int *end = t + themesh->tstrips.size();
	while (likely(t < end)) {
		int striplen = *t++;
		glDrawElements(GL_TRIANGLE_STRIP, striplen, GL_UNSIGNED_INT, t);
		t += striplen;
	}
}


// Draw the mesh
void draw_mesh()
{
	glPushMatrix();
	glMultMatrixd(xf);

	glDepthFunc(GL_LESS);
	glEnable(GL_DEPTH_TEST);

	if (draw_2side) {
		glDisable(GL_CULL_FACE);
	} else {
		glCullFace(GL_BACK);
		glEnable(GL_CULL_FACE);
	}

	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT,
			sizeof(themesh->vertices[0]),
			&themesh->vertices[0][0]);
	if (!themesh->normals.empty()) {
		glEnableClientState(GL_NORMAL_ARRAY);
		glNormalPointer(GL_FLOAT,
				sizeof(themesh->normals[0]),
				&themesh->normals[0][0]);
	} else {
		glDisableClientState(GL_NORMAL_ARRAY);
	}

	if (!themesh->colors.empty()) {
		glEnableClientState(GL_COLOR_ARRAY);
		glColorPointer(3, GL_UNSIGNED_BYTE,
			       sizeof(themesh->colors[0]),
			       &themesh->colors[0][0]);
		glEnable(GL_COLOR_MATERIAL);
	} else {
		glDisableClientState(GL_COLOR_ARRAY);
		glDisable(GL_COLOR_MATERIAL);
	}

	if (themesh->tstrips.empty()) {
		// No triangles - draw as points
		glPointSize(1);
		glDrawArrays(GL_POINTS, 0, themesh->vertices.size());
		glPopMatrix();
		return;
	}

	if (draw_edges || draw_curv) {
		glPolygonOffset(10.0f, 10.0f);
		glEnable(GL_POLYGON_OFFSET_FILL);
	}

	draw_tstrips();

	glDisable(GL_POLYGON_OFFSET_FILL);
	if (draw_edges) {
		glPolygonMode(GL_FRONT, GL_LINE);
		glDisableClientState(GL_COLOR_ARRAY);
		glDisable(GL_COLOR_MATERIAL);
		GLfloat global_ambient[] = { 0.2, 0.2, 0.2, 1.0 };
		GLfloat light0_diffuse[] = { 0.8, 0.8, 0.8, 0.0 };
		GLfloat light1_diffuse[] = { -0.2, -0.2, -0.2, 0.0 };
		GLfloat light0_specular[] = { 0.0f, 0.0f, 0.0f, 0.0f };
		glLightModelfv(GL_LIGHT_MODEL_AMBIENT, global_ambient);
		glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
		glLightfv(GL_LIGHT1, GL_DIFFUSE, light1_diffuse);
		glLightfv(GL_LIGHT0, GL_SPECULAR, light0_specular);
		GLfloat mat_diffuse[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, mat_diffuse);
		glColor3f(0, 0, 1); // Used iff unlit
		draw_tstrips();
		glPolygonMode(GL_FRONT, GL_FILL);
	}

	if (draw_curv) {
		glDisable(GL_LIGHTING);
		themesh->need_curvatures();
		int nv = themesh->vertices.size();
		glColor3f(1,0,0);
		glBegin(GL_LINES);
		int i;
		for (i = 0; i < nv; i++) {
			glVertex3fv(themesh->vertices[i]);
			glVertex3fv(themesh->vertices[i] +
				    clen * themesh->normals[i]);
		}
		glColor3f(0.7,0.7,0);
		for (i = 0; i < nv; i++) {
			glVertex3fv(themesh->vertices[i] -
				    clen * themesh->pdir1[i]);
			glVertex3fv(themesh->vertices[i] +
				    clen * themesh->pdir1[i]);
		}
		glColor3f(0,1,0);
		for (i = 0; i < nv; i++) {
			glVertex3fv(themesh->vertices[i] -
				    clen * themesh->pdir2[i]);
			glVertex3fv(themesh->vertices[i] +
				    clen * themesh->pdir2[i]);
		}
		glEnd();
		//glDrawArrays(GL_POINTS, 0, nv);
	}

	glPopMatrix();
}


// Draw the scene
void redraw()
{
	timestamp t = now();
	camera.setupGL(xf * themesh->bsphere.center, themesh->bsphere.r);
	cls();
	setup_lighting();
	draw_mesh();
	glutSwapBuffers();
	printf("\r                        \r%.1f msec.", 1000.0f * (now() - t));
	fflush(stdout);
}


// Set the view...
void resetview()
{
	if (!xf.read(xffilename))
		xf = xform::trans(0, 0, -5.0f * themesh->bsphere.r) *
		     xform::trans(-themesh->bsphere.center);
	camera.stopspin();
}


// Handle mouse button and motion events
static unsigned buttonstate = 0;

void mousemotionfunc(int x, int y)
{
	static const Mouse::button physical_to_logical_map[] = {
		Mouse::NONE, Mouse::ROTATE, Mouse::MOVEXY, Mouse::MOVEZ,
		Mouse::MOVEZ, Mouse::MOVEXY, Mouse::MOVEXY, Mouse::MOVEXY,
	};

	Mouse::button b = Mouse::NONE;
	if (buttonstate & (1 << 3))
		b = Mouse::WHEELUP;
	else if (buttonstate & (1 << 4))
		b = Mouse::WHEELDOWN;
	else if (buttonstate & (1 << 30))
		b = Mouse::LIGHT;
	else
		b = physical_to_logical_map[buttonstate & 7];

	camera.mouse(x, y, b,
		     xf * themesh->bsphere.center, themesh->bsphere.r,
		     xf);
	if (b != Mouse::NONE)
		need_redraw();
}

void mousebuttonfunc(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
		buttonstate |= (1 << button);
	else
		buttonstate &= ~(1 << button);
	if (glutGetModifiers() & GLUT_ACTIVE_CTRL)
		buttonstate |= (1 << 30);
	else
		buttonstate &= ~(1 << 30);

	mousemotionfunc(x, y);
}


// Idle callback
void idle()
{
	if (camera.autospin(xf))
		need_redraw();
	else
		usleep(10000);
}


// Keyboard
void keyboardfunc(unsigned char key, int x, int y)
{
	switch (key) {
		case ' ':
			resetview(); break;
		case 'e':
			draw_edges = !draw_edges; break;
		case 'c':
			draw_curv = !draw_curv; break;
		case '2':
			draw_2side = !draw_2side; break;
		case 'l':
			lit = !lit; break;
		case 's':
			shiny = !shiny; break;
		case 'x':
			xf.write(xffilename); break;
		case '\033': // Esc
		case '\021': // Ctrl Q
		case 'Q':
		case 'q':
			exit(0);
	}
	need_redraw();
}


void usage(const char *myname)
{
	fprintf(stderr, "Usage: %s infile\n", myname);
	exit(1);
}

int main(int argc, char *argv[])
{
	glutInitWindowSize(512, 512);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInit(&argc, argv);

	if (argc < 2)
		usage(argv[0]);
	const char *filename = argv[1];

	themesh = TriMesh::read(filename);
	if (!themesh)
		usage(argv[0]);

	xffilename = new char[strlen(filename) + 4];
	strcpy(xffilename, filename);
	char *dot = strrchr(xffilename, '.');
	if (!dot)
		dot = strrchr(xffilename, '\0');
	strcpy(dot, ".xf");

	if (!themesh->faces.empty() || !themesh->tstrips.empty())
		themesh->need_normals();
	clen = 0.5f * themesh->feature_size();
	themesh->need_tstrips();
	themesh->need_bsphere();

	glutCreateWindow(filename);
	glutDisplayFunc(redraw);
	glutMouseFunc(mousebuttonfunc);
	glutMotionFunc(mousemotionfunc);
	glutKeyboardFunc(keyboardfunc);
	glutIdleFunc(idle);

	resetview();

	glutMainLoop();
}

