/*
Szymon Rusinkiewicz
Princeton University

mesh_filter.cc
Apply a variety of tranformations to a mesh
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "TriMesh.h"
#include "TriMesh_algo.h"
#include "lineqn.h"
#include "XForm.h"

#ifndef M_PI
# define M_PI 3.14159265358979323846
#endif


// Quick 'n dirty portable random number generator 
static inline float tinyrnd()
{
	static unsigned trand = 0;
	trand = 1664525u * trand + 1013904223u;
	return (float) trand / 4294967296.0f;
}


// Is this argument a number?
static bool isanumber(const char *c)
{
	if (!c || !*c)
		return false;
	char *endptr;
	strtod(c, &endptr);
	return (endptr && *endptr == '\0');
}


// Transform the mesh by the given matrix
void apply_xform(TriMesh *mesh, const xform &xf)
{
	int nv = mesh->vertices.size();
	for (int i = 0; i < nv; i++)
		mesh->vertices[i] = xf * mesh->vertices[i];
	if (!mesh->normals.empty()) {
		xform nxf = norm_xf(xf);
		for (int i = 0; i < nv; i++) {
			mesh->normals[i] = nxf * mesh->normals[i];
			normalize(mesh->normals[i]);
		}
	}
}


// Transform the mesh by a matrix read from a file
void apply_xform(TriMesh *mesh, const char *xffilename)
{
	xform xf;
	if (!xf.read(xffilename))
		fprintf(stderr, "Couldn't open %s\n", xffilename);
	else
		apply_xform(mesh, xf);
}


// Clip mesh to the given bounding box
bool clip(TriMesh *mesh, const char *bboxfilename)
{
	FILE *f = fopen(bboxfilename, "r");
	if (!f) {
		fprintf(stderr, "Couldn't open %s\n", bboxfilename);
		return false;
	}

	TriMesh::BBox b;
	if (fscanf(f, "%f%f%f%f%f%f",
		      &b.min[0], &b.min[1], &b.min[2],
		      &b.max[0], &b.max[1], &b.max[2]) != 6) {
		fclose(f);
		fprintf(stderr, "Couldn't read %s\n", bboxfilename);
		return false;
	}
	fclose(f);

	if (b.min[0] > b.max[0]) swap(b.min[0], b.max[0]);
	if (b.min[1] > b.max[1]) swap(b.min[1], b.max[1]);
	if (b.min[2] > b.max[2]) swap(b.min[2], b.max[2]);


	int nv = mesh->vertices.size();
	std::vector<bool> toremove(nv, false);
	for (int i = 0; i < nv; i++)
		if (mesh->vertices[i][0] < b.min[0] ||
		    mesh->vertices[i][0] > b.max[0] ||
		    mesh->vertices[i][1] < b.min[1] ||
		    mesh->vertices[i][1] > b.max[1] ||
		    mesh->vertices[i][2] < b.min[2] ||
		    mesh->vertices[i][2] > b.max[2])
			toremove[i] = true;

	remove_vertices(mesh, toremove);

	return true;
}


// Translate the mesh by (tx, ty, tz)
static inline void trans(TriMesh *mesh, const vec &transvec)
{
	apply_xform(mesh, xform::trans(transvec));
}


// Find center of mass
point center_of_mass(TriMesh *mesh)
{
	point com;
	float totwt = 0;

	mesh->need_faces();
	int nf = mesh->faces.size();
	for (int i = 0; i < nf; i++) {
		const point &v0 = mesh->vertices[mesh->faces[i][0]];
		const point &v1 = mesh->vertices[mesh->faces[i][1]];
		const point &v2 = mesh->vertices[mesh->faces[i][2]];

		point face_com = (v0+v1+v2) / 3.0f;
		float wt = len((v1-v0) CROSS (v2-v0));
		com += wt * face_com;
		totwt += wt;
	}
	return com / totwt;
}


// Scale the mesh so that mean squared distance from center of mass is 1
void normalize_variance(TriMesh *mesh)
{
	point com = center_of_mass(mesh);
	trans(mesh, -com);

	float var = 0;
	int nv = mesh->vertices.size();
	for (int i = 0; i < nv; i++)
		var += len2(mesh->vertices[i]);
	float s = 1.0f / sqrt(var);
	apply_xform(mesh, xform::scale(s));

	trans(mesh, com);
}


// Rotate model so that first principal axis is along +X (using
// forward weighting), and the second is along +Y
void pca_rotate(TriMesh *mesh)
{
	point com = center_of_mass(mesh);
	trans(mesh, -com);

	float C[3][3];
	int nv = mesh->vertices.size();
	for (int i = 0; i < nv; i++) {
		const vec &p = mesh->vertices[i];
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 3; k++)
				C[j][k] += p[j] * p[k];
	}
	float e[3];
	eigdc<3,float>(C, e);

	// Sorted in order from smallest to largest, so grab third column
	vec first(C[0][2], C[1][2], C[2][2]);
	int npos = 0;
	for (int i = 0; i < nv; i++)
		if ((mesh->vertices[i] DOT first) > 0.0f)
			npos++;
	if (npos < nv/2)
		first = -first;

	vec second(C[0][1], C[1][1], C[2][1]);
	npos = 0;
	for (int i = 0; i < nv; i++)
		if ((mesh->vertices[i] DOT second) > 0.0f)
			npos++;
	if (npos < nv/2)
		second = -second;

	vec third = first CROSS second;

	xform xf;
	xf[0] = first[0];  xf[1] = first[1];  xf[2] = first[2];
	xf[4] = second[0]; xf[5] = second[1]; xf[6] = second[2];
	xf[8] = third[0];  xf[9] = third[1];  xf[10] = third[2];

	invert(xf);
	for (int i = 0; i < nv; i++)
		mesh->vertices[i] = xf * mesh->vertices[i];

	trans(mesh, com);
}


// Helper function: return the largest X coord for this face
static float max_x(const TriMesh *mesh, int i)
{
	return max(max(mesh->vertices[mesh->faces[i][0]][0],
		       mesh->vertices[mesh->faces[i][1]][0]),
		       mesh->vertices[mesh->faces[i][2]][0]);
}


// Flip faces so that orientation among touching faces is consistent
void orient(TriMesh *mesh)
{
	mesh->need_faces();
	mesh->tstrips.clear();
	mesh->need_adjacentfaces();

	mesh->flags.clear();
	const unsigned NONE = ~0u;
	mesh->flags.resize(mesh->faces.size(), NONE);

	TriMesh::dprintf("Auto-orienting mesh... ");
	unsigned cc = 0;
	std::vector<int> cc_farthest;
	for (int i = 0; i < mesh->faces.size(); i++) {
		if (mesh->flags[i] != NONE)
			continue;
		mesh->flags[i] = cc;
		cc_farthest.push_back(i);
		float farthest_val = max_x(mesh, i);

		std::vector<int> q;
		q.push_back(i);
		while (!q.empty()) {
			int f = q.back();
			q.pop_back();
			for (int j = 0; j < 3; j++) {
				int v0 = mesh->faces[f][j];
				int v1 = mesh->faces[f][(j+1)%3];
				const std::vector<int> &a = mesh->adjacentfaces[v0];
				for (int k = 0; k < a.size(); k++) {
					int f1 = a[k];
					if (mesh->flags[f1] != NONE)
						continue;
					int i0 = mesh->faces[f1].indexof(v0);
					int i1 = mesh->faces[f1].indexof(v1);
					if (i0 < 0 || i1 < 0)
						continue;
					if (i1 == (i0 + 1) % 3)
						swap(mesh->faces[f1][1],
						     mesh->faces[f1][2]);
					mesh->flags[f1] = cc;
					if (max_x(mesh, f1) > farthest_val) {
						farthest_val = max_x(mesh, f1);
						cc_farthest[cc] = f1;
					}
					q.push_back(f1);
				}
			}
		}
		cc++;
	}

	std::vector<bool> cc_flip(cc, false);
	for (int i = 0; i < cc; i++) {
		int f = cc_farthest[i];
		const point &v0 = mesh->vertices[mesh->faces[f][0]];
		const point &v1 = mesh->vertices[mesh->faces[f][1]];
		const point &v2 = mesh->vertices[mesh->faces[f][2]];
		vec n = (v1 - v0) CROSS (v2 - v0);
		if (n[0] < 0.0f)
			cc_flip[i] = true;
	}

	for (int i = 0; i < mesh->faces.size(); i++) {
		if (cc_flip[mesh->flags[i]])
			swap(mesh->faces[i][1], mesh->faces[i][2]);
	}
	TriMesh::dprintf("Done.\n");
}


// Remove boundary vertices (and faces that touch them)
void erode(TriMesh *mesh)
{
	int nv = mesh->vertices.size();
	std::vector<bool> bdy(nv);
	for (int i = 0; i < nv; i++)
		bdy[i] = mesh->is_bdy(i);
	remove_vertices(mesh, bdy);
}


// Add a bit of noise to the mesh
void noisify(TriMesh *mesh, float amount)
{
	mesh->need_normals();
	mesh->need_neighbors();
	int nv = mesh->vertices.size();
	std::vector<vec> disp(nv);

	for (int i = 0; i < nv; i++) {
		point &v = mesh->vertices[i];
		// Tangential
		for (int j = 0; j < mesh->neighbors[i].size(); j++) {
			const point &n = mesh->vertices[mesh->neighbors[i][j]];
			float scale = amount / (amount + len(n-v));
			disp[i] += (float) tinyrnd() * scale * (n-v);
		}
		if (mesh->neighbors[i].size())
			disp[i] /= (float) mesh->neighbors[i].size();
		// Normal
		disp[i] += (2.0f * (float) tinyrnd() - 1.0f) *
			   amount * mesh->normals[i];
	}
	for (int i = 0; i < nv; i++)
		mesh->vertices[i] += disp[i];
}


void usage(const char *myname)
{
	fprintf(stderr, "Usage: %s infile [options] [outfile]\n", myname);
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "	-color		Add per-vertex color\n");
	fprintf(stderr, "	-nocolor	Remove per-vertex color\n");
	fprintf(stderr, "	-conf		Add per-vertex confidence\n");
	fprintf(stderr, "	-noconf		Remove per-vertex confidence\n");
	fprintf(stderr, "	-tstrip		Convert to use triangle strips\n");
	fprintf(stderr, "	-notstrip	Unpack triangle strips to faces\n");
	fprintf(stderr, "	-reorder	Optimize order of vertices\n");
	fprintf(stderr, "	-orient		Auto-orient faces within the mesh\n");
	fprintf(stderr, "	-faceflip	Flip the order of vertices within each face\n");
	fprintf(stderr, "	-edgeflip	Optimize triangle connectivity by flipping edges\n");
	fprintf(stderr, "	-subdiv		Subdivide faces (planar)\n");
	fprintf(stderr, "	-loop		Perform Loop subdivision\n");
	fprintf(stderr, "	-smooth s	Smooth surface with sigma=s*edgelength\n");
	fprintf(stderr, "	-sharpen s	Sharpen surface with sigma=s*edgelength\n");
	fprintf(stderr, "	-smoothnorm s	Diffuse normals with sigma=s*edgelength\n");
	fprintf(stderr, "	-usmooth n	Perform n iterations of simple umbrella smoothing\n");
	fprintf(stderr, "	-lmsmooth n	Perform n iterations of Taubin's lambda-mu smoothing\n");
	fprintf(stderr, "	-inflate s	Create offset surface s*edgelength away\n");
	fprintf(stderr, "	-noisify s	Add O(s*edgelength) noise to each vertex\n");
	fprintf(stderr, "	-clip bbox	Clip to the given bbox (file has 6 numbers)\n");
	fprintf(stderr, "	-xform file.xf	Transform by the given matrix\n");
	fprintf(stderr, "	-rot r x y z	Rotate r degrees around axis (x,y,z)\n");
	fprintf(stderr, "	-trans x y z	Translate by (x,y,z)\n");
	fprintf(stderr, "	-scale s	Uniform scale by s\n");
	fprintf(stderr, "	-scale x y z	Scale by (x,y,z)\n");
	fprintf(stderr, "	-scale s x y z	Scale by s in direction (x,y,z)\n");
	fprintf(stderr, "	-center		Translate so center of mass is at (0,0,0)\n");
	fprintf(stderr, "	-bbcenter	Translate so center of bbox is at (0,0,0)\n");
	fprintf(stderr, "	-varnorm	Scale so variance (RMS distance) from center is 1\n");
	fprintf(stderr, "	-bbnorm		Scale so bbox has maximum extent 1\n");
	fprintf(stderr, "	-pcarot		Rotate so that principal axes lie along X, Y, Z\n");
	fprintf(stderr, "	-rmunused	Remove unreferenced vertices\n");
	fprintf(stderr, "	-rmslivers	Remove long, skinny faces\n");
	fprintf(stderr, "	-erode		Enlarge boundaries by removing boundary vertices\n");
	exit(1);
}

int main(int argc, char *argv[])
{
	if (argc < 3)
		usage(argv[0]);
	const char *filename = argv[1];

	TriMesh *themesh = TriMesh::read(filename);
	if (!themesh)
		usage(argv[0]);

	bool have_tstrips = !themesh->tstrips.empty();
	for (int i = 2; i < argc; i++) {
		if (!strcmp(argv[i], "-color") ||
		    !strcmp(argv[i], "-colors")) {
			if (themesh->colors.empty()) {
				int nv = themesh->vertices.size();
				themesh->colors.resize(nv, Color::white());
			}
		} else if (!strcmp(argv[i], "-nocolor") ||
			   !strcmp(argv[i], "-nocolors")) {
			themesh->colors.clear();
		} else if (!strcmp(argv[i], "-conf")) {
			if (themesh->confidences.empty()) {
				int nv = themesh->vertices.size();
				themesh->confidences.resize(nv, 1);
			}
		} else if (!strcmp(argv[i], "-noconf")) {
			themesh->confidences.clear();
		} else if (!strcmp(argv[i], "-tstrip") ||
			   !strcmp(argv[i], "-tstrips") ||
			   !strcmp(argv[i], "-strip") ||
			   !strcmp(argv[i], "-strips")) {
			themesh->need_tstrips();
			have_tstrips = true;
		} else if (!strcmp(argv[i], "-notstrip") ||
			   !strcmp(argv[i], "-notstrips") ||
			   !strcmp(argv[i], "-nostrip") ||
			   !strcmp(argv[i], "-nostrips") ||
			   !strcmp(argv[i], "-unstrip")) {
			themesh->need_faces();
			themesh->tstrips.clear();
			have_tstrips = false;
		} else if (!strcmp(argv[i], "-reorder")) {
			reorder_verts(themesh);
		} else if (!strcmp(argv[i], "-orient")) {
			orient(themesh);
		} else if (!strcmp(argv[i], "-faceflip")) {
			faceflip(themesh);
		} else if (!strcmp(argv[i], "-edgeflip")) {
			edgeflip(themesh);
		} else if (!strcmp(argv[i], "-subdiv")) {
			subdiv(themesh, false);
		} else if (!strcmp(argv[i], "-loop")) {
			subdiv(themesh);
		} else if (!strcmp(argv[i], "-smooth")) {
			i++;
			if (!(i < argc && isanumber(argv[i]))) {
				fprintf(stderr, "\n-smooth requires one float parameter: s\n\n");
				usage(argv[0]);
			}
			float amount = atof(argv[i]) * themesh->feature_size();
			smooth_mesh(themesh, amount);
			themesh->pointareas.clear();
			themesh->normals.clear();
		} else if (!strcmp(argv[i], "-sharpen")) {
			i++;
			if (!(i < argc && isanumber(argv[i]))) {
				fprintf(stderr, "\n-sharpen requires one float parameter: s\n\n");
				usage(argv[0]);
			}
			float amount = atof(argv[i]) * themesh->feature_size();
			std::vector<point> origverts = themesh->vertices;
			smooth_mesh(themesh, amount);
			for (int v = 0; v < themesh->vertices.size(); v++)
				themesh->vertices[v] += 2.0f *
					(origverts[v] - themesh->vertices[v]);
			themesh->pointareas.clear();
			themesh->normals.clear();
		} else if (!strcmp(argv[i], "-smoothnorm")) {
			i++;
			if (!(i < argc && isanumber(argv[i]))) {
				fprintf(stderr, "\n-smoothnorm requires one float parameter: s\n\n");
				usage(argv[0]);
			}
			float amount = atof(argv[i]) * themesh->feature_size();
			diffuse_normals(themesh, amount);
		} else if (!strcmp(argv[i], "-usmooth")) {
			i++;
			if (!(i < argc && isdigit(argv[i][0]))) {
				fprintf(stderr, "\n-usmooth requires one int parameter: n\n\n");
				usage(argv[0]);
			}
			int niters = atoi(argv[i]);
			for (int iter = 0; iter < niters; iter++)
				umbrella(themesh, 0.5f);
		} else if (!strcmp(argv[i], "-lmsmooth")) {
			i++;
			if (!(i < argc && isdigit(argv[i][0]))) {
				fprintf(stderr, "\n-lmsmooth requires one int parameter: n\n\n");
				usage(argv[0]);
			}
			int niters = atoi(argv[i]);
			lmsmooth(themesh, niters);
		} else if (!strcmp(argv[i], "-inflate")) {
			i++;
			if (!(i < argc && isanumber(argv[i]))) {
				fprintf(stderr, "\n-inflate requires one float parameter: s\n\n");
				usage(argv[0]);
			}
			float amount = atof(argv[i]) * themesh->feature_size();
			inflate(themesh, amount);
		} else if (!strcmp(argv[i], "-noisify")) {
			i++;
			if (!(i < argc && isanumber(argv[i]))) {
				fprintf(stderr, "\n-noisify requires one float parameter: s\n\n");
				usage(argv[0]);
			}
			float amount = atof(argv[i]) * themesh->feature_size();
			noisify(themesh, amount);
		} else if (!strcmp(argv[i], "-clip")) {
			i++;
			if (!(i < argc)) {
				fprintf(stderr, "\n-clip requires one argument\n\n");
				usage(argv[0]);
			}
			if (!clip(themesh, argv[i]))
				usage(argv[0]);
		} else if (!strcmp(argv[i], "-xf") ||
			   !strcmp(argv[i], "-xform")) {
			i++;
			if (!(i < argc)) {
				fprintf(stderr, "\n-xform requires one argument\n\n");
				usage(argv[0]);
			}
			apply_xform(themesh, argv[i]);
		} else if (!strcmp(argv[i], "-rot") ||
			   !strcmp(argv[i], "-rotate")) {
			i += 4;
			if (!(i < argc &&
			      isanumber(argv[i]) && isanumber(argv[i-1]) &&
			      isanumber(argv[i-2]) && isanumber(argv[i-3]))) {
				fprintf(stderr, "\n-rot requires four arguments\n\n");
				usage(argv[0]);
			}
			vec ax(atof(argv[i-2]), atof(argv[i-1]), atof(argv[i]));
			float ang = M_PI / 180.0f * atof(argv[i-3]);
			apply_xform(themesh, xform::rot(ang, ax));
		} else if (!strcmp(argv[i], "-trans") ||
			   !strcmp(argv[i], "-translate")) {
			i += 3;
			if (!(i < argc && isanumber(argv[i]) &&
			      isanumber(argv[i-1]) && isanumber(argv[i-2]))) {
				fprintf(stderr, "\n-trans requires three arguments\n\n");
				usage(argv[0]);
			}
			vec t(atof(argv[i-2]), atof(argv[i-1]), atof(argv[i]));
			trans(themesh, t);
		} else if (!strcmp(argv[i], "-scale")) {
			int nargs = 0;
			float args[4];
			while (nargs < 4) {
				if (++i >= argc)
					break;
				if (!sscanf(argv[i], "%f", &(args[nargs]))) {
					--i;
					break;
				}
				nargs++;
			}
			if (!(i < argc) || nargs == 0 || nargs == 2) {
				fprintf(stderr, "\n-scale requires 1, 3, or 4 arguments\n\n");
				usage(argv[0]);
			}
			xform s = xform::scale(args[0]);
			if (nargs == 3)
				s = xform::scale(args[0], args[1], args[2]);
			else if (nargs == 4)
				s = xform::scale(args[0], args[1], args[2], args[3]);
			apply_xform(themesh, s);
		} else if (!strcmp(argv[i], "-center")) {
			trans(themesh, -center_of_mass(themesh));
		} else if (!strcmp(argv[i], "-bbcenter")) {
			themesh->need_bbox();
			trans(themesh, -themesh->bbox.center());
		} else if (!strcmp(argv[i], "-varnorm")) {
			normalize_variance(themesh);
		} else if (!strcmp(argv[i], "-bbnorm")) {
			themesh->need_bbox();
			vec l = themesh->bbox.size();
			float ll = max(max(l[0], l[1]), l[2]);
			trans(themesh, -themesh->bbox.center());
			float s = 1.0f / ll;
			apply_xform(themesh, xform::scale(s));
			trans(themesh, themesh->bbox.center());
		} else if (!strcmp(argv[i], "-pcarot")) {
			pca_rotate(themesh);
		} else if (!strcmp(argv[i], "-rmunused")) {
			remove_unused_vertices(themesh);
		} else if (!strcmp(argv[i], "-rmslivers")) {
			remove_sliver_faces(themesh);
		} else if (!strcmp(argv[i], "-erode")) {
			erode(themesh);
		} else if (i == argc - 1 &&
			   (argv[i][0] != '-' || argv[i][1] == '\0')) {
			if (have_tstrips && themesh->tstrips.empty())
				themesh->need_tstrips();
			themesh->write(argv[i]);
		} else {
			fprintf(stderr, "\nUnrecognized option [%s]\n\n", argv[i]);
			usage(argv[0]);
		}
	}
}

