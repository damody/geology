/*
Szymon Rusinkiewicz
Princeton University

diffuse.cc
Smoothing of meshes and per-vertex fields
*/

#include <stdio.h>
#include "TriMesh.h"
#include "TriMesh_algo.h"
#include "timestamp.h"
#include <cmath>
#pragma warning(push ,0)


// Approximation to Gaussian...  Used in filtering
static inline float wt(const TriMesh *themesh, int v1, int v2, float invsigma2)
{
	float d2 = invsigma2 * dist2(themesh->vertices[v1],
				     themesh->vertices[v2]);
	return (d2 >= 9.0f) ? 0.0f : exp(-0.5f*d2);
	//return (d2 >= 25.0f) ? 0.0f : exp(-0.5f*d2);
}


// Functor classes for adding std::vector or tensor fields on the surface
struct AccumVec {
	const std::vector<vec> &field;
	AccumVec(const std::vector<vec> &field_) : field(field_)
		{}
	void operator() (const TriMesh *themesh, int v0, vec &f,
			 float w, int v)
	{
		f += w * field[v];
	}
};

struct AccumCurv {
	void operator() (const TriMesh *themesh, int v0, vec &c,
			 float w, int v)
	{
		vec ncurv;
		proj_curv(themesh->pdir1[v], themesh->pdir2[v],
			  themesh->curv1[v], 0, themesh->curv2[v],
			  themesh->pdir1[v0], themesh->pdir2[v0],
			  ncurv[0], ncurv[1], ncurv[2]);
		c += w * ncurv;
	}
};

struct AccumDCurv {
	void operator() (const TriMesh *themesh, int v0, Vec<4> &d,
			 float w, int v)
	{
		Vec<4> ndcurv;
		proj_dcurv(themesh->pdir1[v], themesh->pdir2[v],
			   themesh->dcurv[v],
			   themesh->pdir1[v0], themesh->pdir2[v0],
			   ndcurv);
		d += w * ndcurv;
	}
};


// Diffuse a std::vector field at 1 vertex, weighted by
// a Gaussian of width 1/sqrt(invsigma2)
template <class ACCUM, class T>
static void diffuse_vert_field(TriMesh *themesh, ACCUM accum,
			       int v, float invsigma2, T &flt)
{
	flt = T();
	accum(themesh, v, flt, themesh->pointareas[v], v);
	float sum_w = themesh->pointareas[v];
	const vec &nv = themesh->normals[v];

	unsigned &flag = themesh->flag_curr;
	flag++;
	themesh->flags[v] = flag;
	std::vector<int> boundary = themesh->neighbors[v];
	while (!boundary.empty()) {
		int n = boundary.back();
		boundary.pop_back();
		if (themesh->flags[n] == flag)
			continue;
		themesh->flags[n] = flag;
		if ((nv DOT themesh->normals[n]) <= 0.0f)
			continue;
		// Gaussian weight
		float w = wt(themesh, n, v, invsigma2);
		if (w == 0.0f)
			continue;
		// Downweight things pointing in different directions
		w *= nv DOT themesh->normals[n];
		// Surface area "belonging" to each point
		w *= themesh->pointareas[n];
		// Accumulate weight times field at neighbor
		accum(themesh, v, flt, w, n);
		sum_w += w;
		for (int i = 0; i < themesh->neighbors[n].size(); i++) {
			int nn = themesh->neighbors[n][i];
			if (themesh->flags[nn] == flag)
				continue;
			boundary.push_back(nn);
		}
	}
	flt /= sum_w;
}


// Smooth the mesh geometry.
// XXX - this is perhaps not a great way to do this,
// but it seems to work better than most other things I've tried...
void smooth_mesh(TriMesh *themesh, float sigma)
{
	themesh->need_faces();
	diffuse_normals(themesh, 0.5f * sigma);
	int nv = themesh->vertices.size();

	TriMesh::dprintf("\rSmoothing... ");
	timestamp t = now();

	float invsigma2 = 1.0f / sqr(sigma);

	std::vector<point> dflt(nv);
	int i;
	for (i = 0; i < nv; i++) {
		diffuse_vert_field(themesh, AccumVec(themesh->vertices),
				   i, invsigma2, dflt[i]);
		// Just keep the displacement
		dflt[i] -= themesh->vertices[i];
	}

	// Slightly better small-neighborhood approximation
	int nf = themesh->faces.size();
	for (i = 0; i < nf; i++) {
		point c = themesh->vertices[themesh->faces[i][0]] +
			  themesh->vertices[themesh->faces[i][1]] +
			  themesh->vertices[themesh->faces[i][2]];
		c /= 3.0f;
		for (int j = 0; j < 3; j++) {
			int v = themesh->faces[i][j];
			vec d = 0.5f * (c - themesh->vertices[v]);
			dflt[v] += themesh->cornerareas[i][j] /
				   themesh->pointareas[themesh->faces[i][j]] *
				   exp(-0.5f * invsigma2 * len2(d)) * d;
		}
	}

	// Filter displacement field
	std::vector<point> dflt2(nv);
	for (i = 0; i < nv; i++) {
		diffuse_vert_field(themesh, AccumVec(dflt),
				   i, invsigma2, dflt2[i]);
	}

	// Update vertex positions
	for (i = 0; i < nv; i++)
		themesh->vertices[i] += dflt[i] - dflt2[i]; // second Laplacian

	TriMesh::dprintf("Done.  Filtering took %f sec.\n", now() - t);
}


// Diffuse the normals across the mesh
void diffuse_normals(TriMesh *themesh, float sigma)
{
	themesh->need_normals();
	themesh->need_pointareas();
	themesh->need_neighbors();
	int nv = themesh->vertices.size();
	if (themesh->flags.size() != nv)
		themesh->flags.resize(nv);

	TriMesh::dprintf("\rSmoothing normals... ");
	timestamp t = now();

	float invsigma2 = 1.0f / sqr(sigma);

	std::vector<vec> nflt(nv);
	for (int i = 0; i < nv; i++) {
		diffuse_vert_field(themesh, AccumVec(themesh->normals),
				   i, invsigma2, nflt[i]);
		normalize(nflt[i]);
	}

	themesh->normals = nflt;

	TriMesh::dprintf("Done.  Filtering took %f sec.\n", now() - t);
}


// Diffuse the curvatures across the mesh
void diffuse_curv(TriMesh *themesh, float sigma)
{
	themesh->need_normals();
	themesh->need_pointareas();
	themesh->need_curvatures();
	themesh->need_neighbors();
	int nv = themesh->vertices.size();
	if (themesh->flags.size() != nv)
		themesh->flags.resize(nv);

	TriMesh::dprintf("\rSmoothing curvatures... ");
	timestamp t = now();

	float invsigma2 = 1.0f / sqr(sigma);

	std::vector<vec> cflt(nv);
	int i;
	for (i = 0; i < nv; i++)
		diffuse_vert_field(themesh, AccumCurv(), i, invsigma2, cflt[i]);
	for (i = 0; i < nv; i++)
		diagonalize_curv(themesh->pdir1[i], themesh->pdir2[i],
				 cflt[i][0], cflt[i][1], cflt[i][2],
				 themesh->normals[i],
				 themesh->pdir1[i], themesh->pdir2[i],
				 themesh->curv1[i], themesh->curv2[i]);

	TriMesh::dprintf("Done.  Filtering took %f sec.\n", now() - t);
}


// Diffuse the curvature derivatives across the mesh
void diffuse_dcurv(TriMesh *themesh, float sigma)
{
	themesh->need_normals();
	themesh->need_pointareas();
	themesh->need_curvatures();
	themesh->need_dcurv();
	themesh->need_neighbors();
	int nv = themesh->vertices.size();
	if (themesh->flags.size() != nv)
		themesh->flags.resize(nv);

	TriMesh::dprintf("\rSmoothing curvature derivatives... ");
	timestamp t = now();

	float invsigma2 = 1.0f / sqr(sigma);

	std::vector< Vec<4> > dflt(nv);
	for (int i = 0; i < nv; i++)
		diffuse_vert_field(themesh, AccumDCurv(), i, invsigma2, dflt[i]);

	themesh->dcurv = dflt;
	TriMesh::dprintf("Done.  Filtering took %f sec.\n", now() - t);
}

#pragma warning(pop)