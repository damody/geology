/*
Szymon Rusinkiewicz
Princeton University

lmsmooth.cc
Taubin lambda/mu mesh smoothing
*/

#include <stdio.h>
#include "TriMesh.h"
#include "TriMesh_algo.h"


// One iteration of umbrella-operator smoothing
void umbrella(TriMesh *mesh, float stepsize)
{
	mesh->need_neighbors();
	int nv = mesh->vertices.size();
	std::vector<vec> disp(nv);
	int i;
	for (i = 0; i < nv; i++) {
		if (mesh->is_bdy(i)) {
#if 0
			int nn = mesh->neighbors[i].size();
			int nnused = 0;
			if (!nn)
				continue;
			for (int j = 0; j < nn; j++) {
				if (!mesh->is_bdy(mesh->neighbors[i][j]))
					continue;
				disp[i] += mesh->vertices[mesh->neighbors[i][j]];
				nnused++;
			}
			disp[i] /= nnused;
			disp[i] -= mesh->vertices[i];
#else
			disp[i].clear();
#endif
		} else {
			int nn = mesh->neighbors[i].size();
			if (!nn)
				continue;
			for (int j = 0; j < nn; j++)
				disp[i] += mesh->vertices[mesh->neighbors[i][j]];
			disp[i] /= nn;
			disp[i] -= mesh->vertices[i];
		}
	}
	for (i = 0; i < nv; i++)
		mesh->vertices[i] += stepsize * disp[i];
}


// Several iterations of Taubin lambda/mu
void lmsmooth(TriMesh *mesh, int niters)
{
	mesh->need_neighbors();
	TriMesh::dprintf("Smoothing mesh... ");
	for (int i = 0; i < niters; i++) {
		umbrella(mesh, 0.330f);
		umbrella(mesh, -0.331f);
	}
	TriMesh::dprintf("Done.\n");
}

