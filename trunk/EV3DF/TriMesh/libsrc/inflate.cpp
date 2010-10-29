/*
Szymon Rusinkiewicz
Princeton University

inflate.cc
Create an offset surface from a mesh.  Dumb - just moves along the
normal by the given distance, making no attempt to avoid self-intersection.

Eventually, this could/should be extended to use the method in
 Peng, J., Kristjansson, D., and Zorin, D.
 "Interactive Modeling of Topologically Complex Geometric Detail"
 Proc. SIGGRAPH, 2004.
*/


#include <stdio.h>
#include "TriMesh.h"
#include "TriMesh_algo.h"


void inflate(TriMesh *mesh, float amount)
{
	mesh->need_normals();

	TriMesh::dprintf("Creating offset surface... ");
	int nv = mesh->vertices.size();
	for (int i = 0; i < nv; i++)
		mesh->vertices[i] += amount * mesh->normals[i];
	TriMesh::dprintf("Done.\n");
}

