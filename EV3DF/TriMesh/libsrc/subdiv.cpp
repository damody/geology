/*
Szymon Rusinkiewicz
Princeton University

subdiv.cc
Perform subdivision on a mesh.
*/


#include <stdio.h>
#include "TriMesh.h"
#include "TriMesh_algo.h"
#pragma warning(push ,0)

// Subdivide a mesh
void subdiv(TriMesh *mesh, bool Loop /* = true */)
{
	bool have_col = !mesh->colors.empty();
	bool have_conf = !mesh->confidences.empty();
	mesh->flags.clear();
	mesh->normals.clear();
	mesh->pdir1.clear(); mesh->pdir2.clear();
	mesh->curv1.clear(); mesh->curv2.clear();
	mesh->dcurv.clear();
	mesh->cornerareas.clear(); mesh->pointareas.clear();
	mesh->need_faces(); mesh->tstrips.clear();
	mesh->neighbors.clear();
	mesh->need_across_edge(); mesh->need_adjacentfaces();


	TriMesh::dprintf("Subdividing mesh... ");

	// Introduce new vertices
	int nf = mesh->faces.size();
	std::vector<TriMesh::Face> newverts(nf, TriMesh::Face(-1,-1,-1));
	int old_nv = mesh->vertices.size();
	mesh->vertices.reserve(4 * old_nv);
	std::vector<int> newvert_count(old_nv + 3*nf);
	if (have_col)
		mesh->colors.reserve(4 * old_nv);
	if (have_conf)
		mesh->confidences.reserve(4 * old_nv);
	int i;
	for (i = 0; i < nf; i++) {
		for (int j = 0; j < 3; j++) {
			int ae = mesh->across_edge[i][j];
			if (newverts[i][j] == -1 && ae != -1) {
				if (mesh->across_edge[ae][0] == i)
					newverts[i][j] = newverts[ae][0];
				else if (mesh->across_edge[ae][1] == i)
					newverts[i][j] = newverts[ae][1];
				else if (mesh->across_edge[ae][2] == i)
					newverts[i][j] = newverts[ae][2];
			}
			if (newverts[i][j] == -1) {
				mesh->vertices.push_back(point());
				newverts[i][j] = mesh->vertices.size() - 1;
				if (ae != -1) {
					if (mesh->across_edge[ae][0] == i)
						newverts[ae][0] = newverts[i][j];
					else if (mesh->across_edge[ae][1] == i)
						newverts[ae][1] = newverts[i][j];
					else if (mesh->across_edge[ae][2] == i)
						newverts[ae][2] = newverts[i][j];
				}
				if (have_col)
					mesh->colors.push_back(Color());
				if (have_conf)
					mesh->confidences.push_back(0);
			}
			const TriMesh::Face &v = mesh->faces[i];
			if (Loop && ae != -1) {
				mesh->vertices[newverts[i][j]] +=
					0.25f  * mesh->vertices[v[ j     ]] +
					0.375f * mesh->vertices[v[(j+1)%3]] +
					0.375f * mesh->vertices[v[(j+2)%3]];
			} else {
				mesh->vertices[newverts[i][j]] +=
					0.5f   * mesh->vertices[v[(j+1)%3]] +
					0.5f   * mesh->vertices[v[(j+2)%3]];
			}
			if (have_col) {
				unsigned char *c1 = mesh->colors[v[(j+1)%3]];
				unsigned char *c2 = mesh->colors[v[(j+2)%3]];
				mesh->colors[newverts[i][j]] =
					Color((c1[0]+c2[0])/2,
					      (c1[1]+c2[1])/2,
					      (c1[2]+c2[2])/2);
			}
			if (have_conf)
				mesh->confidences[newverts[i][j]] +=
					0.5f   * mesh->confidences[v[(j+1)%3]] +
					0.5f   * mesh->confidences[v[(j+2)%3]];
			newvert_count[newverts[i][j]]++;
		}
	}
	for (i = old_nv; i < mesh->vertices.size(); i++) {
		if (!newvert_count[i])
			continue;
		float scale = 1.0f / newvert_count[i];
		mesh->vertices[i] *= scale;
		if (have_conf)
			mesh->confidences[i] *= scale;
		// Colors don't need to be normalized
	}

	// Update old vertices
	if (Loop) {
		for (i = 0; i < old_nv; i++) {
			point bdyavg, nbdyavg;
			int nbdy = 0, nnbdy = 0;
			int naf = mesh->adjacentfaces[i].size();
			if (!naf)
				continue;
			for (int j = 0; j < naf; j++) {
				int af = mesh->adjacentfaces[i][j];
				int afi = mesh->faces[af].indexof(i);
				int n1 = (afi+1) % 3;
				int n2 = (afi+2) % 3;
				if (mesh->across_edge[af][n1] == -1) {
					bdyavg += mesh->vertices[newverts[af][n1]];
					nbdy++;
				} else {
					nbdyavg += mesh->vertices[newverts[af][n1]];
					nnbdy++;
				}
				if (mesh->across_edge[af][n2] == -1) {
					bdyavg += mesh->vertices[newverts[af][n2]];
					nbdy++;
				} else {
					nbdyavg += mesh->vertices[newverts[af][n2]];
					nnbdy++;
				}
			}

			float alpha;
			point newpt;
			if (nbdy) {
				newpt = bdyavg / (float) nbdy;
				alpha = 0.5f;
			} else if (nnbdy) {
				newpt = nbdyavg / (float) nnbdy;
				if (nnbdy == 6)
					alpha = 1.05;
				else if (nnbdy == 8)
					alpha = 0.86;
				else if (nnbdy == 10)
					alpha = 0.7;
				else
					alpha = 0.6;
			} else {
				continue;
			}
			mesh->vertices[i] *= 1.0f - alpha;
			mesh->vertices[i] += alpha * newpt;
		}
	}

	// Insert new faces
	mesh->adjacentfaces.clear(); mesh->across_edge.clear();
	mesh->faces.reserve(4*nf);
	for (i = 0; i < nf; i++) {
		TriMesh::Face &v = mesh->faces[i];
		TriMesh::Face &n = newverts[i];
		mesh->faces.push_back(TriMesh::Face(v[0], n[2], n[1]));
		mesh->faces.push_back(TriMesh::Face(v[1], n[0], n[2]));
		mesh->faces.push_back(TriMesh::Face(v[2], n[1], n[0]));
		v = n;
	}

	TriMesh::dprintf("Done.\n");
}

#pragma warning(pop)