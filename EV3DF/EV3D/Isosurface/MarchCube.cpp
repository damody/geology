#include <math.h>

#include "MarchCube.h"

//*****************************************************************************
//
// * Linearly interpolate the position where an isosurface cuts
//   an edge between two vertices, each with their own scalar value
//==============================================================================
SJCVector3d VertexInterp(double isolevel, SJCVector3d p1, SJCVector3d p2, 
			 double valp1, double valp2)
//==============================================================================
{
	
   double	mu;
   SJCVector3d	p;

   // Check whether the two value is the same
   if (SJCAbs(isolevel - valp1) < SJC_EPSILON)
      return(p1);
   if (SJCAbs(isolevel - valp2) < SJC_EPSILON)
      return(p2);
   if (SJCAbs(valp1 - valp2) < SJC_EPSILON)
      return(p1);

   mu = (isolevel - valp1) / (valp2 - valp1);

   p.x( p1.x() + mu * (p2.x() - p1.x()));
   p.y( p1.y() + mu * (p2.y() - p1.y()));
   p.z( p1.z() + mu * (p2.z() - p1.z()));

   return(p);
}

//*****************************************************************************
//
// * Given a grid cell and an isolevel, calculate the triangular
//   facets required to represent the isosurface through the cell.
//   Return the number of triangular facets, the array "triangles"
//   will be loaded up with the vertices at most 5 triangular facets.
//   0 will be returned if the grid cell is either totally above
//   of totally below the isolevel.
//==============================================================================
int Polygonise(SMarchCubeGridCell grid, double isolevel, 
	       SMarchCubeTriangle *triangles)
//==============================================================================
{
	int		cubeindex;    // Index of cube
	SJCVector3d	vertlist[12]; // Vertex list

	// Determine the index into the edge table which
	// tells us which vertices are inside of the surface
	cubeindex = 0;
	if (grid.val[0] < isolevel) 
		cubeindex |= 1;
	if (grid.val[1] < isolevel) 
		cubeindex |= 2;
	if (grid.val[2] < isolevel) 
		cubeindex |= 4;
	if (grid.val[3] < isolevel) 
		cubeindex |= 8;
	if (grid.val[4] < isolevel) 
		cubeindex |= 16;
	if (grid.val[5] < isolevel) 
		cubeindex |= 32;
	if (grid.val[6] < isolevel) 
		cubeindex |= 64;
	if (grid.val[7] < isolevel) 
		cubeindex |= 128;

	// Cube is entirely in/out of the surface
	if (g_VEdgeTable[cubeindex] == 0)
		return(0);

	// Find the vertices where the surface intersects the cube
	if (g_VEdgeTable[cubeindex] & 1)
		vertlist[0] =
			VertexInterp(isolevel, grid.p[0], grid.p[1], grid.val[0], grid.val[1]);
	if (g_VEdgeTable[cubeindex] & 2)
		vertlist[1] =
			VertexInterp(isolevel, grid.p[1], grid.p[2],grid.val[1], grid.val[2]);
	if (g_VEdgeTable[cubeindex] & 4)
		vertlist[2] =
			VertexInterp(isolevel, grid.p[2], grid.p[3], grid.val[2], grid.val[3]);
	if (g_VEdgeTable[cubeindex] & 8)
		vertlist[3] =
			VertexInterp(isolevel, grid.p[3], grid.p[0], grid.val[3], grid.val[0]);
	if (g_VEdgeTable[cubeindex] & 16)
		vertlist[4] =
			VertexInterp(isolevel, grid.p[4], grid.p[5], grid.val[4], grid.val[5]);
	if (g_VEdgeTable[cubeindex] & 32)
		vertlist[5] =
			VertexInterp(isolevel, grid.p[5], grid.p[6], grid.val[5], grid.val[6]);
	if (g_VEdgeTable[cubeindex] & 64)
		vertlist[6] =
			VertexInterp(isolevel, grid.p[6], grid.p[7], grid.val[6], grid.val[7]);
	if (g_VEdgeTable[cubeindex] & 128)
		vertlist[7] =
			VertexInterp(isolevel, grid.p[7], grid.p[4], grid.val[7], grid.val[4]);
	if (g_VEdgeTable[cubeindex] & 256)
		vertlist[8] =
			VertexInterp(isolevel, grid.p[0], grid.p[4], grid.val[0], grid.val[4]);
	if (g_VEdgeTable[cubeindex] & 512)
		vertlist[9] =
			VertexInterp(isolevel, grid.p[1], grid.p[5], grid.val[1], grid.val[5]);
	if (g_VEdgeTable[cubeindex] & 1024)
		vertlist[10] =
			VertexInterp(isolevel, grid.p[2], grid.p[6], grid.val[2], grid.val[6]);
	if (g_VEdgeTable[cubeindex] & 2048)
		vertlist[11] =
			VertexInterp(isolevel, grid.p[3], grid.p[7], grid.val[3], grid.val[7]);

	// Create the triangle 
	int ntriang = 0;
	for (int i = 0; g_VTriangleTable[cubeindex][i] != -1; i += 3) {
		triangles[ntriang].p[0] = vertlist[g_VTriangleTable[cubeindex][i  ]];
		triangles[ntriang].p[2] = vertlist[g_VTriangleTable[cubeindex][i+1]];
		triangles[ntriang].p[1] = vertlist[g_VTriangleTable[cubeindex][i+2]];
		ntriang++;
	}

	return(ntriang);
}

//*****************************************************************************
//
// * Add a vertex to the list and return its index
//
//   if it's already in the list (closer than hash function defined to another point), 
//   skip it and return the existing index
//==============================================================================
int AddVertex(SJCScalarField3d  *pLevelSet, 
	      TriMesh		*pOutMesh, 
	      SJCVector3d	*pVertex, 
	      stdext::hash_map<SJCVector3d, int, CMarchCubeVectorHash> *hmap)
//==============================================================================
{
	int retIndexOfVertex = -1;
	point p    = point(pVertex->x(), pVertex->y(), pVertex->z());

	// Check whether we have saved the vertex yet
	if (hmap->count(*pVertex) > 0)
		// Return the index of the vertex
		retIndexOfVertex = (*hmap)[*pVertex];
	else{
		// Compute the normal direction
		SJCVector3d grad = pLevelSet->Grad(pVertex->x(), pVertex->y(), pVertex->z());
		grad.normalize();

		// Get the index of the current 
		retIndexOfVertex = pOutMesh->vertices.size();
		
		// PUsh in the vertex
		pOutMesh->vertices.push_back(p);

		// Push in the normal of the vertex
		pOutMesh->normals.push_back(vec(grad.x(), grad.y(), grad.z()));

		// Set up the hash value
		(*hmap)[*pVertex] = retIndexOfVertex;
	}

	return retIndexOfVertex;
}

//****************************************************************************
//
// * Do marching cubes on the level set
//============================================================================
TriMesh *MarchCubes(SJCScalarField3d  *pLevelSet, double isolevel)
//============================================================================
{
	SJCVector3d pos;

	// Create the new triangle mesh
	TriMesh *pOutMesh = new TriMesh();

	SMarchCubeGridCell newCell;
	SMarchCubeTriangle pTriangles[5];

	// Create the hash map
	stdext::hash_map<SJCVector3d, int, CMarchCubeVectorHash> hmap;

	// Go through all vertices in the cube
	for(uint i = 0;i < pLevelSet->NumX() - 1;i++) {
		for(uint j = 0;j < pLevelSet->NumY() - 1; j++) {
			for(uint k = 0; k < pLevelSet->NumZ() - 1; k++) {
				uint n = 0;
				for(int dz = 0;dz <= 1; dz++) {
					for(int dy = 0;dy <= 1; dy++) {
						for(int dx = 0;dx <= 1;dx++) {
							int realdx = (dy == 1) ?  1 - dx  : dx;
							// Set levelset coordinate
							pLevelSet->Coord(i + realdx, j + dy, k + dz, pos);
							// Set up the cell position
							newCell.p[n].x( pos.x());
							newCell.p[n].y( pos.y());
							newCell.p[n].z( pos.z());

							// Get the cell value
							newCell.val[n] = pLevelSet->Value(pos.x(), pos.y(), pos.z());
							n++;
						} // end of for dx
					} // end of for dy
				} // end of for dz

				// Generate the polygon from the cell
				uint nTriangles = Polygonise(newCell, isolevel, pTriangles);

				// Add in the triangles
				for(uint m = 0; m < nTriangles; m++)	{
					uint i1 = AddVertex(pLevelSet, pOutMesh, 
							    &pTriangles[m].p[0], &hmap);
					uint i2 = AddVertex(pLevelSet, pOutMesh, 
							    &pTriangles[m].p[1], &hmap);
					uint i3 = AddVertex(pLevelSet, pOutMesh, 
						            &pTriangles[m].p[2], &hmap);

					pOutMesh->faces.push_back(TriMesh::Face(i1, i2, i3));
				} // end of for n
			} // end of for k
		} // end of for j
	} // end of for i

	return pOutMesh;
}
