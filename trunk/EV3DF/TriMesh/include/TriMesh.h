#ifndef TRIMESH_H
#define TRIMESH_H
/*
Szymon Rusinkiewicz
Princeton University

TriMesh.h
Class for triangle meshes.
*/

#include "TriMeshLib.h"
#include "Vec.h"
#include "Color.h"
#include <vector>


class TRIMESHDLL TriMesh {
public:
	// Types
	struct Face {
		int v[3];

		Face() {}
		Face(const int &v0, const int &v1, const int &v2)
			{ v[0] = v0; v[1] = v1; v[2] = v2; }
		Face(const int *v_)
			{ v[0] = v_[0]; v[1] = v_[1]; v[2] = v_[2]; }
		int &operator[] (int i) { return v[i]; }
		const int &operator[] (int i) const { return v[i]; }
		operator const int * () const { return &(v[0]); }
		operator const int * () { return &(v[0]); }
		operator int * () { return &(v[0]); }
		int indexof(int v_) const
		{
			return (v[0] == v_) ? 0 :
			       (v[1] == v_) ? 1 :
			       (v[2] == v_) ? 2 : -1;
		}
	};

	struct BBox {
		point min, max;
		point center() const { return 0.5f * (min+max); }
		vec size() const { return max - min; }
	};

	struct BSphere {
		point center;
		float r;
	};

	// Enums
	enum tstrip_rep { TSTRIP_LENGTH, TSTRIP_TERM };

	// The basics: vertices and faces
	std::vector<point> vertices;
	std::vector<Face> faces;

	// Triangle strips
	std::vector<int> tstrips;

	// Other per-vertex properties
	std::vector<Color> colors;
	std::vector<float> confidences;
	std::vector<unsigned> flags;
	unsigned flag_curr;

	// Computed per-vertex properties
	std::vector<vec> normals;
	std::vector<vec> pdir1, pdir2;
	std::vector<float> curv1, curv2;
	std::vector< Vec<4,float> > dcurv;
	std::vector<vec> cornerareas;
	std::vector<float> pointareas;

	// Bounding structures
	BBox bbox;
	BSphere bsphere;

	// Connectivity structures:
	//  For each vertex, all neighboring vertices
	std::vector< std::vector<int> > neighbors;
	//  For each vertex, all neighboring faces
	std::vector< std::vector<int> > adjacentfaces;
	//  For each face, the three faces attached to its edges
	//  (for example, across_edge[3][2] is the number of the face
	//   that's touching the edge opposite vertex 2 of face 3)
	std::vector<Face> across_edge;

	// Compute all this stuff...
	void need_tstrips();
	void convert_strips(tstrip_rep rep);
	void need_faces();
	void need_normals();
	void need_pointareas();
	void need_curvatures();
	void need_dcurv();
	void need_bbox();
	void need_bsphere();
	void need_neighbors();
	void need_adjacentfaces();
	void need_across_edge();

	// Input and output
	static TriMesh *read(const char *filename);
	void write(const char *filename);

	// Statistics
	// XXX - Add stuff here
	float feature_size();

	// Useful queries
	// XXX - Add stuff here
	bool is_bdy(int v)
	{
		if (neighbors.empty()) need_neighbors();
		if (adjacentfaces.empty()) need_adjacentfaces();
		return neighbors[v].size() != adjacentfaces[v].size();
	}

	// Debugging printout, controllable by a "verbose"ness parameter
	static int verbose;
	static void set_verbose(int);
	static int dprintf(const char *format, ...);
};

#endif
