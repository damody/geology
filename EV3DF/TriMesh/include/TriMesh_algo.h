#ifndef TRIMESH_ALGO_H
#define TRIMESH_ALGO_H
/*
Szymon Rusinkiewicz
Princeton University

TriMesh_algo.h
Various mesh-munging algorithms using TriMeshes
*/


// Optimally re-triangulate a mesh by doing edge flips
extern void TRIMESHDLL edgeflip(TriMesh *mesh);

// Flip the order of vertices in each face.  Turns the mesh inside out.
extern void TRIMESHDLL faceflip(TriMesh *mesh);

// Create an offset surface from a mesh
extern void TRIMESHDLL inflate(TriMesh *mesh, float amount);

// One iteration of umbrella-operator smoothing
extern void TRIMESHDLL umbrella(TriMesh *mesh, float stepsize);

// Taubin lambda/mu mesh smoothing
extern void TRIMESHDLL lmsmooth(TriMesh *mesh, int niters);

// Remove the indicated vertices from the TriMesh.
extern void TRIMESHDLL remove_vertices(TriMesh *mesh, const std::vector<bool> &toremove);

// Remove vertices that aren't referenced by any face
extern void TRIMESHDLL remove_unused_vertices(TriMesh *mesh);

// Remove faces as indicated by toremove.  Should probably be
// followed by a call to remove_unused_vertices()
extern void TRIMESHDLL remove_faces(TriMesh *mesh, const std::vector<bool> &toremove);

// Remove long, skinny faces.  Should probably be followed by a
// call to remove_unused_vertices()
extern void TRIMESHDLL remove_sliver_faces(TriMesh *mesh);

// Reorder vertices in a mesh according to the order in which
// they are referenced by the tstrips or faces.
extern void TRIMESHDLL reorder_verts(TriMesh *mesh);

// Perform one iteration of subdivision on a mesh.
extern void TRIMESHDLL subdiv(TriMesh *mesh, bool Loop = true);

// Smooth the mesh geometry
extern void TRIMESHDLL smooth_mesh(TriMesh *themesh, float sigma);

// Diffuse the normals across the mesh
extern void TRIMESHDLL diffuse_normals(TriMesh *themesh, float sigma);

// Diffuse the curvatures across the mesh
extern void TRIMESHDLL diffuse_curv(TriMesh *themesh, float sigma);

// Diffuse the curvature derivatives across the mesh
extern void TRIMESHDLL diffuse_dcurv(TriMesh *themesh, float sigma);

// Given a curvature tensor, find principal directions and curvatures
extern void TRIMESHDLL diagonalize_curv(const vec &old_u, const vec &old_v,
			     float ku, float kuv, float kv,
			     const vec &new_norm,
			     vec &pdir1, vec &pdir2, float &k1, float &k2);

// Reproject a curvature tensor from the basis spanned by old_u and old_v
// (which are assumed to be unit-length and perpendicular) to the
// new_u, new_v basis.
extern void TRIMESHDLL proj_curv(const vec &old_u, const vec &old_v,
		      float old_ku, float old_kuv, float old_kv,
		      const vec &new_u, const vec &new_v,
		      float &new_ku, float &new_kuv, float &new_kv);

// Like the above, but for dcurv
extern void TRIMESHDLL proj_dcurv(const vec &old_u, const vec &old_v,
		       const Vec<4> old_dcurv,
		       const vec &new_u, const vec &new_v,
		       Vec<4> &new_dcurv);

#endif
