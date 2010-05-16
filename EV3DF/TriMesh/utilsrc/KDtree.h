#ifndef KDTREE_H
#define KDTREE_H
/*
Szymon Rusinkiewicz
Princeton University

KDtree.h
A K-D tree for points, with limited capabilities (find nearest point to 
a given point, or to a ray). 
*/

#include <vector>

class KDtree {
private:
	mutable const float *cache;
	class Node;
	Node *root;
	void build(const float *ptlist, int n);

public:
	struct CompatFunc { virtual bool operator () (const float *p) const = 0; };

	KDtree(const float *ptlist, int n)
		{ build(ptlist, n); }
	template <class T> KDtree(std::vector<T> &v)
		{ build((const float *) &v[0], v.size()); }
	~KDtree();

	const float *closest_to_pt(const float *p,
				   float maxdist2,
				   const CompatFunc *iscompat = NULL) const;
	const float *closest_to_ray(const float *p, const float *dir,
				    float maxdist2,
				    const CompatFunc *iscompat = NULL) const;

};

#endif
