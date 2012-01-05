// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
//  In academic purposes only(2012/1/12)

#pragma once

#include <vector>
#include <algorithm>

struct point
{
	point():x(0), y(0), z(0) {}
	point(float _x, float _y, float _z)
		:x(_x), y(_y), z(_z)
	{}
	union
	{
		struct 
		{
			float x, y, z;
		};
		float p[3];
	};
	operator float*()
	{
		return (float*)this;
	}
};
struct point_ptr
{
	point_ptr(const point* _p, int _idx)
		:p(_p), idx(_idx)
	{}
	const point* p;
	int idx;
};

struct kd_node
{
	char  dir;
	float div;
	float bounds[6];
	int idx;
	union
	{
		struct
		{
			int left, right;
		};
		int  pts[2];
	};
	kd_node():left(-1), right(-1), dir(-1)
	{
		pts[0] = -1;
		pts[1] = -1;
	}
};
struct kd_fast_node
{
	char  dir;
	float bounds[2];
	union
	{
		struct
		{
			int left, right;
		};
		int  pts[2];
	};
	kd_fast_node():left(-1), right(-1), dir(-1)
	{
		pts[0] = -1;
		pts[1] = -1;
	}
};

typedef std::vector<kd_node> kd_nodes;
typedef std::vector<kd_fast_node> kd_fast_nodes;
typedef std::vector<point> points;
typedef std::vector<point_ptr> point_ptrs;

class kd_tree
{
public:
	enum DIMENSION
	{
		DIMENSION_X = 0,
		DIMENSION_Y = 1,
		DIMENSION_Z = 2,
		MAX_STACK = 16
	};
	kd_nodes	m_nodes;
	kd_fast_nodes	m_fnodes;
	int		m_node_idx;
public:
	kd_tree():m_node_idx(0)
	{}
	int GetGoodSplitDimension(point_ptrs ps);
	int SplitSpace(kd_node* node, int dir, point_ptrs pps);
	int BuildKdtree(const point* ps, int size);
	void SearchKdtreeByBounding(point* pts, float* bound, int *res, int size);
	void FSearchKdtreeByBounding(point* pts, float* bound, int *res, int size);
	void ConvertFastKdtree();
	bool FBoundingCollision( float* b1, float* range, int dir );
	bool BoundingCollision( float* b1, float* b2 );
	void GetBounding( const point* pts, int size, float* bound );
	void GetBoundingByRadius( float x, float y, float z, float r, float* bounding );
	float Distance_Point_Boundary(float x, float y, float z, float* bounding);
	bool IsCollision( float* p, float* bounding );
};

// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
//  In academic purposes only(2012/1/12)