#include "kd_tree.h"
#include <cassert>
#include <cmath>

bool cmpptr_x(const point_ptr& lhs, const point_ptr& rhs)
{
	return lhs.p->x < rhs.p->x;
}
bool cmpptr_y(const point_ptr& lhs, const point_ptr& rhs)
{
	return lhs.p->y < rhs.p->y;
}
bool cmpptr_z(const point_ptr& lhs, const point_ptr& rhs)
{
	return lhs.p->z < rhs.p->z;
}


int kd_tree::GetGoodSplitDimension( point_ptrs ps )
{
	int size = ps.size();
	// small range
	if (size < 3) return 0;
	sort(ps.begin(), ps.end(), cmpptr_x);
	float x_range = ps[size/2].p->x - ps[0].p->x;
	sort(ps.begin(), ps.end(), cmpptr_y);
	float y_range = ps[size/2].p->y - ps[0].p->y;
	sort(ps.begin(), ps.end(), cmpptr_z);
	float z_range = ps[size/2 ].p->z - ps[0].p->z;
	if (x_range >= y_range && x_range >= z_range)
		return DIMENSION_X;
	if (y_range >= x_range && y_range >= z_range)
		return DIMENSION_Y;
	return DIMENSION_Z;
}

int kd_tree::SplitSpace( kd_node* node, int dir, point_ptrs pps )
{
	int half = pps.size()/2;
	if (pps.size() <= 2)
	{
		for (int i=0;i<pps.size();++i)
			node->pts[i] = pps[i].idx;
		return 0;
	}
	switch (dir)
	{
	case DIMENSION_X:
		sort(pps.begin(), pps.end(), cmpptr_x);
		break;
	case DIMENSION_Y:
		sort(pps.begin(), pps.end(), cmpptr_y);
		break;
	case DIMENSION_Z:
		sort(pps.begin(), pps.end(), cmpptr_z);
		break;
	}
	point_ptrs left(pps.begin(), pps.begin()+half);
	point_ptrs right(pps.begin()+half, pps.end());
	int ldir = GetGoodSplitDimension(left);
	int rdir = GetGoodSplitDimension(right);
	int leftnode_idx = m_node_idx++;
	int rightnode_idx = m_node_idx++;
	// set left right
	node->left = leftnode_idx;
	node->right = rightnode_idx;
	node->dir = dir; // set dimension
	node->div = pps[half].p->p[dir]; // set split value
	m_nodes[leftnode_idx].idx = leftnode_idx;
	m_nodes[rightnode_idx].idx = rightnode_idx;
	memcpy(m_nodes[leftnode_idx].bounds,  node->bounds, sizeof(node->bounds));
	memcpy(m_nodes[rightnode_idx].bounds, node->bounds, sizeof(node->bounds));
	switch (dir)
	{
	case DIMENSION_X:
		m_nodes[leftnode_idx].bounds[1] = node->div;
		m_nodes[rightnode_idx].bounds[0] = node->div;
		break;
	case DIMENSION_Y:
		m_nodes[leftnode_idx].bounds[3] = node->div;
		m_nodes[rightnode_idx].bounds[2] = node->div;
		break;
	case DIMENSION_Z:
		m_nodes[leftnode_idx].bounds[5] = node->div;
		m_nodes[rightnode_idx].bounds[4] = node->div;
		break;
	} 
	SplitSpace(&m_nodes[leftnode_idx], ldir, left);
	SplitSpace(&m_nodes[rightnode_idx], rdir, right);
	return 0;
}

int kd_tree::BuildKdtree(const point* ps, int size )
{
	// set pointer to sort
	point_ptrs pps;
	const point* ops = ps;
	for (int i=0;i<size;++i)
	{
		pps.push_back(point_ptr(ps, i));
		++ps;
	}
	int dir = GetGoodSplitDimension(pps);
	m_nodes.resize(size*2);
	m_node_idx = 0;
	m_nodes[0].idx = 0;
	GetBounding(ops, size, m_nodes[0].bounds);
	SplitSpace(&m_nodes[m_node_idx++], dir, pps);
	m_nodes.resize(m_node_idx);
	ConvertFastKdtree();
	return 0;
}

void kd_tree::SearchKdtreeByBounding(point* pts, float* bound, int *res, int size )
{
	// init result
	for (int i=0;i<500;i++)
		res[i] = -1;
	// search
	int stack[MAX_STACK] = {0};
	int s_idx = 0, res_idx = 0;
#define push(X) stack[s_idx++] = (X)
#define get() stack[s_idx-1]
#define pop() --s_idx
	push(0);
	for (;s_idx != 0;)
	{
		kd_node* n = &m_nodes[get()];
		pop();
		if (n->dir == -1) // is leaf
		{
			if (n->pts[0] != -1 && IsCollision(pts[n->pts[0]], bound)) 
				res[res_idx++] = n->pts[0];
			if (res_idx >= size) return;
			if (n->pts[1] != -1 && IsCollision(pts[n->pts[1]], bound)) 
				res[res_idx++] = n->pts[1];
			if (res_idx >= size) return;
		}
		else if (BoundingCollision(n->bounds, bound))	// has node
		{
			kd_node* nn = &m_nodes[n->left];
			if (nn->dir != -1)
			push(n->left);
			push(n->right);
		}
	}
}

void kd_tree::ConvertFastKdtree()
{
	m_fnodes.resize(m_nodes.size());
	for (int i=0;i<m_nodes.size();++i)
	{
		m_fnodes[i].dir = m_nodes[i].dir;
		m_fnodes[i].left = m_nodes[i].left;
		m_fnodes[i].right = m_nodes[i].right;
		switch (m_nodes[i].dir)
		{
		case DIMENSION_X:
			m_fnodes[i].bounds[0] = m_nodes[i].bounds[0];
			m_fnodes[i].bounds[1] = m_nodes[i].bounds[1];
			break;
		case DIMENSION_Y:
			m_fnodes[i].bounds[0] = m_nodes[i].bounds[2];
			m_fnodes[i].bounds[1] = m_nodes[i].bounds[3];
			break;
		case DIMENSION_Z:
			m_fnodes[i].bounds[0] = m_nodes[i].bounds[4];
			m_fnodes[i].bounds[1] = m_nodes[i].bounds[5];
			break;
		default:
			m_fnodes[i].bounds[0] = 0;
			m_fnodes[i].bounds[1] = 0;
		}
	}
}

void kd_tree::FSearchKdtreeByBounding( point* pts, float* bound, int *res, int size )
{
	// init result
	for (int i=0;i<size;i++)
		res[i] = -1;
	// search
	int stack[MAX_STACK] = {0};
	int s_idx = 0, res_idx = 0;
#define push(X) stack[s_idx++] = (X)
#define get() stack[s_idx-1]
#define pop() --s_idx
	push(0);
	for (;s_idx != 0;)
	{
		kd_fast_node* n = &m_fnodes[get()];
		//printf("now: %d\n", get());
		pop();
		if (n->dir == -1) // is leaf
		{
			if (n->pts[0] != -1 && IsCollision(pts[n->pts[0]], bound))
			{
				res[res_idx++] = n->pts[0];
				if (res_idx >= size) return;
				if (n->pts[1] != -1 && IsCollision(pts[n->pts[1]], bound))
				{
					res[res_idx++] = n->pts[1];
					if (res_idx >= size) return;
				}
			}
		}
		else if (FBoundingCollision(bound, n->bounds, n->dir))	// has node
		{
			push(n->left);
			push(n->right);
		}
	}
}

bool kd_tree::FBoundingCollision( float* b1, float* range, int dir )
{
	int i = dir*2;
	if (b1[i] <= range[0] && b1[i+1] >= range[0])
		return true;
	if (b1[i] <= range[1] && b1[i+1] >= range[1])
		return true;
	if (range[0] <= b1[i] && range[1] >= b1[i])
		return true;
	if (range[0] <= b1[i+1] && range[1] >= b1[i+1])
		return true;
	return false;
}

bool kd_tree::BoundingCollision( float* b1, float* b2 )
{
	bool b[3] = {false};
	for (int i=0;i<5;i+=2)
	{
		if (b1[i] <= b2[i] && b1[i+1] >= b2[i])
			b[i/2] = true;
		if (b1[i] <= b2[i+1] && b1[i+1] >= b2[i+1])
			b[i/2] = true;
		if (b2[i] <= b1[i] && b2[i+1] >= b1[i])
			b[i/2] = true;
		if (b2[i] <= b1[i+1] && b2[i+1] >= b1[i+1])
			b[i/2] = true;
	}
	if (b[0] && b[1] && b[2])
		return true;
	return false;
}

void kd_tree::GetBounding( const point* pts, int size, float* bound )
{
	//xmin xmax, ymin ymax, zmin zmax
	bound[0] = pts->x;
	bound[1] = pts->x;
	bound[2] = pts->y;
	bound[3] = pts->y;
	bound[4] = pts->z;
	bound[5] = pts->z;
	for (int i=1;i<size;++i)
	{
		if (bound[0] > pts[i].x)
			bound[0] = pts[i].x;
		if (bound[1] < pts[i].x)
			bound[1] = pts[i].x;
		if (bound[2] > pts[i].y)
			bound[2] = pts[i].y;
		if (bound[3] < pts[i].y)
			bound[3] = pts[i].y;
		if (bound[4] > pts[i].z)
			bound[4] = pts[i].z;
		if (bound[5] < pts[i].z)
			bound[5] = pts[i].z;
	}
}

void kd_tree::GetBoundingByRadius( float x, float y, float z, float r, float* bounding )
{
	bounding[0] = x-r;
	bounding[1] = x+r;
	bounding[2] = y-r;
	bounding[3] = y+r;
	bounding[4] = z-r;
	bounding[5] = z+r;
}

bool kd_tree::IsCollision( float* p, float* bounding )
{
	if ((bounding[0] <= p[0] && bounding[1] >= p[0]) &&
		(bounding[2] <= p[1] && bounding[3] >= p[1]) &&
		(bounding[4] <= p[2] && bounding[5] >= p[2]))
		return true;
	return false;
}

float kd_tree::Distance_Point_Boundary( float x, float y, float z, float* bound )
{
	float dx = abs(bound[1]-bound[0]-x);
	float dy = abs(bound[3]-bound[2]-y);
	float dz = abs(bound[5]-bound[4]-z);
	return sqrt(dx*dx+dy*dy+dz*dz);
}
