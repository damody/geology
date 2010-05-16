
#include "FindSpan.h"
#include <vector>
#include <cmath>


//============================================================================
//
// �G�a, �ҥH�ڭ̪�data file�����ӭn��w�q�U���X�ӭ�
// 1. Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, deltaX, deltaY, deltaZ
// 2. Nx = (Xmax - Xmin) / deltaX : x �b�W cut�X��
//
//============================================================================

// �G��double���ȬO�_�۵� => Remove this when you go through the code,
//****************************************************************************
//
// * Two double are the same if their difference is smaller then a
//   sigma value
//============================================================================
inline bool IsEqual(double x1, double x2, double sigma)
//============================================================================
{
	if(abs(x1-x2) < sigma)
		return true;
	else
		return false;
}

// ��X�o�ռƦr����delta�ȸ��`�@���X��
//****************************************************************************
//
// * Find how many cuts in this set of numbers and also the delta and also find
//   the maximum and minimum
//============================================================================
double FindDeltaAndSpan ( double *values, int num, int step, double sigma, 
			double& vmin, double& vmax, double& delta, int& span)
//============================================================================
{
	std::vector<double> vals;
	vals.clear();

	vmin = values[0];
	vmax = values[step];

	// Go through all the number
	for(int i = 0; i < num; i+=step)
	{
		bool bGetOne = false;
		// Check whether it is maximum
		if(values[i] > vmax)
			vmax = values[i];
		// Check whether it is minimum
		if(values[i] < vmin)
			vmin = values[i];
		// Find whether we have saved the values or not
		for(unsigned int j = 0; j < vals.size(); j++)
		{
			if( IsEqual(vals[j], values[i], sigma))
			{
				bGetOne = true;
				break;
			}
		}
		if(bGetOne)
			continue;
		vals.push_back(values[i]);
	}
	span = vals.size() - 1;
	delta = (vmax - vmin) / (double)span;
	return delta;
}