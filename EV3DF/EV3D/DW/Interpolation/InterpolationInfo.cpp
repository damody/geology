#include "InterpolationInfo.h"

void InterpolationInfo::GetPosFromXYZArray( float *data )
{
	for (int i=0;i<m_total;i++)
	{
		m_posAry[0][i] = data[i*4];
		m_posAry[1][i] = data[i*4+1];
		m_posAry[2][i] = data[i*4+2];
		m_data_ary[i] = data[i*4+3];
	}
}

void InterpolationInfo::GetPosFromXYZArray( double *data )
{
	for (int i=0;i<m_total;i++)
	{
		m_posAry[0][i] = (float)data[i*4];
		m_posAry[1][i] = (float)data[i*4+1];
		m_posAry[2][i] = (float)data[i*4+2];
		m_data_ary[i] = (float)data[i*4+3];
	}
}

InterpolationInfo::~InterpolationInfo()
{
	free(m_posAry[0]);
	free(m_posAry[1]);
	free(m_posAry[2]);
	free(m_data_ary);
}

InterpolationInfo::InterpolationInfo( int datasize )
{
	m_total = datasize;
	m_posAry[0] = (float*)malloc(sizeof(float)*datasize);
	m_posAry[1] = (float*)malloc(sizeof(float)*datasize);
	m_posAry[2] = (float*)malloc(sizeof(float)*datasize);
	m_data_ary = (float*)malloc(sizeof(float)*datasize);
}
