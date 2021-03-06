// author: t1238142000@gmail.com Liang-Shiuan Huang ���G�a
// author: a910000@gmail.com Kuang-Yi Chen ������
//  In academic purposes only(2012/1/12)

#pragma once

#include "Variogram.h"
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
typedef boost::numeric::ublas::matrix<double> matrixd;

//3-D double position struct
struct dPos3
{
	double pos[3];
	dPos3()
	{
		pos[0] = pos[1] = pos[2] = 0;
	}
	dPos3(double _pos[3])
	{
		pos[0] = _pos[0]; pos[1] = _pos[1]; pos[2] = _pos[2];
	}
	double GetDist(dPos3& _pos)
	{
		double res=0;
		for (int i=0;i<3;i++)
		{
			res+= pow( pos[i]-_pos.pos[i], 2.0 );
		}

		return sqrt(res);	
	}
};

//data record struct, include a 3-D position and a value
struct Record3
{
	dPos3 pos;
	double val;

	Record3():val(0){}
	Record3(double _pos[3], double _val){
		pos.pos[0] = _pos[0]; pos.pos[1] = _pos[1]; pos.pos[2] = _pos[2];
		val = _val;
	}
	Record3(dPos3 _pos, double _val){
		pos = _pos;
		val = _val;
	}
};

//kriging class, only for  Ordinary kriging
class kriging
{
private:
	std::vector<Record3> m_dataset;			//output grid data
	std::vector<Record3> m_smpset;			//input sample set data
	
	int			m_uNumSamples;				//number of samples point
	int			m_uNumDataItems;			//number of data item for a samples point
	double		m_DistStep;					//cloud variogram distance step
	bool		m_AutoSetDistStep;			//whether use autometic to get distance step
	//! Control space domain -> record the max and min of each domain
	double			m_vLower[3];
	double			m_vUpper[3];
	double			m_Interval[3];
	//! Variogram cloud
	Variogram*	m_pVarioCloud;
	//! Experimental/theoretical variogram
	Variogram*	m_pVariogram;
	//! Precomputed kriging system ???
	matrixd		m_pPrePseudoInverse;

public:
	double GetDistStep()				//get cloud variogram distance step
	{return m_DistStep;};
	kriging(void);
	~kriging(void){Destroy();}

	bool Initial(int smpl, int numDataItems);		//initial kriging, set number of smaple and data items

	void SetSample(std::vector<Record3>& smp){m_smpset = smp;}
	void SetRange(double rmax[3], double rmin[3] );

	void SetInterval(double itv[3])
	{m_Interval[0] = itv[0]; m_Interval[1] = itv[1]; m_Interval[2] = itv[2];}

	Record3 GetData(int index);

	// Estimate trend surface and theoretical variogram
	virtual bool Estimate(int model, double power, double step);

	double kriging::GetPredictData( double pos[], double &pred, double &var ) const;

	double GetTrend(double pos[]) const;

	int GetDataSize(){return m_dataset.size();}

	void SetAutoGetDistStep(bool setting){m_AutoSetDistStep = setting;}

	void GetDoubleMatrix(std::vector<double>& mat);
	void GetFloatMatrix(std::vector<float>& mat);
	void GetFloatMatrix(float* mat);

	float GetNugget(){return (float)m_pVariogram->Nugget()[0];}
	float GetSill(){return (float)m_pVariogram->Sill()[0];}
	float GetRange(){return (float)m_pVariogram->Range()[0];}

protected:

	// Memory allocation
	bool Allocate(int smpl, int numDataItems);

	// Memory destruction
	void Destroy(void);

	// Compute variogram cloud
	std::vector<VariogramItem> ComputeVariogramCloud(void) const;

	// Compute experimental variogram
	std::vector<VariogramItem> 
		ComputeExperimentalVariogram(const std::vector<VariogramItem> &vcloud, double step) const;

	// Pre-compute kriging system
	bool PrecomputeKrigingSystem(void);

	void AutoSetDistSetp();
};

// author: t1238142000@gmail.com Liang-Shiuan Huang ���G�a
// author: a910000@gmail.com Kuang-Yi Chen ������
//  In academic purposes only(2012/1/12)