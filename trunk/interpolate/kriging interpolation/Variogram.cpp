
#include "Variogram.h"

#include <math.h>

#include <algorithm>
#include <functional>      // For greater<int>( )
#include <iostream>

#include "VariogramGaussianModel.h"
#include "VariogramStableModel.h"
#include "VariogramSphericalModel.h"

//#include "boost_mat.h"

//****************************************************************************
//
// * Default constructor
//============================================================================
Variogram::
Variogram(void)
//============================================================================
{
	m_pDistance		= NULL;
	m_pVariogram	= NULL;
	m_uNumSamples	= 0;

	m_VSill;
	m_VRange;
	m_VNugget;
	m_VPower;

	m_iModel		= VARIO_NONE;
}

//****************************************************************************
//
// * Destructor
//============================================================================
Variogram::
~Variogram(void)
//============================================================================
{
	Destroy();
}

//****************************************************************************
//
// * Memory allocation
//============================================================================
bool Variogram::
Allocate(const int smpl, const int numItem)
//============================================================================
{
	// Clean up the data
	Destroy();
	
	// Set up the memory for sample distance and variogram
	m_pDistance		= new double[smpl];
	m_pVariogram	= new vectord[smpl];

	// Fail to allocate the memory
	if (!m_pDistance|| !m_pVariogram) {
		Destroy();
		return false;
	}

	m_uNumDataItems = numItem;
	m_uNumSamples	= smpl;

	m_VNugget.resize(numItem);
	m_VPower.resize(numItem);
	m_VSill.resize(numItem);
	m_VRange.resize(numItem);

	return true;
}

//****************************************************************************
//
// * Memory destruction
//============================================================================
void Variogram::
Destroy(void)
//============================================================================
{
	if (m_pDistance != NULL) {
		delete[] m_pDistance;
		m_pDistance = NULL;
	}
	if (m_pVariogram != NULL) {
		delete[] m_pVariogram;
		m_pVariogram = NULL;
	}

	m_iModel	= VARIO_NONE;
	m_VNugget.clear();
	m_VSill.clear();
	m_VRange.clear();
	m_VPower.clear();
}


//****************************************************************************
//
// * Return whether the first element is greater than the second
//============================================================================
bool UDGreater (VariogramItem& elem1, 
				VariogramItem& elem2 )
//============================================================================
{
	return elem1.m_dDistance < elem2.m_dDistance;
}


//****************************************************************************
//
// * Set of variogram cloud
//   vecSample [in] sample container (distance & dissimilarity)
//   We will first sort the vecSample and then copy into data
//============================================================================
bool Variogram::
SetSample(std::vector<VariogramItem> &vecSample)
//============================================================================
{
	// Check whether we have sample data
	if (vecSample.empty())
		return false;

	// Allocate the memory
	if (!Allocate(vecSample.size(), vecSample[0].m_VDissimilarity.size()) )
		return false;

	// Sort the element by distance
	sort(vecSample.begin(), vecSample.end(), UDGreater);

	// Copy the data into our data structure
	for (int i = 0;i < m_uNumSamples; i++) {
		m_pDistance[i]	= vecSample[i].m_dDistance;
		m_pVariogram[i] = vecSample[i].m_VDissimilarity;
	}
	
	// Set up the operation model to none
	m_iModel = VARIO_NONE;

	return true;
}

//****************************************************************************
//
// * Set of theoretical variogram model and it have using the maxdist 
//   => This should have problem because where is the model data come from
//
// @param model [in] theoretical variogram model
// @param nugget [in] nugget
// @param sill [in] sill
// @param range [in] range
// @param power [in] power coefficient of stable variograms
// @param step [in] step width of reogionalization
// @param maxdist [in] range of experimental variogram
// @retval true success
//============================================================================
bool Variogram::
SetModel(int model, vectord& nugget, vectord& sill, 
		vectord range, vectord& power, double step, 
		double maxdist, int numDataItem)
//============================================================================
{
	// Currently only implement to number
	if (model >= Variogram::VARIO_NUM)
		return false;

	// Will this conflict with the SetSample ????
	if (!Allocate(static_cast<int>(fabs(maxdist / step)), numDataItem))
		return false;

	m_iModel	= model;
	m_VNugget	= nugget;
	m_VSill		= sill;
	m_VRange	= range;
	m_VPower	= power;

	// Set up the sample values
	for (int i = 0; i < m_uNumSamples;i++)	{
		m_pDistance[i]	= step * i;
		// => allocate will kill all data how does the GetModelData comes from????
		m_pVariogram[i] = GetModelData(step * i);
	}
	return true;
}

//****************************************************************************
//
// *  Get sample data
//============================================================================
bool Variogram::
GetSample(const int smpl, double &dist, vectord &vario) const
//============================================================================
{
	if (smpl >= m_uNumSamples)
		return false;

	dist	= m_pDistance[smpl];
	vario	= m_pVariogram[smpl];
	return true;
}

//****************************************************************************
//
// * Get number of samples whose distance is less than given threshold
//============================================================================
int Variogram::
CountLessDist(const double cap) const
//============================================================================
{

	int cnt =0;
	while (++cnt < m_uNumSamples && m_pDistance[cnt] < cap)
		;
	return cnt;
}


//****************************************************************************
//
// * Estimation of theoretical variogram from experimental variogram 
//
//   by non-linear least square fitting
// * @param nugget [out] nugget
// * @param sill [out] sill
// * @param range [out] range
// * @param power [in] power coefficient of stable variograms
// * @param maxdist [in] range of distance for estimation
//============================================================================
bool Variogram::
EstimateModel(int model, vectord &nugget, vectord &sill, vectord &range, 
			  vectord power, double maxdist)
//============================================================================
{
	for(int i = 0; i < m_uNumDataItems; i++){
		// Check the power
		if (power[i] < 0.0 || power[i] > 2.0)
			return false;
	}

	// Construct the vector and matrix
	// For the function operation
	void (*valueDerivative)(const DP, Vec_I_DP &, DP &, Vec_O_DP &);
	m_iModel = model;
	switch (model)	{
		case VARIO_SPERICAL:
			valueDerivative = SphericalValueDerivative;
			break;
		case VARIO_STABLE:
			valueDerivative = StableValueDerivative;
			break;
		case VARIO_GAUSSIAN:
			valueDerivative = GaussianValueDerivative;
			break;
		default:
			m_iModel = VARIO_NONE;
			return false;
	}

	// Count the total number of valid samples
	int numValidSamples	= CountLessDist(maxdist);


   	if (numValidSamples>15)
   		numValidSamples = 15;



	// Set up the sigma for each one of them
	Vec_DP  sig(numValidSamples);
	for(int i = 0; i < numValidSamples; i++){
		sig[i] = 1.0;
	}

	// Index whether any one of them are needed for consideration
	Vec_BOOL ia(4);
	ia[0] = ia[1] = ia[2] =true;
	ia[3] = false;
	DP tol = 1;                   // How should we set up the tolerance

	// Set up the sample distance
	Vec_DP  x(numValidSamples);
	for(int s = 0; s < numValidSamples; s++){
		x[s] = m_pDistance[ s ];
	}

	for(int i = 0; i < 1; i++) {

		// Set up the initial guess
		Vec_IO_DP a(4);
		a[0] = nugget[i];
		a[1] = sill[i]; 
		a[2] = range[i]; 
		a[3] = power[i];

		Mat_DP   covar(4, 4);
		Mat_DP   alpha(4, 4);
		DP chisq;
		DP alamda = -0.001;				// From Numerical recipe

		// Set up the sample data
		Vec_DP  y(numValidSamples);
		for(int s = 0; s < numValidSamples; s++){
			y[s] = m_pVariogram[s][i];
		}


		int iter =0;

		double stc_chisq = 0;		//static chisq
		int same_cnt = 0, max_cnt = 5;
		// iteration
		do 	{
			iter++;

			// do the nonlinear fitting process
			NR::mrqmin(x, y, sig, a, ia, covar, alpha, chisq, valueDerivative, alamda);

			if (fabs(stc_chisq-chisq)<=0.0001 )
			{
				same_cnt++;
			}
			else
			{
				same_cnt=0;
				stc_chisq = chisq;
			}

			if(chisq < tol || same_cnt >= max_cnt){
				alamda=0.0;
				NR::mrqmin(x, y, sig, a, ia, covar, alpha, chisq, valueDerivative, alamda);
				break;
			}
		} while (iter < 1000);

		if (iter >= 1000) {
			printf("Fail to converge at %d data with chisqare is %f\n", i, chisq);
			return false;
		}

		nugget[i]	= m_VNugget[i]	= a[0];
		sill[i]		= m_VSill[i]	= a[1];
		range[i]	= m_VRange[i]	= a[2];
// 		if (range[i]<0)
 //			m_VRange[i]*=-1;
		power[i]	= m_VPower[i]	= a[3];
	}

	return true;
}

//****************************************************************************
//
// * get theoretical dissimilarity
// @param dist [in] distance
// @return dissimilarity
//============================================================================
vectord Variogram::
GetModelData(const double dist) const
//============================================================================
{
	vectord ret;
	ret.resize(m_uNumDataItems);

	double (*value) (double, double, double, double, double);

	switch (m_iModel)	{
		case VARIO_SPERICAL:
			value = Spherical;
			break;
		case VARIO_STABLE:
			value =  Stable;
			break;
		case VARIO_GAUSSIAN:
			value = Gaussian;
			break;
		default:
			break;
	}
	for(int i = 0; i < m_uNumDataItems; i++){
		//ret[i] = (float) value(0, 1, 50, 2, (double)dist);
		ret[i] = value(m_VNugget[i], 
			m_VSill[i], m_VRange[i], 
			m_VPower[i], dist);
	}
	return ret;
}

//****************************************************************************
//
// * Get covariance corresponding to theoretical dissimilarity
//============================================================================
vectord Variogram::
GetModelCovariance(const double dist) const
//============================================================================
{

	vectord ret;
	ret.resize(m_uNumDataItems);

	double (*value) (double, double, double, double, double);

	switch (m_iModel)	{
		case VARIO_SPERICAL:
			value = Spherical;
			break;
		case VARIO_STABLE:
			value =  Stable;
			break;
		case VARIO_GAUSSIAN:
			value = Gaussian;
			break;
		default:
			break;
	}
	for(int i = 0; i < m_uNumDataItems; i++){
		ret[i] =  value(m_VNugget[i], m_VSill[i], m_VRange[i], m_VPower[i], dist);
	}
	return ret;
}

double Variogram::GetDistance( int n )
{
	if (n>=0 && n < m_uNumSamples) 
		return m_pDistance[n]; 
	else 
		return -1;
}

