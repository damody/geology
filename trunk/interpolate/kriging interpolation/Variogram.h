#pragma once

#include <vector>
//#include "mrqmin.h"
#include <nr.h>
typedef std::vector<double> vectord;

// Structure of distance and dissimilarity: VarioItm -> LYCVariogramItem
struct VariogramItem{
	//! Distance => The distance to the center
	double m_dDistance;
	//! Dissimilarity => The dissimilarity between theoretical values and the measurement in distance r(h) = (r1 - r2)^2 
	vectord m_VDissimilarity;
};

class Variogram
{
	// member enumeration
public:
	enum VariogramModel {
		//! not estimated
		VARIO_NONE,
		//! spherical model
		VARIO_SPERICAL,
		//! stable model -> Exponential data format
		VARIO_STABLE,
		//! Gaussian models
		VARIO_GAUSSIAN,
		//! number of models
		VARIO_NUM,
	};

	// Member classes
public:



	// Constructor / destructor
public:
	// Default constructor
	Variogram(void);

	// Destructor
	virtual ~Variogram(void);

	// Attributes
public:

	// The total number of samples
	int   NumSamples(void) const { return m_uNumSamples; }

	// The total number of samples
	int   NumDataItems(void) const { return m_uNumDataItems;}

	// Return the maximum distance  -> because the data in variogram is sorting in order
	double MaxDistance(void) const { return m_pDistance[m_uNumSamples - 1];	}

	// Return minimum distance
	double MinDistance(void) const{	return m_pDistance[0];}

	// Return the value of nugget
	vectord Nugget(void) const{	return m_VNugget; }

	// Return the value of sill
	vectord Sill(void) const{ return m_VSill; }

	// Return the value of range
	vectord Range(void) const{	return m_VRange; }

	// Power coefficient (only for stable models)
	vectord Power(void) const{	return m_VPower; }

	// Return type of theoretical variogram
	int ModelType(void) const{	return m_iModel; }

	// Get number of samples whose distance is less than given value
	int CountLessDist(double cap) const;

	// Data reader
public:
	// Get sample data (distance & dissimilarity)
	bool GetSample(const int smpl, double &dist, vectord &vario) const;

	// Get theoretical dissimilarity => do the estimation from the model created
	vectord GetModelData(const double dist) const;

	// Get covariance corresponding to theoretical dissimilarity from the model created
	vectord GetModelCovariance(const double dist) const;

	double GetDistance(int n);

	double GetNumOfSmp(){return m_uNumSamples;}

	// Data writer
public:
	// Set sample data (distance & dissimilarity)
	bool SetSample(std::vector<VariogramItem> &vecSample);

	// Set parameters of theoretical variogram (nugget, sill, range, power, and so on)
	bool SetModel(int model, vectord& nugget, vectord& sill, 
		vectord range, vectord& power, double step, 
		double maxdist, int numDataItems);

public:
	// Estimate theoretical variogram by non-linear least square fitting
	bool EstimateModel(int model, vectord &nugget, vectord &sill, 
		vectord &range, vectord power, double maxdist =1.0e6);

	// memory manager
private:
	// memory allocation
	bool Allocate(const int smpl, const int numDataItems);

	// memory destruction => clear up the data
	void Destroy(void);

	// variables
private:
	//! Theoretical variogram model
	int				m_iModel;

	//! Number of samples
	int			m_uNumSamples;

	//! Number of data items
	int          m_uNumDataItems;

	//! Container for distance
	double			*m_pDistance;

	//! Container for experimental variogram
	vectord	*m_pVariogram;

	// ***********************************************************************
	//
	// Parameters of theoretical variogram
	//
	// ***********************************************************************

	// Spherical control parameter
	//! nugget
	vectord		m_VNugget;

	//! sill
	vectord		m_VSill;

	//! range
	vectord		m_VRange;

	// Stable control parameters
	//! power coefficient of stable model
	vectord		m_VPower;
};
