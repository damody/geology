#include "Kriging.h"
#include <map>

kriging::kriging( void )
{
	m_AutoSetDistStep = true;
	m_pVariogram = NULL; 
	m_pVarioCloud = NULL;
	m_DoPreCompute = true;
}

void kriging::SetRange( double rmax[3], double rmin[3] )
{
	for (int i=0; i<3; i++)
	{
		m_vUpper[i] = rmax[i];
		m_vLower[i] = rmin[i];
	}
}

Record3 kriging::GetData( int index )
{
	if (index<(int)m_dataset.size())
		return m_dataset[index];
}

//****************************************************************************
//
// * Compute variogram cloud
//============================================================================
std::vector<VariogramItem> kriging::ComputeVariogramCloud( void ) const
{
	// Check if not active return nothing.
	std::vector<VariogramItem>	vecVario;

	double  r1;
	dPos3 pos1, pos2;
	VariogramItem	itm;
	itm.m_VDissimilarity.resize(1);


	// Go through all sample once
	for (int i = 0; i < (int)m_smpset.size() - 1;i++) {


		//m_smpset[i].GetPos(pos1);
		pos1 = m_smpset[i].pos;
		r1 = m_smpset[i].val;
		// Get the residual from prediction
		double r1 = m_smpset[i].val;

		// Go through the different part of samples
		for (int j = i + 1;j < (int)m_smpset.size();j++) {

			double r2 = m_smpset[j].val;
			//m_smpset[j].GetPos(pos2);
			pos2 = m_smpset[j].pos;
			
			// Set up the distance
			itm.m_dDistance = pos1.GetDist(pos2);


			// (R1 - R2)^2 ??????? recheck this part
			double d = r1 - r2;
			itm.m_VDissimilarity[0] = (d * d / 2.0);
			vecVario.push_back(itm);

		} // end of for j
	} // end of for i 

	return vecVario;
}



double kriging::IsoDis( double* p1, double* p2 )
{
	double res=0;
	for (int i=0;i<3;i++)
	{
		res+= pow( p1[i]-p2[i], 2.0 );
	}

	return sqrt(res);
}

//****************************************************************************
//
// * Compute experimental variogram
//
// 1. vcloud [in] variogram cloud
// 2. step [in] step width of equally devided region
//============================================================================
std::vector<VariogramItem> kriging::ComputeExperimentalVariogram( 
	const std::vector<VariogramItem> &vcloud, double step ) const
{
	// Use map to condense the data. Go through all variogram data
	std::map<int, VariogramItem> variomap;
	for (int i = 0; i < (int)vcloud.size(); i++) {
		// Compute the key which is the integer part of the step
		int key = (int)ceil(vcloud[i].m_dDistance / step);

		// If we cannot find the corresponding key
		if (variomap.find(key) == variomap.end())	{
			variomap[key].m_dDistance = 1.0;
			variomap[key].m_VDissimilarity = vcloud[i].m_VDissimilarity;
		}
		else {
			variomap[key].m_dDistance += 1.0;
			for (int j = 0 ; j < m_uNumDataItems; j++){
				variomap[key].m_VDissimilarity[j] += vcloud[i].m_VDissimilarity[j];
			} // end of for j
		} // end of else
	} // end of for i

	// Go through all element
	std::vector<VariogramItem>	 vmodel;

	FILE* expv_file = fopen("ExperimentalVariogram", "r");
	for (std::map<int, VariogramItem>::const_iterator it = variomap.begin();
		it != variomap.end(); ++it) {

			// Compute the item distance
			VariogramItem	 itm;
			itm.m_dDistance = (float) (it->first * step);

			// Normalize the value
			for (int j = 0 ; j < m_uNumDataItems; j++){
				itm.m_VDissimilarity.push_back(it->second.m_VDissimilarity[j] / it->second.m_dDistance);
			}
			vmodel.push_back(itm);
	}

	return vmodel;
}

bool kriging::PrecomputeKrigingSystem( void )
{
	// Get the number of samples
	int numSamples     = m_smpset.size();

	// Get the dimension for the array
	int dim			= numSamples + 1;
	dPos3 pos1, pos2;

	Mat_DP tmp = Mat_DP(dim, dim);
	Mat_DP ivtmp = Mat_DP(dim, dim);

	//vectord dataset;

	for (int s1 = 0; s1 < numSamples; s1++)	{
		// Get the position
		//m_smpset[s1].GetPos(pos1);
		pos1 = m_smpset[s1].pos;
		for (int s2 = s1; s2 < numSamples; s2++) {
			pos2 = m_smpset[s2].pos;
			double dist = pos1.GetDist(pos2);

			std::vector<double> data = m_pVariogram->GetModelData(dist);

			tmp[s1][s2] = tmp[s2][s1] = data[0]; 
			
		} // end of s2
	} // end of s1


	for (int s = 0; s < numSamples; s++){
		tmp[s][numSamples] = 1.0;
		tmp[numSamples][s] = 1.0;
	}

	// Set up the 0.0
	Mat_DP testmat(dim, dim);
	matrixd testmatD(dim, dim);
	tmp[numSamples][numSamples] = 0.0;

	NR::LYCSVDPseudoInverse(ivtmp , tmp);

	for (int i1=0; i1<ivtmp.nrows(); i1++)
	{
		for (int i2=0; i2<ivtmp.ncols(); i2++)
		{
			m_pPrePseudoInverse(i1, i2) = ivtmp[i1][i2];
		}
	}

	return true;
}

bool kriging::Allocate( int smpl, int numDataItems )
{
	// Clear up the data
	Destroy();

	// Allocate the sample data
	//m_dataset.resize(smpl);
	m_uNumSamples   = smpl;

	// Create the variogram for each one
	m_pVarioCloud	= new Variogram;
	m_pVariogram	= new Variogram;

	// Set up the matrix
	m_uNumDataItems = numDataItems;

	m_pPrePseudoInverse = matrixd(smpl + 1, smpl + 1, 0);

	return true;
}

void kriging::Destroy( void )
{
	// Clear up the sample data
	m_dataset.clear();
	m_uNumSamples		= 0;

	// Clear up the variogram
	if (m_pVariogram != NULL) {
		delete m_pVariogram;
		m_pVariogram = NULL;
	}

	// Clear up the variogram cloud
	if (m_pVarioCloud != NULL) {
		delete m_pVarioCloud;
		m_pVarioCloud = NULL;
	}

	m_uNumDataItems	= 0;

}

double kriging::GetPredictData( double pos[], double &pred, double &var ) const
{
	// Get the number of samples
	int numSamples = m_smpset.size();

	// Get the dimension of arrays
	int dim = numSamples + 1;


	// Allocate the temporarily data
	vectord vvario(dim);
	vectord weight(dim);
	dPos3 spos, npos(pos);

	for (int s = 0; s < numSamples; s++) {

		// Get the coordinate
		spos = m_smpset[s].pos;

		// Compute the distance 
		float dist = (float)npos.GetDist(spos);

		// Get the data from the variogram model
		vectord data  = m_pVariogram->GetModelData(dist);
		vvario[s] = data[0];
	} // end of for s

	// Set up 
	double ttw = 0;

	vvario[numSamples] = 1.0;
	for (int mx = 0; mx<m_uNumSamples; mx++)
	{
		weight[mx] = m_pPrePseudoInverse(mx, m_uNumSamples);
		for (int my=0; my<m_uNumSamples; my++)
			weight[mx] += m_pPrePseudoInverse(mx, my)*vvario[my];
		ttw+=weight[mx];
	}
	// Inversever matrix Inv * vvario
	//		LYCMatVecMultiplication(weight[i], m_pPrePseudoInverse[i], vvario[i]);
	
	if ( fabs(ttw-1) > 0.0001 )
	{
		for (int i=0; i<m_uNumSamples; i++)
			weight[i]/=ttw;
	}

	// Compute the trend 
	pred = GetTrend(pos);
	for (int s = 0; s < numSamples; s++){
		double residual = m_smpset[s].val;
		pred += (weight[s] * residual);
	}

	// Compute the variance
	// Set up 
	vvario[numSamples] = 0.0;

	var = 0.0;
	for(int s = 0; s < numSamples; s++ ){
		var += (float)( weight[s] * vvario[s]);
	}

	return ttw;
}


//****************************************************************************
//
// *  Get trend component -> 
//
// 1. c [in] target parameter
//============================================================================
double kriging::
GetTrend(double pos[]) const
//============================================================================
{
	return 0;
}

bool kriging::Estimate( int model, double power, double step )
{
	// Use the residual to compute the variogram and compute the proper sample
	std::vector<VariogramItem> vcloud = ComputeVariogramCloud();

	m_pVarioCloud->SetSample(vcloud);
	if (m_AutoSetDistStep)
		AutoSetDistSetp();
	else
		m_DistStep = step;

	// Set the sample for the variogram
	m_pVariogram->SetSample(ComputeExperimentalVariogram(vcloud, m_DistStep));


	// Get the sample from the variogram ?????? recheck
	double dist;
	vectord vario(1);
	m_pVariogram->GetSample(m_pVariogram->NumSamples() / 2, dist, vario);

	// Estimate the model of the variogram ?????
	vectord n(1, 0.f);
	vectord s = vario;
	vectord r(1, dist);
	vectord powerset;
	powerset.push_back(power);
	m_pVariogram->EstimateModel(model, n, s, r, powerset, m_pVariogram->MaxDistance() / 2.0);


	// Pre-compute the inverse matrix
	//if (m_DoPreCompute)
		PrecomputeKrigingSystem();

	return true;
}

bool kriging::Initial( int smpl, int numDataItems )
{
	return Allocate(smpl, numDataItems);
}

void kriging::AutoSetDistSetp()
{
	int n = 150;

	if (m_pVarioCloud->GetNumOfSmp()>n)
	{
		m_DistStep = m_pVarioCloud->GetDistance(n)/15;
	}
	else
		m_DistStep = m_pVarioCloud->MaxDistance()/15;
}

void kriging::GetDoubleMatrix( std::vector<double>& mat )
{
	int size = m_uNumSamples*m_uNumSamples;
	if (mat.size()!=size)
		mat.resize(size);
	for (int i=0; i < size; i++)
	{
		mat[i] = m_pPrePseudoInverse(int(size/m_uNumSamples), size%m_uNumSamples);
	}
}

void kriging::GetFloatMatrix( std::vector<float>& mat )
{
	int size = m_uNumSamples*m_uNumSamples;
	if (mat.size()!=size)
		mat.resize(size);
	for (int i=0; i < size; i++)
	{
		mat[i] = (float)m_pPrePseudoInverse(int(size/m_uNumSamples), size%m_uNumSamples);
	}
}

void kriging::GetFloatMatrix( float* mat )
{
	for (int i=0; i < m_uNumSamples+1; i++)
	{
		for (int j=0; j < m_uNumSamples+1; j++)
		{
			mat[i*(m_uNumSamples+1)+j] = (float)m_pPrePseudoInverse(i, j);
		}
	}
}
