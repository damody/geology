#include "FastMarching.h"
#include "LevelSet.h"
#pragma warning(push ,0)
//********************************************************************************
//
// * Operator on <
//================================================================================
bool operator < (const SLevelValue & lhs, const SLevelValue & rhs) 
//================================================================================
{
	return lhs.m_dValue < rhs.m_dValue; 
}

//********************************************************************************
//
// * Constructor to set up the basic information
//================================================================================
FastMarching::
FastMarching(int nx, int ny, int nz, double gs) : 
        m_dGridSize(gs),
        m_iNumX(nx + 2), 
        m_iNumY(ny + 2), 
        m_iNumZ(nz + 2),
        m_iDi(m_iNumY * m_iNumZ),
        m_dGridSizeInv(1. / m_dGridSize), 
        m_dNumZInv(1. / m_iNumZ), 
        m_dNumZNumYInv(1. / m_iDi)
//================================================================================
{
	// Create the heap
	m_VFMHeap.reserve(m_iNumX * m_iNumY * m_iNumZ);
	// Create the level values
	m_pLevels = new SLevelValue[m_iNumX * m_iNumY * m_iNumZ];
}

//********************************************************************************
//
// * Destructor
//================================================================================
FastMarching::
~FastMarching(void) 
//================================================================================
{
    delete[] m_pLevels;
}

//********************************************************************************
//
// * Set up the initial state and level set
//   Do the level set once when the value outside is outside and inside is inside
//   Clear up the data
//   Do the level set second time when the value outside is inside and inside is outside
//================================================================================
void FastMarching::
UpdateLevelSet(SJCScalarField3d *v) 
//================================================================================
{
	// Set up the scalar field
	m_pValues = v;

	// Set up the states 
	SetStates();

	// Do marching
	DoHalfStep();

	// Flip m_pValues to work on outside (positive) values...
	int index = 0;
	for (int i = 0; i < m_iNumX; i++) {
		for (int j = 0; j < m_iNumY; j++) {
			for (int k = 0; k < m_iNumZ; k++) {
				if ((i != 0) && (i < m_iNumX - 1) && 
				    (j != 0) && (j < m_iNumY - 1) && 
				    (k != 0) && (k < m_iNumZ - 1)) {
						m_pLevels[index].m_dValue	 = 
							-m_pLevels[index].m_dValue;
						m_pLevels[index].m_eState	 = 
							(m_pLevels[index].m_dValue < 0.) ? OUTSIDE : INSIDE;
						m_pLevels[index].m_iHeapPosition = -1;
				} // end if

				index++;
			} // end of for k
		} // end of for j
	} // end of for i

	DoHalfStep();
	for (int i = 1; i < m_iNumX - 1; i++)
		for (int j = 1; j < m_iNumY - 1; j++)
			for (int k = 1; k < m_iNumZ - 1; k++)
				m_pValues->Value(i - 1, j - 1, k - 1, 
						 m_pLevels[GetIndex(i, j, k)].m_dValue);

	// clean up a little... just in case.
	m_VInitialClosePoints.clear();
}

//********************************************************************************
//
// * Initialize the basic level set information for those outside, just outside
//   for those inside, just inside, the boundary is assume to be outside
//================================================================================
void FastMarching::
SetStates(void) 
//================================================================================
{
	SJCVector3d pos;
	int index = 0;

	// Go through each one
	for (int i = 0; i < m_iNumX; i++) {
		for (int j = 0; j < m_iNumY; j++) {
			for (int k = 0; k < m_iNumZ; k++) {
				// Get the level's index
				m_pLevels[index].m_iIndex = index;

				// Initialize the heap position
				m_pLevels[index].m_iHeapPosition = -1;

				// Boundary 
				if ((i == 0) || (i == m_iNumX - 1) || 
				    (j == 0) || (j == m_iNumY - 1) || 
				    (k == 0) || (k == m_iNumZ - 1)) {
					m_pLevels[index].m_eState = OUTSIDE;
				} else {
					// Set the coordinate's x, y, z to the position
					m_pValues->Coord(i - 1, j - 1, k - 1, pos);
					//++++++++++++ Grab the scalar value from the scalar field 
					m_pLevels[index].m_dValue = -(*m_pValues)(pos);
				    m_pLevels[index].m_eState = (m_pLevels[index].m_dValue < 0.) ? OUTSIDE : INSIDE;
				}

				index++;
			} // end of for k
		} // end of for j
	} // end of for i
 }

//********************************************************************************
//
// * Get the index of the element
//================================================================================
int FastMarching::
GetIndex(int i, int j, int k) 
//================================================================================
{
    return (i * m_iDi) + (j * m_iNumZ) + k;
}

//********************************************************************************
//
// * From index to get the i, j, k value 
//================================================================================
void FastMarching::
GetIJK(int index, int& i, int& j, int& k)
//================================================================================
{ 
    i      = int(index * m_dNumZNumYInv);
    index -= i * m_iDi;
    j      = int(index * m_dNumZInv);
    k      = int(index - j * m_iNumZ);
}

//********************************************************************************
//
// * Get the value at the index
//================================================================================
SLevelValue FastMarching::
GetLevelValue(int i, int j, int k) 
//================================================================================
{
	return m_pLevels[GetIndex(i, j, k)];
}

//********************************************************************************
//
// * Finding the boundary
//================================================================================
void FastMarching::
DoHalfStep(void) 
//================================================================================
{
    m_iHeapSize = 0;

    // Build the boundary of level set
    BuildLevelSetBoundary();
    InitHeap();
    March();
}

//********************************************************************************
//
// * First from z direction to find the one (inside) grid cell which is closest to
//   the boundary
//   second from y direction to find the one (inside) grid cell which is closest to
//   the boundary
//   Third from x direction to find the one (inside) grid cell which is closest to
//   the boundary
//   problem when the grid size is not constant how should we handle this
//================================================================================
void FastMarching::
BuildLevelSetBoundary() 
//================================================================================
{
	EGridState prev_state, cur_state;
	int prev_index, cur_index;

	for (int i = 1; i < m_iNumX - 1; i++){
		for (int j = 1; j < m_iNumY - 1; j++) {
			// why k = 2 because we already start up the initial 
			// state at k = 1
			prev_state = GetLevelValue(i, j, 1).m_eState;
			for (int k = 2; k < m_iNumZ - 1; k++){ 
				cur_index = GetIndex(i, j, k);
				cur_state = m_pLevels[cur_index].m_eState;

				// Check whether one state is out and another is in
				if (ONEISOUTSIDE(prev_state, cur_state)){
					prev_index = GetIndex(i, j, k - 1);

					// If current state is out and then set it to done
					if (ISINSIDE(cur_state)) {
						m_pLevels[cur_index].m_eState = DONE;
						AddClose(i, j, k + 1);

						//+++++++++++ Grid size will affect the result
						m_pLevels[cur_index].m_dValue = 
							min(m_pLevels[cur_index].m_dValue, 
							    abs(m_dGridSize + m_pLevels[prev_index].m_dValue));
					}
					else {
						m_pLevels[prev_index].m_eState = DONE;
						AddClose(i, j, k - 2);
						m_pLevels[prev_index].m_dValue = 
							min(m_pLevels[prev_index].m_dValue, 
							    abs(m_dGridSize + m_pLevels[cur_index].m_dValue));
					}
				} // end of is outside

				prev_state = cur_state;
			}// end of for k
		} // end of for j
	} // end of for i

	for (int i = 1; i < m_iNumX - 1; i++) {
		for (int k = 1; k < m_iNumZ - 1; k++) {
			prev_state = GetLevelValue(i, 1, k).m_eState;
			// why j = 2 because j=1 is used in previous state
			for (int j = 2; j < m_iNumY - 1; j++)	{ 
				cur_index = GetIndex(i, j, k);
				cur_state = m_pLevels[cur_index].m_eState;

				if (ONEISOUTSIDE(prev_state, cur_state)) {
					prev_index = GetIndex(i, j-1, k);

					if (ISINSIDE(cur_state)) {
						m_pLevels[cur_index].m_eState = DONE;
						AddClose(i, j + 1, k);
						// ++ Grid size will affect the value
						m_pLevels[cur_index].m_dValue = 
							min(m_pLevels[cur_index].m_dValue, 
							    abs(m_dGridSize + 
							        m_pLevels[prev_index].m_dValue));
					}
					else {
						m_pLevels[prev_index].m_eState = DONE;
						AddClose(i, j-2, k);
						m_pLevels[prev_index].m_dValue = min(m_pLevels[prev_index].m_dValue, abs(m_dGridSize + m_pLevels[cur_index].m_dValue));
					}
				} // end of if

				prev_state = cur_state;
			}//  end of for j
		} // end of for k
	} // end of for i

	for (int j = 1; j < m_iNumY - 1; j++) {
		for (int k = 1; k < m_iNumZ - 1; k++)	{
			prev_state = GetLevelValue(1, j, k).m_eState;

			// why i = 2, i=1 is used
			for (int i = 2; i < m_iNumX - 1; i++)	{ 
				cur_index = GetIndex(i, j, k);
				cur_state = m_pLevels[cur_index].m_eState;

				if (ONEISOUTSIDE(prev_state, cur_state)){
					prev_index = GetIndex(i - 1, j, k);

					if (ISINSIDE(cur_state)) {
						m_pLevels[cur_index].m_eState = DONE;
						AddClose(i + 1, j, k);
						// ++++ Grid size will effect the value
						m_pLevels[cur_index].m_dValue = 
							min(m_pLevels[cur_index].m_dValue, abs(m_dGridSize + m_pLevels[prev_index].m_dValue));
					}
					else {
						m_pLevels[prev_index].m_eState = DONE;
						AddClose(i-2, j, k);
						m_pLevels[prev_index].m_dValue = 
							min(m_pLevels[prev_index].m_dValue, abs(m_dGridSize + m_pLevels[cur_index].m_dValue));
					}
				}

				prev_state = cur_state;
			} // end of for i
		} // end of for j
	} // end of for k
}


//********************************************************************************
//
// * If the cube is inside and then we add the point into the Close point
//   the element that is close to the boundary which is not the one have value cross
//================================================================================
void FastMarching::
AddClose(int i, int j, int k) 
//================================================================================
{
	// Compute the index of i, j, k
	int index = GetIndex(i, j, k);

	//
	if (m_pLevels[index].m_eState == INSIDE)
		m_VInitialClosePoints.push_back(index);
}

//********************************************************************************
//
// * Initialize the heap
//================================================================================
void FastMarching::
InitHeap(void) 
//================================================================================
{
	// Go through all closest point and then set phi value if it is not initialized
	// Yeah
	static int x, y, z;
	for (int i = 0; i < m_VInitialClosePoints.size(); i++) {
		int index = m_VInitialClosePoints[i];

		if (m_pLevels[index].m_iHeapPosition == -1 && 
		    (m_pLevels[index].m_eState == INSIDE)) {
			GetIJK(index, x, y, z);
			FindPhi(index, x, y, z);
		}// end of if
	} // end of for
}

//********************************************************************************
//
// * 
//================================================================================
void FastMarching::
March(void) 
//================================================================================
{
	static int x, y, z;
	
	// Go through every element in the heap
	for (int index = PopHeap(); index != -1; index = PopHeap()) {
		//if (m_pLevels[index].m_dValue > FASTMARCH_LIMIT) return;

		// Find the neighbors' Phi according to the current update
		GetIJK(index, x, y, z);
		if (m_pLevels[GetIndex(x-1, y, z)].m_eState == INSIDE) 
			FindPhi(GetIndex(x-1,y,z), x-1, y, z);
		if (m_pLevels[GetIndex(x+1, y, z)].m_eState == INSIDE) 
			FindPhi(GetIndex(x+1,y,z), x+1, y, z);
		if (m_pLevels[GetIndex(x, y-1, z)].m_eState == INSIDE) 
			FindPhi(GetIndex(x,y-1,z), x, y-1, z);
		if (m_pLevels[GetIndex(x, y+1, z)].m_eState == INSIDE) 
			FindPhi(GetIndex(x,y+1,z), x, y+1, z);
		if (m_pLevels[GetIndex(x, y, z-1)].m_eState == INSIDE) 
			FindPhi(GetIndex(x,y,z-1), x, y, z-1);
		if (m_pLevels[GetIndex(x, y, z+1)].m_eState == INSIDE) 
			FindPhi(GetIndex(x,y,z+1), x, y, z+1);
	}
}

//********************************************************************************
//
// * Compute the phi of the current element
//================================================================================
void FastMarching::
FindPhi(int index, int x, int y, int z) 
//================================================================================
{
	static double	phiX, phiY, phiZ, b, quotient, phi;
	static int	a;
	static bool	flagX, flagY, flagZ;

	phiX = phiY = phiZ = 0.;
	a = 0;
	flagX = flagY = flagZ = 0;

	// Find The phiS
	CheckFront (phiX, a, flagX, GetIndex(x + 1,       y,       z));
	CheckBehind(phiX, a, flagX, GetIndex(x - 1,       y,       z));
	CheckFront (phiY, a, flagY, GetIndex(    x,   y + 1,       z));
	CheckBehind(phiY, a, flagY, GetIndex(    x,   y - 1,       z));
	CheckFront (phiZ, a, flagZ, GetIndex(    x,       y,   z + 1));
	CheckBehind(phiZ, a, flagZ, GetIndex(    x,       y,   z - 1));

	//Max Tests
	if (a == 3) {
		if      ((phiX >= phiY) && (phiX >= phiZ)) 
			CheckMax3(a, flagX, phiX, phiY, phiZ);
		else if ((phiY >= phiX) && (phiY >= phiZ)) 
			CheckMax3(a, flagY, phiY, phiX, phiZ);
		else		//?????							   
			CheckMax3(a, flagZ, phiZ, phiX, phiY);
	}
	if (a == 2) {
		if (!flagX) {
			if (phiY >= phiZ) 
				CheckMax2(a, phiY, phiZ);
			else			  
				CheckMax2(a, phiZ, phiY);
		}
		else if (!flagY){
			if (phiX >= phiZ) 
				CheckMax2(a, phiX, phiZ);
			else			  
				CheckMax2(a, phiZ, phiX);
		}
		else {
			if (phiX >= phiY) 
				CheckMax2(a, phiX, phiY);
			else			  
				CheckMax2(a, phiY, phiX);
		}
	}

	b = phiX + phiY + phiZ;
	quotient = square(b) - (double) a * (square(phiX) + 
		   square(phiY) + square(phiZ) - 
		   square(m_dGridSize));
	if (quotient < 0.) 
		std::cout << "0 ";
	else {
		phi = b + sqrt(quotient);
		phi /= (double) a;
		m_pLevels[index].m_dValue = phi;
		if (m_pLevels[index].m_iHeapPosition == -1) 
			AddToHeap(index);
		else
			UpdateHeap(index); 
	}
}

//********************************************************************************
//
// * Check whether the front element is done
//================================================================================
void FastMarching::
CheckFront(double& phi, int& a, bool& flag, int index) 
//================================================================================
{
	if (m_pLevels[index].m_eState == DONE) {
		phi  = m_pLevels[index].m_dValue;
		flag = 1;
		a++;
	}
}

//********************************************************************************
//
// * Check whether the back element is done
//================================================================================
void FastMarching::
CheckBehind(double& phi, int& a, bool& flag, int index)
//================================================================================
{
	if (m_pLevels[index].m_eState == DONE) {
		if (!flag) { 
			phi = m_pLevels[index].m_dValue; 
			a++; 
	        }
		else 
			phi = min(m_pLevels[index].m_dValue, phi);
		flag = 1;
	}
}

//********************************************************************************
//
// * 
//================================================================================
void FastMarching::
CheckMax2(int& a, double& phi1, const double &phi2) 
//================================================================================
{
	if (square((phi1 - phi2) * m_dGridSizeInv) > 1.)  { 
		phi1 = 0; 
		a    = 1; 
	}
}

//********************************************************************************
//
// * 
//================================================================================
void FastMarching::
CheckMax3(int& a, bool& flag, double& phi1, 
	  const double &phi2, const double &phi3) 
//================================================================================
{
	if ((square((phi1 - phi2) * m_dGridSizeInv) + 
	     square((phi1 - phi3) * m_dGridSizeInv)) > 1.) {   
		phi1 = 0; 
		a    = 2; 
		flag = 0;  
	}
}


//********************************************************************************
//
// * 
//================================================================================
void FastMarching::
AddToHeap(int index)
//================================================================================
{
	m_VFMHeap[m_iHeapSize] = index;
	m_pLevels[index].m_iHeapPosition = m_iHeapSize;
	int j, i = m_iHeapSize;
	for(i; i > 0; i = j) {
		j = (i-1)/2;
		if (m_pLevels[ (m_VFMHeap[i]) ].m_dValue < m_pLevels[ (m_VFMHeap[j]) ].m_dValue ){
			m_VFMHeap[i] = m_VFMHeap[j];
			m_pLevels[ (m_VFMHeap[j]) ].m_iHeapPosition = i;
			m_VFMHeap[j] = index;
			m_pLevels[index].m_iHeapPosition = j;
		}
		else
			break;
	}
	m_iHeapSize++;
}

//********************************************************************************
//
// * 
//================================================================================
void FastMarching::
UpdateHeap(int index) 
//================================================================================
{
	int j, i = m_pLevels[index].m_iHeapPosition;
	for(i; i > 0; i = j) {
		j = (i-1)/2;
		if (m_pLevels[ (m_VFMHeap[i]) ].m_dValue < m_pLevels[ (m_VFMHeap[j]) ].m_dValue ) {
			m_VFMHeap[i] = m_VFMHeap[j];
			m_pLevels[ (m_VFMHeap[j]) ].m_iHeapPosition = i;
			m_VFMHeap[j] = index;
			m_pLevels[index].m_iHeapPosition = j;
		}
		else
			break;
	}
}

//********************************************************************************
//
// * Pop an element from the heap
//================================================================================
int FastMarching::
PopHeap(void) 
//================================================================================
{
	if(m_iHeapSize == 0)
		return -1;
	int j, index = m_VFMHeap[0];
	m_pLevels[index].m_eState = DONE;

	// Reduce the size
	m_iHeapSize--;

	// Move the last element to the head of the heap
	m_VFMHeap[0] = m_VFMHeap[m_iHeapSize];
	m_pLevels[m_VFMHeap[m_iHeapSize]].m_iHeapPosition = 0;

	// Process down to update the heap
	for(int i = 0; i < (m_iHeapSize-1); i = j) {
		int lc = 2 * i + 1;
		int rc = 2 * i + 2;
		double current = m_pLevels[ (m_VFMHeap[i]) ].m_dValue;
		double lv, rv;
		if(lc < m_iHeapSize) {
			lv = m_pLevels[ (m_VFMHeap[lc]) ].m_dValue;
			if(rc < m_iHeapSize) {
				rv = m_pLevels[ (m_VFMHeap[rc]) ].m_dValue;
				if(lv > rv) {
					lc = rc;
					lv = rv;
				}
			}
			if(current > lv) {
				m_VFMHeap[i] = m_VFMHeap[lc];
				m_pLevels[ m_VFMHeap[i] ].m_iHeapPosition = i;
				m_VFMHeap[lc] = m_VFMHeap[m_iHeapSize];
				m_pLevels[ m_VFMHeap[m_iHeapSize] ].m_iHeapPosition = lc;
				j = lc;
			}
			else
				break;
		}
		else
			break;
	}
	return index;
}
#pragma warning(pop)
