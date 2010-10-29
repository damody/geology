/************************************************************************
     Main File:   


     File:        FastMarching.h

     Author:    
		  Yu-Chi Lai, yu-chi@mail.ntust.edu.tw
     Comment:     

     Compiler:    Microsoft Visual Studio

     Platform:    Window
*************************************************************************/

#pragma once 

#include <algorithm>
#include <queue>
#include <vector>

// Pre-declared for the ScalarField3d
class SJCScalarField3d;

typedef enum { 
	OUTSIDE = -1, 
	INSIDE = 0, 
	DONE = 1 
} EGridState;

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Need to be change because 
// Moves from OUTSIDE to INSIDE or DONE states (or vice versa)...
#define ONEISOUTSIDE(v1, v2) v1 ^ v2 
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Need to be change because 
// Whether the object is inside or outside
#define ISINSIDE(val) val >= 0
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


//***********************************************************************
//
// * Keep the value in level set
//
//***********************************************************************
struct SLevelValue {
    public:
        EGridState	m_eState;
        double		m_dValue;
        int		m_iIndex;
        int		m_iHeapPosition;

	// Constructor
	SLevelValue(void) {}
};

//***********************************************************************
//
// * Reconstruct the triangle
//
//***********************************************************************
class FastMarching
{
    public:
	// Constructor 
	FastMarching(int numX, int numY, int numZ, double gs);

	// Destructor
	~FastMarching(void);

	// Update the level set value
	void UpdateLevelSet(SJCScalarField3d *v);

     private:
	// Initialize the basic level set information for those outside, just outside
	// for those inside, just inside, the boundary is assume to be outside
	void		SetStates();

	// Build the level set boundary, Initialize the heap and then march
	void		DoHalfStep();
	//
	void		BuildLevelSetBoundary();
	//
	void		March();
	//
	void		InitHeap();
    
	// Get the level value at i, j, k
	SLevelValue	GetLevelValue(int i, int j, int k);

	// Get the index of the element
	inline int	GetIndex(int i, int j, int k);

	// From I, J, K to compute index
	inline void	GetIJK(int index, int& i, int& j, int& k);

	inline void	AddClose(int i, int j, int k);
	void		AddToHeap(int index);
	void		UpdateHeap(int index);
	int		PopHeap();

	void		FindPhi(int index, int x, int y, int z);

	inline void	CheckFront(double& phi, int& a, bool& flag, int index);
	inline void	CheckBehind(double& phi, int& a, bool& flag, int index);
	inline void	CheckMax2(int& a, double& phi1, const double &phi2);
	inline void	CheckMax3(int& a, bool& flag, double& phi1, const double &phi2, const double &phi3);

	inline double	square(double a) { return a*a; }
    private:

	std::vector<int> m_VInitialClosePoints; //
	std::vector<int> m_VFMHeap;		   //

	int    m_iHeapSize;		   //
	int    m_iNumX;                    // Number of values in X dimension
	int    m_iNumY;                    // Number of values in Y dimension
	int    m_iNumZ;                    // Number of values in Z dimension
	int    m_iDi;                      // Y * Z
	double m_dGridSize;                // The grid size which will be a problem in my application
	double m_dGridSizeInv;		   // Inverse of grid size
	double m_dNumZInv;		   // Inverse of number in Z
	double m_dNumZNumYInv;             // Inverse of number in Y
	// priority_queue<SLevelValue, std::vector<SLevelValue>, greater<SLevelValue> > closePoints;

	SJCScalarField3d	*m_pValues;
	SLevelValue		*m_pLevels;

};
