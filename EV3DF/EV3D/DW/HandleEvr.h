#pragma once
#pragma warning (disable : 4996)
#pragma warning(disable:4127)
#include <vector>
#include <cstring>
#include <string>

#include "../lua/LuaCell.h"
#include "../Lua/CreateLua.h"
#include "ConvStr.h"
#include "Windows.h"
#include "SJCScalarField3.h"
#include "SJCVector3.h"
#include "SolidDefine.h"
template <class T, class DT>
void DependenceSort( T* beg, const uint total, std::vector<DT*>& depVector );

template <class T>
struct ptr_value_cmp
	: public std::binary_function<T, T, bool>
{	// functor for operator<
	bool operator()(T* _Left, T* _Right) const
	{	// apply operator< to operands
		return (*_Left) < (*_Right);
	}
};
struct PointDataSource
{
	enum {
		SJCSCALAR_PTR,
		VTKPOLYDATA
	};
	int m_type;
	SJCScalarField3d* m_sjcf3d;
	vtkPolyData_Sptr  m_polydata;
	PointDataSource(SJCScalarField3d* sjc):m_sjcf3d(sjc), m_type(SJCSCALAR_PTR){}
	PointDataSource(vtkPolyData_Sptr poly):m_polydata(poly), m_type(VTKPOLYDATA){}
	operator SJCScalarField3d*()
	{
		if (m_type == SJCSCALAR_PTR)
			return m_sjcf3d;
		return NULL;
	}
	operator vtkPolyData_Sptr()
	{
		if (m_type == VTKPOLYDATA)
			return m_polydata;
		return vtkPolyData_Sptr();
	}
};
class HandleEvr
{
public:
	typedef std::vector<std::string> strVector;
	typedef std::vector<int> iVector;
	typedef std::vector<double> dVector;

	HandleEvr(const char* file):m_pData(NULL),m_isload(false)
	{
		m_cell.InputLuaFile(file);
	}
	~HandleEvr();
	int InitLoad(const std::wstring& directoryPath);
	// 2d test, it will be delete on next version
	double* Get2Ddata();
	double* p2d;
	// test end
	void* GetData()
	{
		return (void*)&m_pData[0];
	}
	double GetData(int index, int column)
	{
		return *(double*)(&m_pData[index*m_dataSize+m_moveTable[column]]);
	}
	int GetTotal()
	{
		return m_total;
	}
	int GetDataSize()
	{
		return m_dataSize;
	}
	int Save_Evr(std::wstring Path, std::wstring filename);
	int Save_EvrA(std::wstring Path, std::wstring filename);
	bool IsLoad();
	void ExitFile();
	typedef std::vector< std::pair<std::string, PointDataSource> > SJCSF3dMap;
	SJCSF3dMap	m_SJCSF3dMap;
	double		Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, deltaX, deltaY, deltaZ, DataAmount;
	dVector		m_Datamax, m_Datamin;
	int		Xspan, Yspan, Zspan;
	int		m_format_count;
	strVector	m_format_name;
	iVector		m_moveTable;
	std::string	m_dataPath;
	std::wstring	m_dataWPath;
private:
	bool		m_isload;
	unsigned int	m_dataSize, m_total;
	unsigned long	m_totalSize;
	std::vector<unsigned char>	m_pData;
	LuaCell		m_cell;
	CreateLua	m_CreateLua;
	int getSize(const std::string& str);
};
