
#include "HandleEvr.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cassert>
#include <algorithm>
#include "VarStr.h"

// directoryPath?槌ua頝臬? lua瑼??vr頝臬?
int HandleEvr::InitLoad ()
{
	DataAmount = m_cell.getLua_UsePath<double> ("DataAmount");
	for (int i = 0; i < DataAmount - 3; i++)
	{
		m_Datamin.push_back(m_cell.getLua_UsePath<double> (("Datamin" + ConvStr::GetStr(i + 1)).c_str()));
		m_Datamax.push_back(m_cell.getLua_UsePath<double> (("Datamax" + ConvStr::GetStr(i + 1)).c_str()));
	}
	Xmin = m_cell.getLua_UsePath<double> ("Xmin");
	Xmax = m_cell.getLua_UsePath<double> ("Xmax");
	deltaX = m_cell.getLua_UsePath<double> ("deltaX");
	Xspan = m_cell.getLua_UsePath<int> ("Xspan");
	Ymin = m_cell.getLua_UsePath<double> ("Ymin");
	Ymax = m_cell.getLua_UsePath<double> ("Ymax");
	deltaY = m_cell.getLua_UsePath<double> ("deltaY");
	Yspan = m_cell.getLua_UsePath<int> ("Yspan");
	Zmin = m_cell.getLua_UsePath<double> ("Zmin");
	Zmax = m_cell.getLua_UsePath<double> ("Zmax");
	deltaZ = m_cell.getLua_UsePath<double> ("deltaZ");
	Zspan = m_cell.getLua_UsePath<int> ("Zspan");
	m_format_count = m_cell.getLua_UsePath<int> ("format_count");
	m_format_name.push_back("x");
	m_format_name.push_back("y");
	m_format_name.push_back("z");
	for (int i = 1; i <= DataAmount - 3; i++)
	{
		m_format_name.push_back(m_cell.getLua_UsePath < const char * > ("format_name/%d", i));
	}
	for (int i = 0; i < DataAmount; i++)
	{
		m_moveTable.push_back(sizeof(double) * i);
	}
	m_format_count += 3;		// add x,y,z col
	m_dataSize = DataAmount * sizeof(double);
	m_total = m_cell.getLua_UsePath<int> ("total");
	m_totalSize = m_total * m_dataSize;
	m_dataPath = m_cell.getLua_UsePath < const char * > ("data");
	m_dataWPath = ConvStr::GetWstr(m_dataPath.c_str());
	if (m_wdir.size() != 0)
	m_dataWPath = m_wdir + L"\\" + m_dataWPath;
	using namespace std;
	string		data_format = m_cell.getLua_UsePath < const char * > ("data_format");
	transform(data_format.begin(), data_format.end(), data_format.begin(), (int(*) (int)) tolower);
	if (data_format == "binary")
	{
		ifstream	fIn;
		unsigned long	ulSize;
		fIn.open(m_dataWPath.c_str(), ios::binary);
		if (fIn == NULL)
			return 0;

		// Get file size
		fIn.seekg(0, ios_base::end);
		ulSize = fIn.tellg();
		fIn.seekg(0, ios_base::beg);
		if (m_totalSize != ulSize)
			MessageBox(NULL, L"m_totalSize != ulSize", L"???", MB_OK | MB_ICONASTERISK);

		// Allocate some space
		// Check and clear pDat, just in case
		m_pData.resize(ulSize);

		// Read the file into memory
		fIn.read((char *) &m_pData[0], ulSize);
		fIn.close();
	}
	else
	{
		ifstream	fIn;
		fIn.open(m_dataWPath.c_str());
		if (fIn == NULL)
			return 0;
		m_pData.resize(m_totalSize);
		fIn.ignore(1000, '\n'); // ignore first line
		double	read;
		int	move = 0;
		double	*psave = (double *) &m_pData[0];
		for (unsigned int i = 0; i < m_total; i++)
		{
			for (int j = 0; j < m_format_count; j++)
			{
				fIn >> read;
				*(psave + move++) = read;
				fIn.ignore(10, ',');
			}
		}
		fIn.close();
	}
	vector<double *>	dpvec;
	bool			isGrided = (Xspan + 1) * (Yspan + 1) * (Zspan + 1) == (int) m_total;
	if (isGrided)
	{
		for (int i = 0; i < m_format_count; i++)
		{
			SJCScalarField3d	*pSF3d = new SJCScalarField3d
				(
					Xspan +
					1, Yspan +
					1, Zspan +
					1, deltaX, deltaY, deltaZ, BOUNDARY_WRAP, BOUNDARY_WRAP, BOUNDARY_WRAP,
						(double *) (&m_pData[0]) + i, m_format_count
				);
			dpvec.push_back(pSF3d->begin());
			m_SJCSF3dMap.push_back(std::make_pair(m_format_name[i], pSF3d));
		}
		for (int i = 0; i < 3; ++i)
		{
			DependenceSort(dpvec[i], m_total, dpvec);
		}
		for (int i = 0; i < 10; i++)
		{
			for (int j = 0; j < 4; j++)
				printf("%f\t", dpvec[j][i]);
			printf("\n");
		}
	}
	else
	{
		for (int i = 0; i < m_format_count; i++)
		{
			vtkPoints_Sptr		points = vtkSmartNew;
			vtkDoubleArray_Sptr	point_array = vtkSmartNew;
			vtkPolyData_Sptr	polydata = vtkSmartNew;
			for (int offset = 0; offset < (int) m_total; offset++)
			{
				point_array->InsertTuple1(offset, *((double *) (&m_pData[0]) + i + offset * m_format_count));
				points->InsertNextPoint
					(
						*((double *) (&m_pData[0]) + 0 + offset * m_format_count), *
							((double *) (&m_pData[0]) + 1 + offset * m_format_count), *
								((double *) (&m_pData[0]) + 2 + offset * m_format_count)
					);
			}
			polydata->SetPoints(points);
			polydata->GetPointData()->SetScalars(point_array);
			assert(polydata->GetNumberOfPoints() != 0);
			//m_SJCSF3dMap.push_back(std::make_pair(m_format_name[i], polydata));
		}
	}
	m_isload = true;
	return 1;
}
HandleEvr::~HandleEvr()
{
	ExitFile();
}

void HandleEvr::ExitFile()
{
	for (SJCSF3dMap::iterator it = m_SJCSF3dMap.begin(); it != m_SJCSF3dMap.end(); it++)
	{
		if (it->second.m_type == PointDataSource::SJCSCALAR_PTR)
			delete it->second.m_sjcf3d;
	}
}

int HandleEvr::Save_Evr(std::wstring Path, std::wstring filename)
{
	m_CreateLua.clear();

	std::string		strFormat = "{";
	SJCSF3dMap::iterator	it = m_SJCSF3dMap.begin();
	it++;
	it++;
	it++;
	for (; it != m_SJCSF3dMap.end();)
	{
		strFormat += "\"";
		strFormat += it->first;
		strFormat += "\"";
		it++;
		if (it != m_SJCSF3dMap.end())
			strFormat += ",";
		else
			break;
	}

	strFormat += "}";
	m_CreateLua.AddRawString("format_name", "{\"resistance\"}");
	m_CreateLua.AddInt("format_count", 1);
	m_CreateLua.AddInt("total", m_total);
	for (int j = 0; j < DataAmount - 3; j++)
	{
		m_CreateLua.AddDouble("Datamin" + ConvStr::GetStr(j), m_Datamin[j]);
		m_CreateLua.AddDouble("Datamax" + ConvStr::GetStr(j), m_Datamax[j]);
	}

	m_CreateLua.AddDouble("Xmin", Xmin);
	m_CreateLua.AddDouble("Xmax", Xmax);
	m_CreateLua.AddDouble("deltaX", deltaX);
	m_CreateLua.AddDouble("Xspan", Xspan);
	m_CreateLua.AddDouble("Ymin", Ymin);
	m_CreateLua.AddDouble("Ymax", Ymax);
	m_CreateLua.AddDouble("deltaY", deltaY);
	m_CreateLua.AddDouble("Yspan", Yspan);
	m_CreateLua.AddDouble("Zmin", Zmin);
	m_CreateLua.AddDouble("Zmax", Zmax);
	m_CreateLua.AddDouble("deltaZ", deltaZ);
	m_CreateLua.AddDouble("Zspan", Zspan);
	m_CreateLua.AddString("data_format", "binary");
	m_CreateLua.AddString("data", ConvStr::GetStr(filename.c_str()) + std::string(".evr"));
	m_CreateLua.SaveLua(Path + L".lua");

	using namespace std;
	ofstream	fOut;
	fOut.open((Path + L".evr").c_str(), ios::binary);
	if (fOut == NULL)
		return 0;
	if (m_pData.empty())
	{
		fOut.close();
		return 0;
	}

	// Write the file into file
	fOut.write((char *) &m_pData[0], m_totalSize);
	fOut.close();
	return 1;
}

int HandleEvr::Save_EvrA(std::wstring Path, std::wstring filename)
{
	m_CreateLua.clear();

	std::string		strFormat = "{";
	SJCSF3dMap::iterator	it = m_SJCSF3dMap.begin();
	it++;
	it++;
	it++;
	for (; it != m_SJCSF3dMap.end();)
	{
		strFormat += "\"";
		strFormat += it->first;
		strFormat += "\"";
		it++;
		if (it != m_SJCSF3dMap.end())
			strFormat += ",";
		else
			break;
	}

	strFormat += "}";
	m_CreateLua.AddRawString("format_name", "{\"resistance\"}");
	m_CreateLua.AddInt("format_count", 1);
	m_CreateLua.AddInt("total", m_total);
	for (int j = 0; j < DataAmount - 3; j++)
	{
		m_CreateLua.AddDouble("Datamin" + ConvStr::GetStr(j), m_Datamin[j]);
		m_CreateLua.AddDouble("Datamax" + ConvStr::GetStr(j), m_Datamax[j]);
	}

	m_CreateLua.AddDouble("Xmin", Xmin);
	m_CreateLua.AddDouble("Xmax", Xmax);
	m_CreateLua.AddDouble("deltaX", deltaX);
	m_CreateLua.AddDouble("Xspan", Xspan);
	m_CreateLua.AddDouble("Ymin", Ymin);
	m_CreateLua.AddDouble("Ymax", Ymax);
	m_CreateLua.AddDouble("deltaY", deltaY);
	m_CreateLua.AddDouble("Yspan", Yspan);
	m_CreateLua.AddDouble("Zmin", Zmin);
	m_CreateLua.AddDouble("Zmax", Zmax);
	m_CreateLua.AddDouble("deltaZ", deltaZ);
	m_CreateLua.AddDouble("Zspan", Zspan);
	m_CreateLua.AddString("data_format", "ascii");
	m_CreateLua.AddString("data", ConvStr::GetStr(filename.c_str()) + std::string(".evr"));
	m_CreateLua.SaveLua(Path + L".lua");

	using namespace std;
	ofstream	fOut;
	fOut.open((Path + L".evr").c_str());
	if (fOut == NULL)
		return 0;
	if (m_pData.empty())
	{
		fOut.close();
		return 0;
	}

	// Write the file into file
	for (int i = 0; i < m_format_count; i++)
	{
		fOut << setw(16) << m_format_name[i];
	}

	fOut << std::endl;
	fOut.setf(ios_base::scientific);
	for (uint i = 0; i < m_total; i++)
	{
		for (it = m_SJCSF3dMap.begin(); it != m_SJCSF3dMap.end(); it++)
		{
			fOut << setw(15) << *(((SJCScalarField3d *) (it->second))->begin() + i) << ",";
		}

		fOut << std::endl;
	}

	fOut.close();
	return 1;
}

int HandleEvr::getSize(const std::string &str)
{
	if (str == "double")
		return 8;
	if (str == "int")
		return 4;
	if (str == "short")
		return 2;
	if (str == "char")
		return 1;
	if (str == "long")
		return 4;
	if (str == "float")
		return 4;
	if (str == "long long")
		return 8;
	assert(0 && "Error Type");
	return 0;
}

double *HandleEvr::Get2Ddata()
{
	p2d = new double[m_total];

	double	*p = p2d;
	for (unsigned int i = 0; i < m_total; i++)
	{
		*p = GetData(i, 3);
		p++;
	}

	return p2d;
}

bool HandleEvr::IsLoad()
{
	return m_isload;
}

template<class T, class DT> void DependenceSort(T *beg, const uint total, std::vector<DT *> &depVector)
{
	T	**ppAry = new T *[total];
	for (uint i = 0; i < total; i++)
	{
		ppAry[i] = beg + i;
	}

	std::stable_sort(ppAry, ppAry + total, ptr_value_cmp<T> ());

	uint	*iAry = new uint[total];
	for (uint i = 0; i < total; i++)
	{
		iAry[i] = ppAry[i] - beg;
	}

	DT	*Ary = new DT[total];
	for (uint num = 0; num < depVector.size(); num++)
	{
		std::copy(depVector[num], depVector[num] + total, Ary);

		DT	*dst = depVector[num];
		for (uint i = 0; i < total; i++)
		{
			dst[i] = Ary[iAry[i]];
		}
	}

	delete[] Ary;
	delete[] iAry;
	delete[] ppAry;
}
