#pragma once
#include <vtkXMLPolyDataWriter.h>
#include <vtkXMLPolyDataReader.h>
#include "SolidDefine.h"
#include <string>
#include <iomanip>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
VTK_SMART_POINTER(vtkXMLPolyDataWriter)
VTK_SMART_POINTER(vtkXMLPolyDataReader)
#include "Interpolation/InterpolationInfo.h"
#include "Interpolation/vtkNearestNeighborFilterCuda.h"

class PolyDataHandler
{
public:
	typedef std::vector<std::string> strVector;
	typedef std::vector<int> iVector;
	typedef std::vector<double> dVector;
	// read, write
	static void SaveFileToVtp(vtkPolyData* polydata, std::string path);
	static vtkPolyData_Sptr LoadFileFromVtp(std::string path);
	// native to polydata
	static vtkPolyData_Sptrs LoadFileFromNative(std::string path);
	static void InterpolationPolyData(vtkPolyData_Sptrs datas, const InterpolationInfo* info);
	static void SavePolyDatasToEvrA(vtkPolyData_Sptrs datas, std::wstring Path, std::wstring filename);
private:
	
	static int GetDataAmount(std::string path);
};
