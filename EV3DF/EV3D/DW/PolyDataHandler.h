// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
// In academic purposes only(2012/1/12)
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
VTKSMART_PTR(vtkXMLPolyDataWriter)
VTKSMART_PTR(vtkXMLPolyDataReader)
#include "Interpolation/InterpolationInfo.h"

// for load polydata
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
	static vtkPolyData_Sptrs LoadFileFromNative(std::wstring path);
	static void InterpolationPolyData(vtkPolyData_Sptrs &datas, const InterpolationInfo* info);
	static void SavePolyDatasToEvrA(vtkPolyData_Sptrs datas, std::wstring Path, std::wstring filename);
private:
	static int GetDataAmount(std::wstring path);
};
// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
// In academic purposes only(2012/1/12)
