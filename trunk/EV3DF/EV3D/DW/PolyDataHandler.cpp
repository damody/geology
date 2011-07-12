#include "PolyDataHandler.h"
#include "SolidDefine.h"
#include <cassert>
#include <istream>
#include <fstream>
#include <iostream>
#include <istream>
#include <sstream>
#include "ConvStr.h"
#include "..\Lua\CreateLua.h"
#include "Interpolation\vtkBounds.h"
#include "Interpolation/vtkNearestNeighborFilter.h"
VTKSMART_PTR(vtkNearestNeighborFilter)

void PolyDataHandler::SaveFileToVtp( vtkPolyData* polydata, std::string path )
{
	vtkXMLPolyDataWriter_Sptr writer = vtkSmartNew;
	writer->SetFileName(path.c_str());
	writer->SetInput(polydata);
	writer->SetCompressorTypeToZLib();
	writer->Write();
}

vtkPolyData_Sptr PolyDataHandler::LoadFileFromVtp( std::string path )
{
	vtkXMLPolyDataReader_Sptr reader = vtkSmartNew;
	reader->SetFileName(path.c_str());
	reader->Update();
	vtkPolyData_Sptr output = vtkSmartNew;
	output->ShallowCopy(reader->GetOutput());
	return output;
}

vtkPolyData_Sptrs PolyDataHandler::LoadFileFromNative( std::wstring path )
{
	const int AMOUNT = GetDataAmount(path);
	vtkPolyData_Sptrs output;
	vtkPoints_Sptr outpoints = vtkSmartNew;
	vtkDoubleArray_Sptrs outScalars;
	
	std::ifstream loader;
	std::vector<double> dataVector;
	double tdata;
	loader.open(path.c_str());
	// read data from file
	for (;!loader.eof();)
	{
		if (loader.peek()==',')
			loader.get();
		loader >> tdata;
		dataVector.push_back(tdata);
	}
	const int DATA_TOTAL = dataVector.size()/AMOUNT;
	for (int i=3;i<AMOUNT;i++) // else x,y,z 3 field
		outScalars.push_back(vtkSmartNew);
	// get data form vector
	for (int i=0;i<DATA_TOTAL;i++)
	{
		outpoints->InsertNextPoint(&dataVector[i * AMOUNT]);
		for (int field_index=0;field_index < AMOUNT-3;field_index++)
		{
			assert(i * AMOUNT+field_index < DATA_TOTAL*AMOUNT && " index > DATA_TOTAL*AMOUNT, error");
			outScalars[field_index]->InsertNextTuple1(dataVector[i * AMOUNT+field_index+3]);
		}
	}
	// format data to polydata vector
	for (int field_index=0;field_index < AMOUNT-3;field_index++)
	{
		vtkPolyData_Sptr out = vtkSmartNew;
		out->SetPoints(outpoints);
		out->GetPointData()->SetScalars(outScalars[field_index]);
		output.push_back(out);
	}
	return output;
}

int PolyDataHandler::GetDataAmount( std::wstring path )
{
	std::ifstream loaderx;
	loaderx.open(path.c_str());
	std::string linedata;
	loaderx.ignore(INT_MAX, loaderx.widen('\n'));
	char ch[512];
	int i=0;
	loaderx.getline(ch, 512);
	loaderx.close();
	std::istringstream split;
	split.str(ch);
	while(!split.eof())
	{
		split >> linedata;
		i++;
	}
	// i = item's amount 
	return i;
}

void PolyDataHandler::SavePolyDatasToEvrA( vtkPolyData_Sptrs datas, std::wstring Path, std::wstring filename )
{
	double		Xdelta, Ydelta, Zdelta;
	int		Xspan, Yspan, Zspan;
	int		m_format_count;
	strVector	m_format_name;
	std::string	m_dataPath;
	std::wstring	m_dataWPath;
	CreateLua	m_CreateLua;
	int	m_dataSize, m_total;
	unsigned long	m_totalSize;
	// get Bounds
	vtkBounds bounds;
	datas[0]->GetBounds(bounds);
	// init lua
	m_format_name.clear();
	m_format_name.push_back("x");
	m_format_name.push_back("y");
	m_format_name.push_back("z");
	m_format_count = datas.size();	
	std::string formatstring;
	for (int j = 0;j < m_format_count;j++)
	{
		formatstring += "\"parameter" + ConvStr::GetStr(j+1) + "\",";
		m_format_name.push_back("parameter" + ConvStr::GetStr(j+1));
	}
	m_CreateLua.AddRawString("format_name", "{" + formatstring + "}");
	m_CreateLua.AddInt("format_count", m_format_count);
	// 算出每一組資料的大小
	m_dataSize = sizeof(double) * (m_format_count+3);
	// 讀出檔案

	m_total = datas[0]->GetNumberOfPoints();
	m_totalSize = m_total * m_dataSize;
	m_CreateLua.AddInt("total",m_total);

	// 讀入資料的最大最小值
	for (int i = 0;i < m_format_count;i++)
	{
		vtkDoubleArray* inScalars = (vtkDoubleArray*)(datas[i]->GetPointData()->GetScalars());
		m_CreateLua.AddDouble("Datamin" + ConvStr::GetStr(i), inScalars->GetDataTypeValueMin());
		m_CreateLua.AddDouble("Datamax" + ConvStr::GetStr(i), inScalars->GetDataTypeValueMax());
	}
	vtkPolyData* polydata = datas[0];
	double p1[3], p2[3];
	Xdelta = Ydelta = Zdelta = 0;
	for (int i=0;Xdelta==0||Ydelta==0||Zdelta==0;i++)
	{
		polydata->GetPoint(i,p1);
		polydata->GetPoint(i+1,p2);
		if (p1[0]-p2[0] != 0 && Xdelta == 0)
			Xdelta = abs(p1[0]-p2[0]);
		if (p1[1]-p2[1] != 0 && Ydelta == 0)
			Ydelta = abs(p1[1]-p2[1]);
		if (p1[2]-p2[2] != 0 && Zdelta == 0)
			Zdelta = abs(p1[2]-p2[2]);
	}
	Xspan = (bounds.xmax-bounds.xmin)/Xdelta;
	Yspan = (bounds.ymax-bounds.ymin)/Ydelta;
	Zspan = (bounds.zmax-bounds.zmin)/Zdelta;
	m_CreateLua.AddDouble("DataAmount", m_format_count+3);
	m_CreateLua.AddDouble("Xmin", bounds.xmin);
	m_CreateLua.AddDouble("Xmax", bounds.xmax);
	m_CreateLua.AddDouble("deltaX", Xdelta);
	m_CreateLua.AddDouble("Xspan", Xspan);
	m_CreateLua.AddDouble("Ymin", bounds.ymin);
	m_CreateLua.AddDouble("Ymax", bounds.ymax);
	m_CreateLua.AddDouble("deltaY", Ydelta);
	m_CreateLua.AddDouble("Yspan", Yspan);
	m_CreateLua.AddDouble("Zmin", bounds.zmin);
	m_CreateLua.AddDouble("Zmax", bounds.zmax);
	m_CreateLua.AddDouble("deltaZ", Zdelta);
	m_CreateLua.AddDouble("Zspan", Zspan);

	m_CreateLua.AddString("data_format","ascii");
	m_CreateLua.AddString("data", ConvStr::GetStr(filename.c_str())+std::string(".evr"));
	m_CreateLua.SaveLua(Path + L".lua");
	using namespace std;
	ofstream fOut;
	fOut.open((Path+L".evr").c_str());
	if(fOut==NULL)
		return ;
	// Write the file into file
	for (int i=0;i<m_format_count+3;i++)
	{
		fOut << setw(16) << m_format_name[i];
	}
	fOut << std::endl;
	fOut.setf(ios_base::scientific);
	for (int i=0;i<m_total;i++)
	{
		double p[3];
		polydata->GetPoint(i,p);
		for (int j=0;j<3;j++)
		{
			fOut << setw(15) << p[j] << ",";
		}
		for (int j=0;j<m_format_count;j++)
		{
			vtkDoubleArray* inScalars = (vtkDoubleArray*)(datas[j]->GetPointData()->GetScalars());
			fOut << setw(15) << inScalars->GetTuple1(i) << ",";
		}
		fOut << std::endl;
	}
	fOut.close();
}

void PolyDataHandler::InterpolationPolyData( vtkPolyData_Sptrs &datas, const InterpolationInfo* info )
{
	for (int i=0;i<datas.size();i++)
	{
		vtkNearestNeighborFilter_Sptr NearestNeighborCuda = vtkSmartNew;
		double bounds[6] = {info->min[0], info->max[0], 
				info->min[1], info->max[1], 
				info->min[2], info->max[2]};
		NearestNeighborCuda->SetBounds(bounds);
		NearestNeighborCuda->SetInterval(info->interval[0], info->interval[1], info->interval[2]);
		NearestNeighborCuda->SetInput(datas[i]);
		NearestNeighborCuda->Update();
		datas[i] = NearestNeighborCuda->GetOutput();
		printf("GetNumberOfPoints: %d\n", datas[i]->GetNumberOfPoints());
	}
}
