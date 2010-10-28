#include "ConvertToEvr.h"
#include <memory>
#include <wx\msgdlg.h>
#include <limits>
#include <sstream>
#include <iostream>
#include <istream>
#include <iosfwd>

ConvertToEvr::ConvertToEvr() :isLoad(false)
{
	m_format_name.clear();
	m_format_name.push_back("x");
	m_format_name.push_back("y");
	m_format_name.push_back("z");
	if (!m_moveTable.empty())
		m_moveTable.clear();
	m_moveTable.push_back(0);
	m_moveTable.push_back(sizeof(double));
	m_moveTable.push_back(sizeof(double)*2);
	m_moveTable.push_back(sizeof(double)*3);
	m_dataSize = -1;		
}

int ConvertToEvr::Load_Dat( std::wstring Path )
{
	std::ifstream loaderx;
	loaderx.open(Path.c_str());
	std::string linedata;
	loaderx.ignore(std::numeric_limits<int>::max(), loaderx.widen('\n'));
	char ch[100];
	int i=0;
	loaderx.getline(ch, 100);
	loaderx.close();
	std::istringstream split;
	split.str(ch);
	while(!split.eof())
	{
		split >> linedata;
		i++;
	}
	// i = item's amount 算出有幾種屬性
	const int amount = i;
	m_format_count = amount;
	m_CreateLua.clear();
	std::string formatstring;
	for (int j = 0;j < amount-3;j++)
	{
		formatstring += "\"parameter" + ConvStr::GetStr(j+1) + "\",";
		m_format_name.push_back("parameter" + ConvStr::GetStr(j+1));
		m_moveTable.push_back(sizeof(double)*(j+4));
	}
	m_CreateLua.AddRawString("format_name", "{" + formatstring + "}");
	m_CreateLua.AddInt("format_count", amount-3);
	 // 算出每一組資料的大小
	m_dataSize = sizeof(double) * amount;
	// 讀出檔案
	std::ifstream loader;
	loader.open(Path.c_str());
	double tdata;
	for (;!loader.eof();)
	{
		if (loader.peek()==',')
			loader.get();
		loader >> tdata;
		m_dataVector.push_back(tdata);
	}
	m_total = m_dataVector.size()/amount;
	m_totalSize = m_dataVector.size() * sizeof(double);
	m_CreateLua.AddInt("total",m_total);
	// 儲存指標
	m_pData = (unsigned char*)&(*m_dataVector.begin());
	// 儲存大小改變
	m_Datamin.resize(amount-3);
	m_Datamax.resize(amount-3);
	for (int j = 0;j < amount-3;j++)
	{
		FindDeltaAndSpan(&(*(m_dataVector.begin()+j+3)),
			m_dataVector.size()-1, amount, 0.01, m_Datamin[j], m_Datamax[j], deltaZ, Zspan);
		// deltaZ, Zspan 是無意義的
	}
	// 計算xyz的最大最小跟間距
	FindDeltaAndSpan(&(*m_dataVector.begin()),
		m_dataVector.size()-1, amount, 0.01, Xmin, Xmax, deltaX, Xspan);
	FindDeltaAndSpan(&(*(m_dataVector.begin()+1)),
		m_dataVector.size()-1, amount, 0.01, Ymin, Ymax, deltaY, Yspan);
	FindDeltaAndSpan(&(*(m_dataVector.begin()+2)),
		m_dataVector.size()-1, amount, 0.01, Zmin, Zmax, deltaZ, Zspan);
	// 讀入資料的最大最小值
	for (int j = 0;j < amount-3;j++)
	{
		m_CreateLua.AddDouble("Datamin" + ConvStr::GetStr(j), m_Datamin[j]);
		m_CreateLua.AddDouble("Datamax" + ConvStr::GetStr(j), m_Datamax[j]);
	}
	m_CreateLua.AddDouble("DataAmount", i);
	m_CreateLua.AddDouble("Xmin",Xmin);
	m_CreateLua.AddDouble("Xmax",Xmax);
	m_CreateLua.AddDouble("deltaX",deltaX);
	m_CreateLua.AddDouble("Xspan",Xspan);
	m_CreateLua.AddDouble("Ymin",Ymin);
	m_CreateLua.AddDouble("Ymax",Ymax);
	m_CreateLua.AddDouble("deltaY",deltaY);
	m_CreateLua.AddDouble("Yspan",Yspan);
	m_CreateLua.AddDouble("Zmin",Zmin);
	m_CreateLua.AddDouble("Zmax",Zmax);
	m_CreateLua.AddDouble("deltaZ",deltaZ);
	m_CreateLua.AddDouble("Zspan",Zspan);
	return 0;
}

int ConvertToEvr::Save_Evr( std::wstring Path, std::wstring filename )
{
	m_CreateLua.AddString("data_format","binary");
	m_CreateLua.AddString("data", ConvStr::GetStr(filename.c_str())+std::string(".evr"));
	m_CreateLua.SaveLua(Path + L".lua");
	using namespace std;
	ofstream fOut;
	fOut.open((Path+L".evr").c_str(),ios::binary);
	if(fOut==NULL)
		return 0;
	if(m_pData==NULL)
	{
		fOut.close();
		return 0;
	}
	// Write the file into file
	fOut.write((char*)m_pData,m_totalSize);
	fOut.close();
	return 0;
}

int ConvertToEvr::Save_EvrA( std::wstring Path, std::wstring filename )
{
	m_CreateLua.AddString("data_format","ascii");
	m_CreateLua.AddString("data", ConvStr::GetStr(filename.c_str())+std::string(".evr"));
	m_CreateLua.SaveLua(Path + L".lua");
	using namespace std;
	ofstream fOut;
	fOut.open((Path+L".evr").c_str());
	if(fOut==NULL)
		return 0;
	if(m_pData==NULL)
	{
		fOut.close();
		return 0;
	}
	// Write the file into file
	for (int i=0;i<m_format_count;i++)
	{
		fOut << setw(16) << m_format_name[i];
	}
	fOut << std::endl;
	fOut.setf(ios_base::scientific);
	for (unsigned int i=0;i<m_total;i++)
	{		
		for (int j=0;j<m_format_count;j++)
		{
			int move = m_moveTable[j];
			fOut << setw(15) << *((double*)(m_pData+(i*m_dataSize+move))) << ",";
		}
		fOut << std::endl;
	}
	fOut.close();
	return 0;
}

