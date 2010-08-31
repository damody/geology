#include "ConvertToEvr.h"
#include <memory>

ConvertToEvr::ConvertToEvr() :isLoad(false)
{
	m_format_name.clear();
	m_format_name.push_back("x");
	m_format_name.push_back("y");
	m_format_name.push_back("z");
	m_format_name.push_back("resistance");
	if (!m_moveTable.empty())
		m_moveTable.clear();
	m_moveTable.push_back(0);
	m_moveTable.push_back(sizeof(double));
	m_moveTable.push_back(sizeof(double)*2);
	m_moveTable.push_back(sizeof(double)*3);
	m_dataSize = sizeof(double)*4;
	m_format_count=4;		
}

int ConvertToEvr::Load_Dat( std::wstring Path )
{
	m_CreateLua.clear();
	m_CreateLua.AddRawString("format_name","{\"resistance\"}");
	m_CreateLua.AddInt("format_count",1);
	std::ifstream loader;
	loader.open(Path.c_str());
	double tdata;
	for (;!loader.eof();)
	{
		loader >> tdata;
		m_dataVector.push_back(tdata);
	}
	m_total = m_dataVector.size()/4;
	m_totalSize = m_total * m_dataSize;
	m_CreateLua.AddInt("total",m_total);
	m_pData = (unsigned char*)&(*m_dataVector.begin());
	FindDeltaAndSpan(&(*(m_dataVector.begin()+3)),
		m_dataVector.size()-1,4,0.01,Datamin,Datamax,deltaZ,Zspan);
	FindDeltaAndSpan(&(*m_dataVector.begin()),
		m_dataVector.size()-1,4,0.01,Xmin,Xmax,deltaX,Xspan);
	FindDeltaAndSpan(&(*(m_dataVector.begin()+1)),
		m_dataVector.size()-1,4,0.01,Ymin,Ymax,deltaY,Yspan);
	FindDeltaAndSpan(&(*(m_dataVector.begin()+2)),
		m_dataVector.size()-1,4,0.01,Zmin,Zmax,deltaZ,Zspan);
	m_CreateLua.AddDouble("Datamin",Datamin);
	m_CreateLua.AddDouble("Datamax",Datamax);
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

