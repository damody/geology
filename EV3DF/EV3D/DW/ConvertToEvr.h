// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
// In academic purposes only(2012/1/12)

#pragma once
#include <iomanip>
#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include "../Lua/CreateLua.h"
#include "ConvStr.h"
#include "FindSpan.h"

// this class is save data by Custom format
class ConvertToEvr
{
public:
	typedef std::vector<std::string> strVector;
	typedef std::vector<int> iVector;
	typedef std::vector<double> dVector;

	ConvertToEvr();
	int Load_Dat(std::wstring Path);
	int Save_Evr(std::wstring Path, std::wstring filename);
	int Save_EvrA(std::wstring path, std::wstring filename);
	bool		isLoad;
	dVector		m_dataVector;
	double		Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, deltaX, deltaY, deltaZ;
	dVector		m_Datamax, m_Datamin;
	int		Xspan, Yspan, Zspan;
	int		m_format_count;
	strVector	m_format_name;
	iVector		m_moveTable;
	std::string	m_dataPath;
	std::wstring	m_dataWPath;
	CreateLua	m_CreateLua;
private:
	int	m_dataSize, m_total;
	unsigned long	m_totalSize;
	unsigned char*	m_pData;
};
// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
// In academic purposes only(2012/1/12)
