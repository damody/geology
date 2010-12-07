#pragma once
#include <string>
#include <vector>
#include <nmea/nmea.h>

typedef std::vector<nmeaINFO> nmeaINFOs;

class NmeaCell
{
public:
	NmeaCell();
	bool InputLine(std::string str);
	bool InputFile(const std::wstring str);
	
	const nmeaINFOs& GetDataList();
	const nmeaINFO& GetLastData();
	bool InfoChange(nmeaINFO& info1, nmeaINFO& info2);
	
private:
	bool InputLine(char* str);
	bool CheckNewInfo();
	void NewInfo();
	char		m_Type[5];
	char		m_CheckInput[NMEA_TYPE_TOTAL];
	nmeaINFOs	m_infos;
	nmeaINFO	m_lastinfo;
	nmeaPARSER	m_parser;
};
