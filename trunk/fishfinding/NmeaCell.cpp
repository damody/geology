#include "NmeaCell.h"
#include <fstream>
#include <iostream>
#include <cstring>

NmeaCell::NmeaCell()
{
	NewInfo();
}

bool NmeaCell::InputLine( std::string str )
{
	if (str.length()<6)
		return false;
	m_buffer_str += str;
	for (int i=0;i<5;i++)
		m_Type[i] = str[i+1];
	if (CheckNewInfo())
		NewInfo();
	str += "\r\n";
	nmeaPARSER parser;
	nmea_parser_init(&parser);
	nmea_parse(&parser, str.c_str(), (int)str.length(), &m_lastinfo);
	m_infos.back() = m_lastinfo;
	return true;
}

bool NmeaCell::InputLine( char* str )
{
	int len = (int)strlen(str);
	for (int i=0;i<5;i++)
		m_Type[i] = str[i+1];
	if (CheckNewInfo())
		NewInfo();
	str[len] = '\r';
	str[len+1] = '\n';
	str[len+2] = '\0';
	nmeaPARSER parser;
	nmea_parser_init(&parser);
	nmea_parse(&parser, str, len+2, &m_lastinfo);
	m_infos.back() = m_lastinfo;
	return true;
}

bool NmeaCell::InputFile( const std::wstring str )
{
	std::ifstream is;
	is.open(str.c_str());
	char buffer[1024];
	while (!is.eof())
	{
		is.getline(buffer, 1024);
		InputLine(buffer);
	}
	is.close();
	return true;
}

const nmeaINFOs& NmeaCell::GetDataList()
{
	return m_infos;
}

const nmeaINFO& NmeaCell::GetLastData()
{
	return m_lastinfo;
}

void NmeaCell::NewInfo()
{
	nmea_zero_INFO(&m_lastinfo);
	memset(m_CheckInput, 0, sizeof(m_CheckInput));
	m_infos.push_back(m_lastinfo);
}
bool NmeaCell::CheckNewInfo()
{
	for (int i = 0;i < g_TypeTotal; i++)
	{
		if (0 == memcmp(m_Type, &(g_nmeaheads[i]), 5))
		{
			if (m_CheckInput[i])
				return true;
			m_CheckInput[i] = true;
			break;
		}
	}
	return false;
}
bool NmeaCell::InfoChange( nmeaINFO& info1, nmeaINFO& info2 )
{
	if (info1.lon !=0 && info2.lon != 0 && info1.lon != info2.lon)
		return true;
	if (info1.elv !=0 && info2.elv != 0 && info1.elv != info2.elv)
		return true;
	if (info1.lat !=0 && info2.lat != 0 && info1.lat != info2.lat)
		return true;
	if (info1.HDOP !=0 && info2.HDOP != 0 && info1.HDOP != info2.HDOP)
		return true;
	if (info1.PDOP !=0 && info2.PDOP != 0 && info1.PDOP != info2.PDOP)
		return true;
	if (info1.VDOP !=0 && info2.VDOP != 0 && info1.VDOP != info2.VDOP)
		return true;
	if (info1.depthinfo.depth_M !=0 && info2.depthinfo.depth_M != 0 && info1.depthinfo.depth_M != info2.depthinfo.depth_M)
		return true;
	return false;
}

void NmeaCell::SaveFile( const std::wstring str )
{
	std::fstream fIn;
	fIn.open(str.c_str(), std::ios_base::in | std::ios_base::out);
	if (fIn.good())
	{
		fIn.seekp(0,std::ios_base::end);
		fIn << m_buffer_str;
		m_buffer_str = "";
	}
	fIn.close();
}

int NmeaCell::GetTotal()
{
	return m_infos.size();
}


