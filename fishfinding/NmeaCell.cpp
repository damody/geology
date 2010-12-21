#include "NmeaCell.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <cassert>

NmeaCell::NmeaCell()
{
	nmea_zero_INFO(&m_lastinfo);
	m_output_index = 0;
}

bool NmeaCell::InputLine( std::string str )
{
	if (str.length()<10)
		return false;
	m_buffer_str += str;
	for (int i=0;i<5;i++)
		m_Type[i] = str[i+1];
	if (!CheckLastInfoCorrect())
		return false;
	if (CheckNewInfo())
		NewInfo();
	nmeaPARSER parser;
	nmea_parser_init(&parser);
	nmea_parse(&parser, str.c_str(), (int)str.length(), &m_lastinfo);
	nmea_parser_destroy(&parser);
	if (m_infos.empty())
		NewInfo();
	m_infos.back() = m_lastinfo;
	return true;
}

bool NmeaCell::InputLine( char* str )
{
	m_buffer_str += str;
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
 	nmea_parser_destroy(&parser);
	if (m_infos.empty())
		NewInfo();
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
		memset(buffer, 0, sizeof(buffer));
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
	if (m_lastinfo.lat!=0)
		return m_lastinfo;
	else
	{
		if (m_infos.size()>1)
			return *(m_infos.end()-2);
		else
			return m_lastinfo;
	}
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
bool NmeaCell::CheckLastInfoCorrect()
{
	for (int i = 0;i < g_TypeTotal; i++)
	{
		if (0 == memcmp(m_Type, &(g_nmeaheads[i]), 5))
		{
			return true;
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
	std::ofstream fIn;
	fIn.open(str.c_str(), std::ios_base::out | std::ios_base::app);
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

void NmeaCell::ReSetGetOne()
{
	m_output_index = 0;
}

int NmeaCell::GetOneIndex()
{
	return m_output_index;
}

const nmeaINFO& NmeaCell::GetOne()
{
	return m_infos[m_output_index++];
}

void NmeaCell::Clear()
{
	m_infos.clear();
	m_last_str = "";
	m_buffer_str = "";
	m_output_index = 0;
	memset(m_Type, 0, 5);
}

void NmeaCell::InputRawData( const char* data, const int size )
{
	const int b_size = size+m_last_str.length()+10;
	char *buffer = (char*)malloc(sizeof(char)*b_size);
	memset(buffer, 0, sizeof(char)*b_size);
	memcpy(buffer, m_last_str.c_str(), m_last_str.length());
	memcpy(buffer + m_last_str.length(), data, size);
	// 增加文字到OutputText
	char xbuf[200];
	memset(xbuf, 0, sizeof(char)*200);
	int len = strlen(buffer);
	for (;len>10;)
	{
		for (int i=0;i<len;i++)
		{
			if (i!=0 && buffer[i]=='$')
			{
				int ok_offset;
				for (ok_offset=0;ok_offset<i;ok_offset++)
				{
					if (buffer[ok_offset] != -52)
						break;
				}
				i -= ok_offset;
				//assert(i-1>=0 && i+1<200);
				memcpy(xbuf, buffer+ok_offset, i);
				memcpy(buffer, buffer+ok_offset+i, len-ok_offset-i+1);
				bool has_rn = false;
				for (int offset=1;offset<5;offset++)
				{
					if ('\r' == xbuf[i-offset])
					{
						xbuf[i-offset] = '\r';
						xbuf[i-offset+1] = '\n';
						xbuf[i-offset+2] = '\0';
						has_rn = true;
					}
				}
				if (!has_rn)
				{
					xbuf[i-1] = '\r';
					xbuf[i+0] = '\n';
					xbuf[i+1] = '\0';
				}
				InputLine(xbuf);
				break;
			}
			else if (i == len-1)
			{
				int ok_offset;
				// 解決測試程式的bug字串有0xcc
				for (ok_offset=strlen(buffer)-1;ok_offset>0;ok_offset--)
				{
					if (buffer[ok_offset] == -52)
						buffer[ok_offset] = '\0';
				}
				buffer[len] = '\0';
				m_last_str = buffer;
				assert(m_last_str.length()<100);
				buffer[0] = '\0';
			}
		}
		len = strlen(buffer);
	}
	free(buffer);
}


