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
	// return all data like vector<nmeaINFO>
	const nmeaINFOs& GetDataList();
	/// return data of last add
	const nmeaINFO& GetLastData();
	/// return GetOne's output index to first
	void	ReSetGetOne();
	/// return one element and pointer next
	const nmeaINFO& GetOne();
	/// return GetOne now index
	int	GetOneIndex();
	/// check info's difference
	bool	InfoChange(nmeaINFO& info1, nmeaINFO& info2);
	/// append not save's info to file
	void	SaveFile(const std::wstring str);
	/// return total of nmeaINFOs
	int	GetTotal();
	bool	CheckInfoCorrect();
private:
	bool InputLine(char* str);
	bool CheckNewInfo();
	void NewInfo();
	std::string	m_buffer_str;
	char		m_Type[5];
	char		m_CheckInput[NMEA_TYPE_TOTAL];
	nmeaINFOs	m_infos;
	nmeaINFO	m_lastinfo;
	nmeaPARSER	m_parser;
	int		m_output_index;
};
