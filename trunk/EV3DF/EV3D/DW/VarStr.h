
#pragma once
#include <vector>
#include <string>
#include <cstring>
#include <cassert>
#include <cstdlib>
#pragma warning(disable : 4996)
typedef std::vector < std::string > strings;
strings					split(const char *str, const char *del);
typedef std::vector<std::wstring>	wstrings;
wstrings				split(const wchar_t *str, const wchar_t *del);
class VarStr
{
public:
	enum { ORGIN_INT, ORGIN_DOUBLE, ORGIN_FLOAT, ORGIN_STRING, ORGIN_WSTRING };
	int m_orgin_type;
	std::wstring m_wstring;
	std::string m_string;
	union
	{
		double m_double;
		float m_float;
		int m_int;
	};
	VarStr(std::string str);
	VarStr(std::wstring wstr);
	VarStr(double num);
	VarStr(float num);
	VarStr(int num);
	std::string	GetStr()	{ return m_string; }
	std::wstring	GetWstr()	{ return m_wstring; }
	operator const char * ()
	{
		return m_string.c_str();
	}
	operator const wchar_t * ()
	{
		return m_wstring.c_str();
	}
	operator double()
	{
		switch (m_orgin_type)
		{
		case ORGIN_INT:
			return m_int;

		case ORGIN_FLOAT:
			return m_float;

		case ORGIN_DOUBLE:
		case ORGIN_STRING:
		case ORGIN_WSTRING:
			return m_double;

		default:
			assert(0 && "VarStr has null value");
			return 0;
		}
	}

	operator float()
	{
		switch (m_orgin_type)
		{
		case ORGIN_INT:
			return (float) m_int;

		case ORGIN_FLOAT:
			return m_float;

		case ORGIN_DOUBLE:
		case ORGIN_STRING:
		case ORGIN_WSTRING:
			return (float) m_double;

		default:
			assert(0 && "VarStr has null value");
			return 0;
		}
	}

	operator int()
	{
		switch (m_orgin_type)
		{
		case ORGIN_INT:
			return m_int;

		case ORGIN_FLOAT:
			return (int) m_float;

		case ORGIN_DOUBLE:
		case ORGIN_STRING:
		case ORGIN_WSTRING:
			return (int) m_double;

		default:
			assert(0 && "VarStr has null value");
			return 0;
		}
	}
};
