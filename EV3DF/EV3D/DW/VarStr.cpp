
#include "VarStr.h"
#include <cstdio>
#include <clocale>
wstrings split (const wchar_t *str, const wchar_t *del)
{
	int			len = wcslen(str);
	std::vector<wchar_t>	tstr;
	tstr.resize(len + 1);
	wcscpy(&tstr[0], str);
	wstrings strs;
	wchar_t *s = wcstok(&tstr[0], del);
	while (s != NULL)
	{
		strs.push_back(s);
		s = wcstok(NULL, del);
	}
	return strs;
}
strings split(const char *str, const char *del)
{
	int len = strlen(str);
	std::vector<char>	tstr;
	tstr.resize(len + 1);
	strcpy(&tstr[0], str);

	strings strs;
	char *s = strtok(&tstr[0], del);
	while (s != NULL)
	{
		strs.push_back(s);
		s = strtok(NULL, del);
	}

	return strs;
}

VarStr::VarStr(std::string ansi) :
	m_string(ansi)
{
	m_orgin_type = ORGIN_STRING;

	std::vector<wchar_t>	unicode;
	unicode.resize(ansi.length() + 1);
	mbstowcs(&unicode[0], &ansi[0], INT_MAX);
	m_wstring = std::wstring(&unicode[0]);
	sscanf(m_string.c_str(), "%lf", &m_double);
}

VarStr::VarStr(std::wstring unicode) :
	m_wstring(unicode)
{
	m_orgin_type = ORGIN_WSTRING;

	std::vector<char>	ansi;
	ansi.resize(unicode.length() * 2 + 1);
	setlocale(LC_ALL, "");	//設置成電腦本地Locale
	wcstombs(&ansi[0], &unicode[0], INT_MAX);
	setlocale(LC_ALL, "C"); //設置回默認
	m_string = std::string(&ansi[0]);
	sscanf(m_string.c_str(), "%lf", &m_double);
}

VarStr::VarStr(double number)
{
	m_orgin_type = ORGIN_DOUBLE;
	m_double = number;

	char chs[32];
	sprintf(chs, "%lf", number);
	m_string = std::string(chs);
	m_wstring = (std::wstring) VarStr(m_string);
}

VarStr::VarStr(float number)
{
	m_orgin_type = ORGIN_FLOAT;
	m_float = number;

	char chs[32];
	sprintf(chs, "%f", number);
	m_string = std::string(chs);
	m_wstring = (std::wstring) VarStr(m_string);
}

VarStr::VarStr(int number)
{
	m_orgin_type = ORGIN_INT;
	m_int = number;

	char chs[32];
	sprintf(chs, "%d", number);
	m_string = std::string(chs);
	m_wstring = (std::wstring) VarStr(m_string);
}
