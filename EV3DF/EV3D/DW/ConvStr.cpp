/*
Copyright (C) 2009  ¹CÀ¸¤Ñ«G¬É

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "ConvStr.h"
#include <windows.h>
#pragma warning(disable : 4996) // use sprintf
std::vector<std::string> SConvStr::array_char;
std::vector<std::wstring> SConvStr::array_wchar_t;

void SConvStr::WcharToChar(const wchar_t *unicode, char *ansi)
{
	int n;
	n = WideCharToMultiByte(CP_ACP, 0, unicode, -1, ansi, 0, NULL, FALSE);
	WideCharToMultiByte(CP_ACP, 0, unicode, -1, ansi, n, NULL, FALSE);
}

void SConvStr::CharToWchar(wchar_t *unicode, const char *ansi)
{
	int n;
	n = MultiByteToWideChar(CP_ACP, 0, ansi, -1, 0, 0);
	MultiByteToWideChar(CP_ACP, 0, ansi, -1, unicode, n);
} 

const wchar_t* SConvStr::GetWchar( const char *ansi )
{
	wchar_t unicode[MAX_PATH];
	CharToWchar(unicode, ansi);
	array_wchar_t.push_back(std::wstring(unicode));
	return array_wchar_t.rbegin()->c_str();
}

const wchar_t* SConvStr::GetWchar( const wchar_t* unicode )
{
	int len = wcslen(unicode)+1;
	wchar_t unicode2[MAX_PATH];
	wcsncpy_s(unicode2, len, unicode, len);
	array_wchar_t.push_back(std::wstring(unicode2));
	return array_wchar_t.rbegin()->c_str();
}

const wchar_t* SConvStr::GetWchar( int number )
{
	wchar_t data[MAX_PATH];
	wsprintf(data, L"%d", number);
	array_wchar_t.push_back(std::wstring(data));
	return array_wchar_t.rbegin()->c_str();
}

const wchar_t* SConvStr::GetWchar( long number )
{
	wchar_t data[MAX_PATH];
	wsprintf(data, L"%d", number);
	array_wchar_t.push_back(std::wstring(data));
	return array_wchar_t.rbegin()->c_str();
}

const wchar_t* SConvStr::GetWchar( long long number )
{
	wchar_t data[MAX_PATH];
	wsprintf(data, L"%ld", number);
	array_wchar_t.push_back(std::wstring(data));
	return array_wchar_t.rbegin()->c_str();
}

const wchar_t* SConvStr::GetWchar( float number )
{
	wchar_t data[MAX_PATH];
	wsprintf(data, L"%f", number);
	array_wchar_t.push_back(std::wstring(data));
	return array_wchar_t.rbegin()->c_str();
}

const wchar_t* SConvStr::GetWchar( double number )
{
	wchar_t data[MAX_PATH];
	wsprintf(data, L"%lf", number);
	array_wchar_t.push_back(std::wstring(data));
	return array_wchar_t.rbegin()->c_str();
}

const char* SConvStr::GetChar( const wchar_t *unicode )
{
	char ansi[MAX_PATH];
	WcharToChar(unicode, ansi);
	array_char.push_back(std::string(ansi));
	return array_char.rbegin()->c_str();
}

const char* SConvStr::GetChar( const char* ansi )
{
	int len = strlen(ansi)+1;
	char ansi2[MAX_PATH];
	strncpy_s(ansi2, len, ansi, len);
	array_char.push_back(std::string(ansi2));
	return array_char.rbegin()->c_str();
}

const char* SConvStr::GetChar( int number )
{
	char data[MAX_PATH];
	sprintf(data, "%d", number);
	array_char.push_back(std::string(data));
	return array_char.rbegin()->c_str();
}

const char* SConvStr::GetChar( long number )
{
	char data[MAX_PATH];
	sprintf(data, "%d", number);
	array_char.push_back(std::string(data));
	return array_char.rbegin()->c_str();
}

const char* SConvStr::GetChar( long long number )
{
	char data[MAX_PATH];
	sprintf(data, "%ld", number);
	array_char.push_back(std::string(data));
	return array_char.rbegin()->c_str();
}

const char* SConvStr::GetChar( float number )
{
	char data[MAX_PATH];
	sprintf(data, "%f", number);
	array_char.push_back(std::string(data));
	return array_char.rbegin()->c_str();
}

const char* SConvStr::GetChar( double number )
{
	char data[MAX_PATH];
	sprintf(data, "%lf", number);
	array_char.push_back(std::string(data));
	return array_char.rbegin()->c_str();
}
std::wstring SConvStr::Getwstring( const wchar_t* unicode )
{
	return std::wstring(unicode);
}

std::wstring SConvStr::Getwstring( const char* ansi )
{
	std::wstring result;
	result = GetWchar(ansi);
	return result;
}

std::string SConvStr::Getstring( const wchar_t* unicode )
{
	std::string result;
	result = GetChar(unicode);
	return result;
}

std::string SConvStr::Getstring( const char* ansi )
{
	return std::string(ansi);
}
void SConvStr::FreeAllMemory()
{
	std::vector<std::string>::iterator i;
	for (i = array_char.begin();i != array_char.end();i++)
	{
		i->clear();
	}
	array_char.clear();
	std::vector<std::wstring>::iterator j;
	for (j = array_wchar_t.begin();j != array_wchar_t.end();j++)
	{
		j->clear();
	}
	array_wchar_t.clear();
}


void ConvStr::WcharToChar(const wchar_t *unicode, char *ansi)
{
	int n;
	n = WideCharToMultiByte(CP_ACP, 0, unicode, -1, ansi, 0, NULL, FALSE);
	WideCharToMultiByte(CP_ACP, 0, unicode, -1, ansi, n, NULL, FALSE);
}

void ConvStr::CharToWchar(wchar_t *unicode, const char *ansi)
{
	int n;
	n = MultiByteToWideChar(CP_ACP, 0, ansi, -1, 0, 0);
	MultiByteToWideChar(CP_ACP, 0, ansi, -1, unicode, n);
} 

const wchar_t* ConvStr::GetWchar( const char *ansi )
{
	wchar_t unicode[MAX_PATH];
	CharToWchar(unicode, ansi);
	array_wchar_t.push_back(std::wstring(unicode));
	return array_wchar_t.rbegin()->c_str();
}

const wchar_t* ConvStr::GetWchar( const wchar_t* unicode )
{
	int len = wcslen(unicode)+1;
	wchar_t unicode2[MAX_PATH];
	wcsncpy_s(unicode2, len, unicode, len);
	array_wchar_t.push_back(std::wstring(unicode2));
	return array_wchar_t.rbegin()->c_str();
}

const char* ConvStr::GetChar( const wchar_t *unicode )
{
	char ansi[MAX_PATH];
	WcharToChar(unicode, ansi);
	array_char.push_back(std::string(ansi));
	return array_char.rbegin()->c_str();
}

const char* ConvStr::GetChar( const char* ansi )
{
	int len = strlen(ansi)+1;
	char ansi2[MAX_PATH];
	strncpy_s(ansi2, len, ansi, len);
	array_char.push_back(std::string(ansi2));
	return array_char.rbegin()->c_str();
}

std::wstring ConvStr::Getwstring( const wchar_t* unicode )
{
	return std::wstring(unicode);
}

std::wstring ConvStr::Getwstring( const char* ansi )
{
	std::wstring result;
	result = GetWchar(ansi);
	return result;
}

std::string ConvStr::Getstring( const wchar_t* unicode )
{
	std::string result;
	result = GetChar(unicode);
	return result;
}

std::string ConvStr::Getstring( const char* ansi )
{
	return std::string(ansi);
}

void ConvStr::FreeAllMemory()
{
	std::vector<std::string>::iterator i;
	for (i = array_char.begin();i != array_char.end();i++)
	{
		i->clear();
	}
	array_char.clear();
	std::vector<std::wstring>::iterator j;
	for (j = array_wchar_t.begin();j != array_wchar_t.end();j++)
	{
		j->clear();
	}
	array_wchar_t.clear();
}