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
#ifndef _ConvertStr_H_
#define _ConvertStr_H_
#include <vector>
#include <cstring>

class SConvStr
{
public:
	~SConvStr()
	{
		FreeAllMemory();
	}
	static void WcharToChar(const wchar_t* unicode, char* ansi);
	static void CharToWchar(wchar_t* unicode, const char* ansi);
	static const wchar_t*	GetWchar(const char* ansi);
	static const wchar_t*	GetWchar(const wchar_t* ansi);
	static const char*	GetChar(const wchar_t* ansi);
	static const char*	GetChar(const char* ansi);
	static std::string	Getstring(const wchar_t* ansi);
	static std::string	Getstring(const char* ansi);
	static std::wstring	Getwstring(const wchar_t* ansi);
	static std::wstring	Getwstring(const char* ansi);
	static const wchar_t*	GetWchar(int number);
	static const wchar_t*	GetWchar(long number);
	static const wchar_t*	GetWchar(long long number);
	static const wchar_t*	GetWchar(float number);
	static const wchar_t*	GetWchar(double number);
	static const char*	GetChar(int number);
	static const char*	GetChar(long number);
	static const char*	GetChar(long long number);
	static const char*	GetChar(float number);
	static const char*	GetChar(double number);
	static void FreeAllMemory();
private:
	static std::vector<std::string>		array_char;
	static std::vector<std::wstring>	array_wchar_t;
};

class ConvStr
{
public:
	~ConvStr()
	{
		FreeAllMemory();
	}
	void WcharToChar(const wchar_t* unicode, char* ansi);
	void CharToWchar(wchar_t* unicode, const char* ansi);
	const wchar_t*	GetWchar(const char* ansi);
	const wchar_t*	GetWchar(const wchar_t* ansi);
	const char*		GetChar(const wchar_t* ansi);
	const char*		GetChar(const char* ansi);
	std::string	Getstring(const wchar_t* ansi);
	std::string	Getstring(const char* ansi);
	std::wstring	Getwstring(const wchar_t* ansi);
	std::wstring	Getwstring(const char* ansi);
	void FreeAllMemory();
private:
	std::vector<std::string>	array_char;
	std::vector<std::wstring>	array_wchar_t;
};
#endif // _ConvertStr_H_