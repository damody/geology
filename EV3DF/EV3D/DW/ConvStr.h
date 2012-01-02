// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
// In academic purposes only(2012/1/12)

#ifndef _ConvertStr_H_
#define _ConvertStr_H_
#include <vector>
#include <cstring>
#include <string>
#include <cstdio>

class ConvStr
{
public:	
	static std::string	GetStr(std::wstring ansi);
	static std::string	GetStr(std::string ansi);
	static std::string	GetStr(int number);
	static std::string	GetStr(unsigned int number);
	static std::string	GetStr(long long number);
	static std::string	GetStr(float number);
	static std::string	GetStr(double number);
	static std::wstring	GetWstr(std::wstring ansi);
	static std::wstring	GetWstr(std::string ansi);
	static std::wstring	GetWstr(int number);
	static std::wstring	GetWstr(unsigned int number);
	static std::wstring	GetWstr(long long number);
	static std::wstring	GetWstr(float number);
	static std::wstring	GetWstr(double number);
	static void WcharToChar(const wchar_t* unicode, char* ansi);
	static void CharToWchar(wchar_t* unicode, const char* ansi);
};

#endif // _ConvertStr_H_
// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
// In academic purposes only(2012/1/12)