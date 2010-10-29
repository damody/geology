#pragma once
#include <vector>
#include <map>
#include <cstring>
#include <iostream>
#include <fstream>

class CreateLua
{
public:
	typedef std::vector< std::pair<std::string, int> > siVector;
	typedef std::vector< std::pair<std::string, double> > sdVector;
	typedef std::vector< std::pair<std::string, std::string> > ssVector;
	void AddInt(std::string name, int num);
	void AddDouble(std::string name, double num);
	void AddString(std::string name, std::string num);
	void AddRawString(std::string name, std::string num);
	void SaveLua(std::wstring str);
	void clear();
private:
	siVector siv;
	sdVector sdv;
	ssVector ssv,rsv;
};
