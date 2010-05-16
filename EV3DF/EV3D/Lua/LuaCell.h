/*
Copyright (C) 2009  遊戲天亮界

This program is free software: you can redistribute it and/or modify
it under the terms of the Lesser GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
/*
使用了void*與template做了同樣的功能
對lua做的包裝，用法大致如下
const char* a = cell.getLua_String_UsePath("table2\\table23\\table2\\1");
cell.setLua_Int_UsePath("table2\\table23\\table2\\1",100);
cell.setLua_String("sssi","20","table2","table23","table2",1);
const char* b = cell.getLua_String_UsePath("table2\\table23\\table2\\1");
const char *a = cell.getLua_String("sssi","table2","table23","table2",1);
*/
#ifndef LuaCell_Im42c45f58m1233b1a3831mm7e5f_H
#define LuaCell_Im42c45f58m1233b1a3831mm7e5f_H
#include "lua.hpp"
#include "lauxlib.h"
#include "lualib.h"
#ifdef _DEBUG
#pragma comment(lib, "Lua/lua5.1d.lib")
#else
#pragma comment(lib, "Lua/lua5.1.lib")
#endif // _DEBUG

#include <cstdarg>

class LuaCell
{
public:
	enum BackType {INTEGER,LONGFLOAT,STRING};
	LuaCell();
	~LuaCell();
	bool callLua_Function(const char* functionName);
	bool callLua_Function(const char* functionName, const char* signString, ...);
	bool InputLuaFile(const char* path);
	int		getLua_Int(const char* signString, ...);
	double		getLua_Double(const char* signString, ...);
	const char*	getLua_String(const char* signString, ...);
	int		getLua_Int_UsePath(const char* pathString);
	double		getLua_Double_UsePath(const char* pathString);
	const char*	getLua_String_UsePath(const char* pathString);
	template <class T> T getLua_UsePath(const char* pathString);
	void setLua_Int_UsePath(const char* pathString, int data);
	void setLua_Double_UsePath(const char* pathString, float data);
	void setLua_String_UsePath(const char* pathString, const char* data);
	void setLua_Value_UsePath(const char* pathString, int type, void* data);
	template <class T> void setLua_UsePath(const char* pathString, T data);
	void setLua_Int(const char* pathString, int value, ...);
	void setLua_Double(const char* signString, float value, ...);
	void setLua_String(const char* signString, const char* value, ...);
	void setLua_Value( const char* signString, int type, void* data, va_list vlist );
	bool setLua_NewTable(const char* table);
	bool IsLua_NotNil(const char* pathString);
private:
	void*		getLua_Value_UsePath(const char* pathString, int type );
	void*		getLua_Value(const char* pathString, int type, va_list vlist);
	template <class T> T getLua() {return NULL;}
	template <> int		getLua<int>();
	template <> double	getLua<double>();
	template <> const char*	getLua<const char*>();
	template <class T> void setLua(T data) {}
	template <> void setLua<int>(int data);
	template <> void setLua<double>(double data);
	template <> void setLua<const char*>(const char* data);
	bool isDigitString(const char *str);
	lua_State* m_LuaState;
	static char buffer[10];
};
//要預先宣告偏特化才不會連結錯誤
template int LuaCell::getLua_UsePath<int>(const char* pathString);
template double LuaCell::getLua_UsePath<double>(const char* pathString);
template const char* LuaCell::getLua_UsePath<const char*>(const char* pathString);
template void LuaCell::setLua_UsePath(const char* pathString, int data);
template void LuaCell::setLua_UsePath(const char* pathString, double data);
template void LuaCell::setLua_UsePath(const char* pathString, const char* data);
#endif // LuaCell_Im42c45f58m1233b1a3831mm7e5f_H

