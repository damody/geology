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
#include "LuaCell.h"
#include <cstdio>
#include <cassert>
#include <cstring>
#include <cctype>
#include <malloc.h>
#include <cstdlib>

char LuaCell::buffer[10];

LuaCell::LuaCell()
{
	m_LuaState = luaL_newstate();
	luaL_openlibs(m_LuaState);
}

LuaCell::~LuaCell()
{
	lua_close(m_LuaState);
}

bool LuaCell::callLua_Function(const char* functionName)
{
	lua_getglobal(m_LuaState, functionName);
	if (lua_isfunction(m_LuaState, -1))
	{
		if (lua_pcall(m_LuaState, 0, 0, 0) != 0)
		{
			return false;
		}
		else
		{
			return true;
		}		
	}
	return false;
}
//format： callLua_Function("GetTime", "fis>ii", 1.0f, 2, "3", *int, *int);
bool LuaCell::callLua_Function( const char* functionName, const char* signString, ... )
{
	va_list v1;
	int narg, nres;
	va_start(v1, signString);
	lua_getglobal(m_LuaState, functionName);

	for (narg = 0; *signString; narg++)
	{
		luaL_checkstack(m_LuaState, 1, "too many arguments");
		switch (*signString++)
		{
		case 'f':
			lua_pushnumber(m_LuaState, (double)va_arg(v1, float));
			break;
		case 'i':
			lua_pushinteger(m_LuaState, va_arg(v1, int));
			break;
		case 's':
			lua_pushstring(m_LuaState, va_arg(v1, char*));
			break;
		case '>':
			goto emdargs;
		default:
#if _DEBUG
			assert(0 && "callLua_Function args error!");
#endif
			break;
		}
	}
emdargs:
	nres = strlen(signString);
	if (lua_pcall(m_LuaState, narg, nres, 0) != 0)
	{
		return false;
	}

	nres = -nres;
	while (*signString)
	{
		switch (*signString++)
		{
		case 'f':
			*va_arg(v1, double *) = lua_tonumber(m_LuaState, nres);
			break;
		case 'i':
			*va_arg(v1, int *) = lua_tointeger(m_LuaState, nres);
			break;
		case 's':
			*va_arg(v1, const char **) = lua_tostring(m_LuaState, nres);
			break;
		default:
			printf("callLua_Function result args error!\n");
		}
		nres++;
	}

	va_end(v1);
	return true;
}
//if success return true
bool LuaCell::InputLuaFile(const char* path)
{
	if (luaL_loadfile(m_LuaState, path) || lua_pcall(m_LuaState, 0, 0, 0)) return false;
	return true;
}

// int LuaCell::getLua_Int(const char* variable)
// {
// 	lua_getglobal(m_LuaState, variable);
// 	int result = lua_tointeger(m_LuaState, -1);
// 	lua_pop(m_LuaState, 1);
// 	return result;
// }

//format： getLua_Int("is", 2, "3");
int LuaCell::getLua_Int( const char* signString, ... )
{
	va_list vlist;
	va_start(vlist, signString);
	int result = *(int*)getLua_Value(signString,BackType(INTEGER),vlist);
	va_end(vlist);
	return result;
}

double LuaCell::getLua_Double( const char* signString, ... )
{
	va_list vlist;
	va_start(vlist, signString);
	double result = *(int*)getLua_Value(signString,BackType(LONGFLOAT),vlist);
	va_end(vlist);
	return result;
}

const char* LuaCell::getLua_String( const char* signString, ... )
{
	va_list vlist;
	va_start(vlist, signString);
	const char *result = (char*)getLua_Value(signString,BackType(STRING),vlist);
	va_end(vlist);
	return result;
}

//從 table1\table2\table3\varableName 這樣的路徑中得到資料的函數
int LuaCell::getLua_Int_UsePath( const char* pathString )
{
	int result = *(int*)getLua_Value_UsePath(pathString,BackType(INTEGER));
	return result;
}

double LuaCell::getLua_Double_UsePath( const char* pathString )
{
	double result = *(double*)getLua_Value_UsePath(pathString,BackType(LONGFLOAT));
	return result;
}

const char* LuaCell::getLua_String_UsePath( const char* pathString )
{
	const char *result = (char*)getLua_Value_UsePath(pathString,BackType(STRING));
	return result;
}

void LuaCell::setLua_Int( const char* signString, int data, ... )
{
	va_list vlist;
	va_start(vlist, data);
	setLua_Value(signString,BackType(INTEGER),&data,vlist);
	va_end(vlist);
}

void LuaCell::setLua_Double( const char* signString, float data, ... )
{
	va_list vlist;
	va_start(vlist, data);
	setLua_Value(signString,BackType(LONGFLOAT),&data,vlist);
	va_end(vlist);
}
void LuaCell::setLua_String( const char* signString, const char* data, ... )
{
	va_list vlist;
	va_start(vlist, data);
	setLua_Value(signString,BackType(STRING),const_cast<char*>(data),vlist);
	va_end(vlist);
}
void LuaCell::setLua_Int_UsePath(const char* pathString, int data)
{
	setLua_Value_UsePath(pathString,BackType(INTEGER),&data);
}
void LuaCell::setLua_Double_UsePath(const char* pathString, float data)
{
	setLua_Value_UsePath(pathString,BackType(LONGFLOAT),&data);
}
void LuaCell::setLua_String_UsePath( const char* pathString, const char* data )
{
	setLua_Value_UsePath(pathString,BackType(STRING),const_cast<char*>(data));
}
bool LuaCell::setLua_NewTable(const char* table)
{
	lua_getglobal(m_LuaState, table);
	if (lua_istable(m_LuaState,-1))
	{
		lua_pop(m_LuaState, 1);
		return false;
	}
	else
	{
		lua_newtable(m_LuaState);
		lua_setglobal(m_LuaState, table);
		return true;
	}	
}

bool LuaCell::isDigitString( const char *str )
{
	for (const char *pos=str; *pos != '\0';pos++)
	{
		if (!isdigit(*pos)) 
			return false;
	}
	return true;
}

void* LuaCell::getLua_Value_UsePath( const char* pathString, int type )
{
	void *result = buffer;
	char path[250] = {0};
	const char *pos;
	char *target = path;
	int pathLayer = 0;
	for (pos = pathString;;pos++)
	{
		if (*pos == '\\' || *pos == '/') //get Layer string
		{
			++pathLayer;
			*target = '\0'; //c string end
			if (pathLayer == 1)
				lua_getglobal(m_LuaState, path);			
			else
			{
				if (isDigitString(path))	//is integer index?
					lua_pushinteger(m_LuaState, atoi(path));
				else
					lua_pushstring(m_LuaState, path);
				lua_gettable(m_LuaState, -2);
			}
			target = path;
		}
		else if (*pos == '\0') //get varable string
		{
			++pathLayer;
			*target = '\0'; //c string end
			if (pathLayer == 1) // if varable in global
				lua_getglobal(m_LuaState, path);
			else
			{
				if (isDigitString(path))	//is integer index?
					lua_pushinteger(m_LuaState, atoi(path));
				else
					lua_pushstring(m_LuaState, path);
				lua_gettable(m_LuaState, -2);
			}			
			switch (type)
			{
			case BackType(INTEGER):
				*((int*)result) = lua_tointeger(m_LuaState, -1);
				break;
			case BackType(LONGFLOAT):
				*((double*)result) = lua_tonumber(m_LuaState, -1);
				break;
			case BackType(STRING):
				result = const_cast<char*>(lua_tostring(m_LuaState, -1));
				break;
			}
			break;
		}
		else
		{
			*target = *pos;
			++target;
		}
	}	
	lua_pop(m_LuaState, pathLayer);
	return result;
}

void* LuaCell::getLua_Value( const char* signString, int type, va_list vlist )
{
	int narg;
	lua_getglobal(m_LuaState, va_arg(vlist, char*));
	for (narg = 0; *signString; narg++)
	{
		if (narg == 0)
		{
			*signString++;
			continue;
		}
		switch (*signString++)
		{
		case 's':
			lua_pushstring(m_LuaState, va_arg(vlist, char*));
			lua_gettable(m_LuaState, -2);
			break;
		case 'i':
			lua_pushinteger(m_LuaState, va_arg(vlist, int));
			lua_gettable(m_LuaState, -2);
			break;
		default:
#ifdef _DEBUG
			assert(0 && "getLua_Int args error!");
#endif
			break;
		}
	}
	void *result = buffer;
	switch (type)
	{
	case BackType(INTEGER):
		*((int*)result) = lua_tointeger(m_LuaState, -1);
		break;
	case BackType(LONGFLOAT):
		*((double*)result) = lua_tonumber(m_LuaState, -1);
		break;
	case BackType(STRING):
		result = const_cast<char*>(lua_tostring(m_LuaState, -1));
		break;
	}
	lua_pop(m_LuaState, narg);
	return result;
}

void LuaCell::setLua_Value_UsePath( const char* pathString, int type, void* data )
{
	char path[250] = {0};
	const char *pos;
	char *target = path;
	int pathLayer = 0;
	for (pos = pathString;;pos++)
	{
		if (*pos == '\\' || *pos == '/') //get Layer string
		{
			++pathLayer;
			*target = '\0'; //c string end
			if (pathLayer == 1)
				lua_getglobal(m_LuaState, path);
			else
			{
				if (isDigitString(path))	//is integer index?
					lua_pushinteger(m_LuaState, atoi(path));
				else
					lua_pushstring(m_LuaState, path);
				lua_gettable(m_LuaState, -2);
			}
			target = path;
		}
		else if (*pos == '\0') //get varable string
		{
			++pathLayer;
			*target = '\0'; //c string end
			switch (type)
			{
			case BackType(INTEGER):
				lua_pushinteger(m_LuaState, *static_cast<int*>(data));				
				break;
			case BackType(LONGFLOAT):
				lua_pushnumber(m_LuaState, *static_cast<double*>(data));
				break;
			case BackType(STRING):
				lua_pushstring(m_LuaState, static_cast<char*>(data));
				break;
			}
			if (pathLayer == 1) // if varable in global
				lua_setglobal(m_LuaState, path);
			else
			{
				if (isDigitString(path))	//is integer index?
					lua_rawseti(m_LuaState, -2, atoi(path));
				else
					lua_setfield(m_LuaState, -2, path);
			}			
			break;
		}
		else
		{
			*target = *pos;
			++target;
		}
	}	
	lua_pop(m_LuaState, pathLayer-1);
}

void LuaCell::setLua_Value( const char* signString, int type, void* data, va_list vlist )
{
	int narg;
	int len = strlen(signString)-1;
	lua_getglobal(m_LuaState, va_arg(vlist, char*));
	for (narg = 0; *signString; narg++)
	{
		if (narg == len) break;
		if (narg == 0)
		{
			*signString++;
			continue;
		}
		switch (*signString++)
		{
		case 's':
			lua_pushstring(m_LuaState, va_arg(vlist, char*));
			lua_gettable(m_LuaState, -2);
			break;
		case 'i':
			lua_pushinteger(m_LuaState, va_arg(vlist, int));
			lua_gettable(m_LuaState, -2);
			break;
		default:
#ifdef _DEBUG
			assert(0 && "getLua_Int args error!");
#endif
			break;
		}
	}
	switch (type)
	{
	case BackType(INTEGER):
		lua_pushinteger(m_LuaState, *static_cast<int*>(data));
		break;
	case BackType(LONGFLOAT):
		lua_pushnumber(m_LuaState, *static_cast<double*>(data));
		break;
	case BackType(STRING):
		lua_pushstring(m_LuaState, static_cast<char*>(data));
		break;
	}
	if (narg == 1) // if varable in global
		lua_setglobal(m_LuaState, va_arg(vlist, char*));
	else
	{
		switch (*signString)
		{
		case 's':
			lua_setfield(m_LuaState, -2, va_arg(vlist, char*));
			break;
		case 'i':
			lua_rawseti(m_LuaState, -2, va_arg(vlist, int));
			break;
		}
	}
	lua_pop(m_LuaState, narg);
}

template <> 
int LuaCell::getLua<int>()
{
	return lua_tointeger(m_LuaState, -1);
}
template <> 
double LuaCell::getLua<double>()
{
	return lua_tonumber(m_LuaState, -1);
}
template <> 
const char* LuaCell::getLua<const char*>()
{
	return lua_tostring(m_LuaState, -1);
}
void PrintFError (char * format, ...)
{
	char buffer[256];
	va_list args;
	va_start (args, format);
	vsprintf (buffer,format, args);
	perror (buffer);
	va_end (args);
}

template <class T> 
T LuaCell::getLua_UsePath(const char* pathString, ...)
{
	int len = strlen(pathString);
	// 預處理資料
	char *buffer = (char*)malloc(len+1);
	va_list args;
	va_start (args, pathString);
	vsprintf (buffer,pathString, args);
	// 準備計算
	T result;
	char path[250] = {0};
	const char *pos;
	char *target = path;
	int pathLayer = 0;
	for (pos = buffer;;pos++)
	{
		if (*pos == '\\' || *pos == '/') //get Layer string
		{
			++pathLayer;
			*target = '\0'; //c string end
			if (pathLayer == 1)
				lua_getglobal(m_LuaState, path);			
			else
			{
				if (isDigitString(path))	//is integer index?
					lua_pushinteger(m_LuaState, atoi(path));
				else
					lua_pushstring(m_LuaState, path);
				lua_gettable(m_LuaState, -2);
			}
			target = path;
		}
		else if (*pos == '\0') //get varable string
		{
			++pathLayer;
			*target = '\0'; //c string end
			if (pathLayer == 1) // if varable in global
				lua_getglobal(m_LuaState, path);
			else
			{
				if (isDigitString(path))	//is integer index?
					lua_pushinteger(m_LuaState, atoi(path));
				else
					lua_pushstring(m_LuaState, path);
				lua_gettable(m_LuaState, -2);
			}
			result = getLua<T>();
			break;
		}
		else
		{
			*target = *pos;
			++target;
		}
	}
	lua_pop(m_LuaState, pathLayer);
	free(buffer);
	va_end (args);
	return result;
}
template <>
void LuaCell::setLua<int>(int data)
{
	lua_pushinteger(m_LuaState, data);
}
template <>
void LuaCell::setLua<double>(double data)
{
	lua_pushnumber(m_LuaState, data);
}
template <>
void LuaCell::setLua<const char*>(const char* data)
{
	lua_pushstring(m_LuaState, data);
}
template <class T>
void LuaCell::setLua_UsePath( const char* pathString, T data )
{
	char path[250] = {0};
	const char *pos;
	char *target = path;
	int pathLayer = 0;
	for (pos = pathString;;pos++)
	{
		if (*pos == '\\' || *pos == '/') //get Layer string
		{
			++pathLayer;
			*target = '\0'; //c string end
			if (pathLayer == 1)
				lua_getglobal(m_LuaState, path);
			else
			{
				if (isDigitString(path))	//is integer index?
					lua_pushinteger(m_LuaState, atoi(path));
				else
					lua_pushstring(m_LuaState, path);
				lua_gettable(m_LuaState, -2);
			}
			target = path;
		}
		else if (*pos == '\0') //get varable string
		{
			++pathLayer;
			*target = '\0'; //c string end
			setLua<T>(data);
			if (pathLayer == 1) // if varable in global
				lua_setglobal(m_LuaState, path);
			else
			{
				if (isDigitString(path))	//is integer index?
					lua_rawseti(m_LuaState, -2, atoi(path));
				else
					lua_setfield(m_LuaState, -2, path);
			}			
			break;
		}
		else
		{
			*target = *pos;
			++target;
		}
	}	
	lua_pop(m_LuaState, pathLayer-1);
}

bool LuaCell::IsLua_NotNil( const char* pathString )
{
	char path[250] = {0};
	const char *pos;
	char *target = path;
	int pathLayer = 0;
	for (pos = pathString;;pos++)
	{
		if (*pos == '\\' || *pos == '/' || *pos == '\0') //get Layer string
		{
			++pathLayer;
			*target = '\0'; //c string end
			if (pathLayer == 1)
			{
				lua_getglobal(m_LuaState, path);
				if (lua_isnil(m_LuaState,-1)) 
				{
					lua_pop(m_LuaState, pathLayer);
					return false;
				}
			}
			else
			{
				if (!lua_istable(m_LuaState,-1)) 
				{
					lua_pop(m_LuaState, pathLayer);
					return false;
				}
				if (isDigitString(path))	//is integer index?
					lua_pushinteger(m_LuaState, atoi(path));
				else
					lua_pushstring(m_LuaState, path);
				lua_gettable(m_LuaState, -2);
				if (lua_isnil(m_LuaState, -1))
				{
					lua_pop(m_LuaState, pathLayer);
					return false;	
				}
			}
			target = path;
			if (*pos == '\0')
				break;
		}
		else
		{
			*target = *pos;
			++target;
		}
	}
	lua_pop(m_LuaState, pathLayer);
	return true;
}