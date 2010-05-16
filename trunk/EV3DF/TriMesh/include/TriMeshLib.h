#ifdef WIN32
// 4996 is warning for the  sprintf, 4244 is conversion from double to float
// 4305 is truncation from double to float
#	pragma warning (disable: 4267 4251 4065 4102 4996)
#	pragma warning( disable: 4190 4244 4305)
#define TRIMESH_LIB 1
#	ifdef TRIMESH_SOURCE
#		define TRIMESHDLL __declspec(dllexport)
#   elif defined TRIMESH_LIB
#		define TRIMESHDLL
#	else
#		define TRIMESHDLL __declspec(dllimport)
#	endif
#	define DLLEXPORT __declspec(dllexport)
#else
#	define TRIMESHDLL
#	define DLLEXPORT
#endif
