// author: t1238142000@gmail.com Liang-Shiuan Huang ���G�a
// author: a910000@gmail.com Kuang-Yi Chen ������
// In academic purposes only(2012/1/12)

#pragma once
#include <vtkMath.h>
#include <cmath>
#include <string>

const static double Math_PI = 3.141592653;
const static double a = 6378137.0;
const static double b = 6356752.314245;
const static double lon0 = 121 * Math_PI / 180;
const static double k0 = 0.9999;
const static int dx = 250000;

class CoordinateTransform
{
	
public:
	CoordinateTransform()
	{
	}

	//��WGS84�g�n�׫פ����নTWD97����
	static void lonlat_To_TWD97x(int lonD,int lonM,int lonS,int latD,int latM,int latS, double *x, double *y);
	//��WGS84�g�n�ש����নTWD97����
	static void lonlat_To_TWD97(double RadianLon, double RadianLat, double *x, double *y);
	static void TWD97_To_lonlat(double x, double y, double *lon ,double *lat);
private:
	static void Cal_lonlat_To_twd97(double lon ,double lat, double *x, double *y);

	
};

// author: t1238142000@gmail.com Liang-Shiuan Huang ���G�a
// author: a910000@gmail.com Kuang-Yi Chen ������
// In academic purposes only(2012/1/12)