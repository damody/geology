// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
// In academic purposes only(2012/1/12)
#pragma once
#include "vtkInterpolationGridingPolyDataFilter.h"
#define WHJ 0.000277778


enum {ONE_CUBEMODE = 1, CUT_PIECEMODE};

enum {INVERSE_FILTER, LIMITED_FILTER, NEARESTNEIGHBOR_FILTER};

struct vtkHeatParmeter
{
	vtkHeatParmeter()
		:Hv(0), Tzero(0), Rt(1), Fppc(0), Life(1), LimitTemperature(125)
	{
	}
	vtkHeatParmeter(double _Hv, double _Tzero, double _Rt, double _Fppc, double _Life, double _LimitTemperature)
		:Hv(_Hv), Tzero(_Tzero), Rt(_Rt), Fppc(_Fppc), Life(_Life), LimitTemperature(_LimitTemperature)
	{
	}
	double	Hv	//地熱儲存層體積比熱
		,Tzero	//排除溫度 (rejection temperature) (相當於一般地面年均氣溫 the average annual ambient temperature)
		,Rt	//地熱轉換效率 (儲存層之現地地熱能轉換為發電廠發電電能之效率)
		,Fppc	//電廠容量因子(power plant capacity factor) (發電廠年發電之發電平均時間比), 
		,Life	//電廠壽命。GeothermEX, Inc. (2005)建議一般電廠容量因子為0.95，電廠壽命以30年計算。
		,LimitTemperature; //溫度限制
};


class vtkHeatTranslationFilter : public vtkInterpolationGridingPolyDataFilter 
{
public:
	vtkTypeMacro(vtkHeatTranslationFilter,vtkPolyDataAlgorithm);
	static vtkHeatTranslationFilter *New();
	void SetParmeter(const vtkHeatParmeter& parmeter)
	{
		m_HeatParmeter = parmeter;
	}
	void SetFilter(int type){m_FilterType = type;}

	double GetEtotal(){return m_Etotal;}
	double GetEjh(){return m_Ejh;}
	double GetEinMW(){return m_Emw;}

	void SetDoInterpolate(bool setting){m_DoInterpolate = setting;}

	void SetNumberOfXYZ(double x, double y, double z);		//set number of point of x,y,z axis
	void SetNumberOfXYZ(double xyz[3]);

protected:
	vtkHeatParmeter	m_HeatParmeter;
	double m_Volume;	//cube volume
	double m_Etotal;
	double m_Ejh;		//E_in_j/h
	double m_Emw;		//E_in_MW
	int m_FilterType;

	bool m_DoInterpolate;

	double CmpE( double Tn );
	vtkHeatTranslationFilter(void);
	~vtkHeatTranslationFilter(void);

	int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);
	bool PosInBound(double pos[]);		//check position in bound
};
// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
// In academic purposes only(2012/1/12)
