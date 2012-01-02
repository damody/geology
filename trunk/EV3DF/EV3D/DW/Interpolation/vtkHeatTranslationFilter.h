// author: t1238142000@gmail.com Liang-Shiuan Huang ���G�a
// author: a910000@gmail.com Kuang-Yi Chen ������
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
	double	Hv	//�a���x�s�h��n���
		,Tzero	//�ư��ū� (rejection temperature) (�۷��@��a���~����� the average annual ambient temperature)
		,Rt	//�a���ഫ�Ĳv (�x�s�h���{�a�a�����ഫ���o�q�t�o�q�q�ध�Ĳv)
		,Fppc	//�q�t�e�q�]�l(power plant capacity factor) (�o�q�t�~�o�q���o�q�����ɶ���), 
		,Life	//�q�t�ةR�CGeothermEX, Inc. (2005)��ĳ�@��q�t�e�q�]�l��0.95�A�q�t�ةR�H30�~�p��C
		,LimitTemperature; //�ū׭���
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
// author: t1238142000@gmail.com Liang-Shiuan Huang ���G�a
// author: a910000@gmail.com Kuang-Yi Chen ������
// In academic purposes only(2012/1/12)
