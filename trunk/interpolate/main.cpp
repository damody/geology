// athour: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a, ³¯¥ú«³
// In academic purposes only
#pragma warning(disable:4996) 
#include <vtkSmartPointer.h>
#include <cassert>
#include <vtkSurfaceReconstructionFilter.h>
#include <vtkProgrammableSource.h>
#include <vtkContourFilter.h>
#include <vtkReverseSense.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkPolyData.h>
#include <vtkCamera.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkPoints.h>
#include <vtkDoubleArray.h>
#include <vtkMath.h>
#include "BenchTicks.hpp"
#include <cstdio>
#include <cctype>


#include "vtkNearestNeighborFilter.h"
#include "vtkNearestNeighborFilterCuda.h"
#include "vtkInverseDistanceFilter.h"
#include "vtkInverseDistanceFilterCuda.h"
#include "vtkLimitedInverseDistanceFilter.h"
#include "vtkLimitedInverseDistanceFilterCuda.h"
#include "vtkKrigingFilter.h"
#include "vtkKrigingFilterCuda.h"
#include <vtkPLYWriter.h>
#include "vtkKdtreeLimitedInverseDistanceFilterCuda.h"


#define VTK_SMART_POINTER(x) \
	typedef vtkSmartPointer< x >	x##_Sptr; \
	typedef std::vector< x##_Sptr >	x##_Sptrs;
VTK_SMART_POINTER(vtkPoints)
VTK_SMART_POINTER(vtkPolyData)
VTK_SMART_POINTER(vtkDoubleArray)

VTK_SMART_POINTER(vtkInterpolationGridingPolyDataFilter)
VTK_SMART_POINTER(vtkNearestNeighborFilter)
VTK_SMART_POINTER(vtkNearestNeighborFilterCuda)
VTK_SMART_POINTER(vtkInverseDistanceFilter)
VTK_SMART_POINTER(vtkInverseDistanceFilterCuda)
VTK_SMART_POINTER(vtkKrigingFilter)
VTK_SMART_POINTER(vtkKrigingFilterCuda)
VTK_SMART_POINTER(vtkLimitedInverseDistanceFilter)
VTK_SMART_POINTER(vtkLimitedInverseDistanceFilterCuda)
VTK_SMART_POINTER(vtkKdtreeLimitedInverseDistanceFilterCuda)

template <class T>
void vtkSmartNew(vtkSmartPointer<T>& Ptr)
{
	Ptr = vtkSmartPointer<T>::New();
	assert(Ptr.GetPointer() != 0);
}

bool LoadInputData(char* filename, vtkPolyData* polydata)
{
	vtkPoints_Sptr input_points;
	vtkDoubleArray_Sptr input_scalars;
	vtkSmartNew(input_scalars);
	vtkSmartNew(input_points);
	double x,y,z,s;
	char c = 0;
	bool haveComma = false;
	FILE* inf = fopen(filename, "r");
	if (!inf)
	{
		printf("Input file error!\n");
		return false;
	}
	while(c!='\n')
	{
		fscanf(inf, "%c", &c);
		if (c==',')
		{
			haveComma = true;
			break;
		}
	}
	rewind(inf);
	while(true)
	{
		if( fscanf(inf, "%lf", &x) == EOF)
			break;
		if (haveComma)
			fscanf(inf, " ,%lf ,%lf ,%lf", &y, &z, &s);
		else
			fscanf(inf, "%lf%lf%lf", &y, &z, &s);
		input_points->InsertNextPoint(x, y, z);
		input_scalars->InsertNextTuple1(s);
	}
	polydata->SetPoints(input_points);
	polydata->GetPointData()->SetScalars(input_scalars);
	fclose(inf);
	return true;
}

int main(int argc, char *argv[])
{
	char *ifilename=NULL;
	char ofilename[256] = "a.out";
	FILE *inf = NULL, *outf = NULL;
	int seg[3] = {50, 50, 50};
	double bound[6];
	double interval[3];
	enum{ USE_NEARESTNEIGHBOR_FILTER, USE_INVERSEDISTANCE_FILTER, 
		USE_KRIGING_FILTER, USE_LIMITEINVERSE_FILTER_NUMBER,
		USE_LIMITEINVERSE_FILTER_RADIUS,
		ONLY_PRINT_INFO, ONLY_PRINT_INSTRUCTION };
	enum{ USE_GPU_COMPUTE, USE_CPU_COMPUTE};
	int interpolater = USE_NEARESTNEIGHBOR_FILTER;
	int computerunit = USE_GPU_COMPUTE;
	double krigingStep = 0;
	double inversePower = 2;
	int	limitnum = -1;
	float	limitRadius = -1;
	bool usedefaultbound = true;
	bool autorun = false;
	bool useSharedMem = false;
	bool noOutput = false;
	BenchTicks_t bt;
	int cudathread = 64;

	vtkInterpolationGridingPolyDataFilter_Sptr	InterpolationFilter;
	vtkPolyData_Sptr input_polydata;
	vtkPolyData* outdata;
	vtkDoubleArray* outScalars; 
	vtkSmartNew(input_polydata);	
	bool cmderr = false;
	if (argc < 2)
	{
		printf("-k : Kriging\n");
		printf("-i : InverseDistance\n");
		printf("-lr%%f: Limited Radius of Points Inverse Distance\n");
		printf("-ln%%d: Limited Number of Points Inverse Distance\n");
		printf("-n : NearestNeighbor(default)\n");
		printf("-c : CPU\n");
		printf("-g : GPU(default)\n");
		printf("-t%%d : number of Cuda thread\n");
		printf("-ms : use shared memory\n");
		printf("-noo : no output\n");
		printf("-sSx,Sy,Sz : segment, Sx is segment of x, Sy is segment of y, Sz is segment of z, default 50\n");
		printf("-bMinX,MaxX,MinY,MaxY,MinZ,MaxZ : Bounding, default is input data bounding\n");
		printf("-I : only print number of data point, data bounding\n");
		printf("-h : only print instruction\n");
		printf("-a : automatic run\n");
		printf("interpolate input -[kin] -[cg] -[o] -[s] -[b] -[a] or\n");
		printf("interpolate input -I or\n");
		printf("interpolate -h\n");
		return 0;
	}
	else
	{
		for (int i=1; i<argc; i++)
		{
			if (memcmp(argv[i], "-k", 2)==0)
			{
				double v=0;
				interpolater = USE_KRIGING_FILTER;
				sscanf(argv[i], "-k%lf", &v);
				if (v>0)
				{
					krigingStep = v;
				}
			}
			else if (memcmp(argv[i], "-i", 2)==0)
			{
				interpolater = USE_INVERSEDISTANCE_FILTER;
				sscanf(argv[i], "-i%lf", &inversePower);
			}
			else if (memcmp(argv[i], "-noo", 4)==0)
			{
				noOutput = true;
			}
			else if (memcmp(argv[i], "-n", 2)==0)
				interpolater = USE_NEARESTNEIGHBOR_FILTER;
			else if (memcmp(argv[i], "-ln", 3)==0)
			{
				inversePower = 2;
				interpolater = USE_LIMITEINVERSE_FILTER_NUMBER;
				sscanf(argv[i], "-ln%d", &limitnum);
			}
			else if (memcmp(argv[i], "-lr", 3)==0)
			{
				inversePower = 2;
				interpolater = USE_LIMITEINVERSE_FILTER_RADIUS;
				sscanf(argv[i], "-lr%f", &limitRadius);
			}
			else if (memcmp(argv[i], "-c", 2)==0)
				computerunit = USE_CPU_COMPUTE;
			else if (memcmp(argv[i], "-g", 2)==0)
				computerunit = USE_GPU_COMPUTE;
			else if (memcmp(argv[i], "-s", 2)==0)
				sscanf(argv[i]+2, "%d,%d,%d", seg, seg+1, seg+2);
			else if (memcmp(argv[i], "-ms", 3)==0)
				useSharedMem = true;
			else if (memcmp(argv[i], "-b", 2)==0)
			{
				sscanf(argv[i]+2, "%lf,%lf,%lf,%lf,%lf,%lf", bound, bound+1, bound+2, bound+3, bound+4, bound+5);
				usedefaultbound = false;
			}
			else if (memcmp(argv[i], "-t", 2)==0)
			{
				sscanf(argv[i]+2, "%d", &cudathread);
			}
			else if (memcmp(argv[i], "-I", 2)==0)
			{
				interpolater = ONLY_PRINT_INFO;
				break;
			}
			else if (memcmp(argv[i], "-h", 2)==0)
			{
				interpolater = ONLY_PRINT_INSTRUCTION;
				break;
			}
			else if (memcmp(argv[i], "-o", 2)==0)
			{
				if (i+1>=argc)
				{
					printf("Error, no filename before -o\n");
					return 0;
				}
				i++;
				strcpy(ofilename, argv[i]);
			}
			else if (memcmp(argv[i], "-a", 2)==0)
				autorun = true;
		}
	}
	if (interpolater == ONLY_PRINT_INSTRUCTION)
	{
		printf("-k : Kriging\n");
		printf("-i : InverseDistance\n");
		printf("-lr%%f: Limited Radius of Points Inverse Distance\n");
		printf("-ln%%d: Limited Number of Points Inverse Distance\n");
		printf("-n : NearestNeighbor(default)\n");
		printf("-c : CPU\n");
		printf("-g : GPU(default)\n");
		printf("-t%%d : number of Cuda thread\n");
		printf("-ms : use shared memory\n");
		printf("-noo : no output\n");
		printf("-sSx,Sy,Sz : segment, Sx is segment of x, Sy is segment of y, Sz is segment of z, default 50\n");
		printf("-bMinX,MaxX,MinY,MaxY,MinZ,MaxZ : Bounding, default is input data bounding\n");
		printf("-I : only print number of data point, data bounding\n");
		printf("-h : only print instruction\n");
		printf("-a : automatic run\n");
		printf("interpolate input -[kin] -[cg] -[o] -[s] -[b] -[a] or\n");
		printf("interpolate input -I or\n");
		printf("interpolate -h\n");
		return 0;
	}	
	ifilename = argv[1];
	if (!LoadInputData(ifilename, input_polydata))
		return 0;
	if (interpolater == ONLY_PRINT_INFO)
	{
		printf("Number of input data points:%d\n", input_polydata->GetNumberOfPoints());
		input_polydata->GetBounds(bound);
		printf("Bounding\n");
		printf("MinX:%lf, MaxX:%lf\n", bound[0], bound[1]);
		printf("MinY:%lf, MaxY:%lf\n", bound[2], bound[3]);
		printf("MinZ:%lf, MaxZ:%lf\n", bound[4], bound[5]);
		return 0;
	}

	if (seg[0]<=0 || seg[1]<=0 || seg[2]<=0)
	{
		printf("Segment must be positive!");
		return 0;
	}

	if (interpolater == USE_NEARESTNEIGHBOR_FILTER)
	{
		printf("Use nearest-neighbor, ");
		if (computerunit==USE_CPU_COMPUTE)
			InterpolationFilter = vtkSmartPointer<vtkNearestNeighborFilter>::New();
		else
			InterpolationFilter = vtkSmartPointer<vtkNearestNeighborFilterCuda>::New();
	}
	else if (interpolater == USE_INVERSEDISTANCE_FILTER)
	{
		printf("Use inverse-distance, power:%f", inversePower);
		if (computerunit==USE_CPU_COMPUTE)
		{
			vtkSmartPointer<vtkInverseDistanceFilter> tmp = vtkSmartPointer<vtkInverseDistanceFilter>::New();
			tmp->SetPowerValue(inversePower);
			InterpolationFilter = tmp;
		}
		else
		{
			vtkSmartPointer<vtkInverseDistanceFilterCuda> tmp = vtkSmartPointer<vtkInverseDistanceFilterCuda>::New();
			tmp->SetPowerValue(inversePower);
			InterpolationFilter = tmp;
		}
	}
	else if (interpolater == USE_KRIGING_FILTER)
	{
		printf("Use kriging, ");
		if (krigingStep != 0)
			printf("DistStep:%f", krigingStep);
		if (computerunit==USE_CPU_COMPUTE)
		{
			vtkSmartPointer<vtkKrigingFilter> tmp = vtkSmartPointer<vtkKrigingFilter>::New();
			if (krigingStep > 0)
			{
				tmp->SetDistStep(krigingStep);
				tmp->SetStepAutomatic(false);
			}
			InterpolationFilter = tmp;
		}
		else
		{
			vtkSmartPointer<vtkKrigingFilterCuda> tmp = vtkSmartPointer<vtkKrigingFilterCuda>::New();
			if (krigingStep > 0)
			{
				tmp->SetDistStep(krigingStep);
				tmp->SetStepAutomatic(false);
			}
			InterpolationFilter = tmp;
		}
	}
	else if (interpolater == USE_LIMITEINVERSE_FILTER_NUMBER)
	{
		if (computerunit==USE_CPU_COMPUTE)
		{
			printf("Use Limited Number of Points Inverse Distance CPU\n");
			vtkSmartPointer<vtkLimitedInverseDistanceFilter> tmp = vtkSmartPointer<vtkLimitedInverseDistanceFilter>::New();
			printf("limit Point:%d\n", limitnum);
			tmp->SetNumOfLimitPoints(limitnum);
			tmp->SetNullValue(0);
			InterpolationFilter = tmp;
		}
		else
		{
			printf("Use Limited Number of Points Inverse Distance GPU\n");
			vtkSmartPointer<vtkKdtreeLimitedInverseDistanceFilterCuda> tmp = vtkSmartPointer<vtkKdtreeLimitedInverseDistanceFilterCuda>::New();
			printf("limit Point:%d\n", limitnum);
			tmp->SetNumOfLimitPoints(limitnum);
			tmp->SetNullValue(0);
			InterpolationFilter = tmp;
		}
		
	}
	else if (interpolater == USE_LIMITEINVERSE_FILTER_RADIUS)
	{
		if (computerunit==USE_CPU_COMPUTE)
		{
			printf("Use Limited Radius of Points Inverse Distance CPU\n");
			vtkSmartPointer<vtkLimitedInverseDistanceFilter> tmp = vtkSmartPointer<vtkLimitedInverseDistanceFilter>::New();
			printf("limit Radius:%f\n", limitRadius);
			tmp->SetLimitRadius(limitRadius);
			tmp->SetNullValue(0);
			InterpolationFilter = tmp;
		}
		else
		{
			printf("Use Limited Radius of Points Inverse Distance GPU\n");
			vtkSmartPointer<vtkLimitedInverseDistanceFilterCuda> tmp = vtkSmartPointer<vtkLimitedInverseDistanceFilterCuda>::New();
			//vtkSmartPointer<vtkKdtreeLimitedInverseDistanceFilterCuda> tmp = vtkSmartPointer<vtkKdtreeLimitedInverseDistanceFilterCuda>::New();
			printf("limit Radius:%f\n", limitRadius);
			tmp->SetLimitRadius(limitRadius);
			tmp->SetNullValue(0);
			InterpolationFilter = tmp;	
		}
	}

	if (usedefaultbound)
		input_polydata->GetBounds(bound);
	interval[0] = (bound[1]-bound[0])/(seg[0]-1);
	interval[1] = (bound[3]-bound[2])/(seg[1]-1);
	interval[2] = (bound[5]-bound[4])/(seg[2]-1);

	if (computerunit==USE_CPU_COMPUTE)
		printf("cpu\n");
	else
		printf("gpu\n");
	printf("Bounding\n");
	printf("MinX:%lf, MaxX:%lf\n", bound[0], bound[1]);
	printf("MinY:%lf, MaxY:%lf\n", bound[2], bound[3]);
	printf("MinZ:%lf, MaxZ:%lf\n", bound[4], bound[5]);
	printf("Interval x:%lf, y:%lf, z:%lf\n", interval[0], interval[1], interval[2]);
	if (ofilename)
		printf("output : %s\n", ofilename);

	char c;
	if (!autorun)
	{
		printf("Ready(y/n)?");
		scanf("%c", &c);
		if (c!='y' && c!='Y')
			return 0;
	}

	InterpolationFilter->SetInput(input_polydata);
	InterpolationFilter->SetBounds(bound);
	InterpolationFilter->SetInterval(interval);
	InterpolationFilter->SetCudaThreadNum(cudathread);
	InterpolationFilter->SetUseSharedMem(useSharedMem);
	printf("number of thread:%d\n", cudathread);
	if (useSharedMem)
		printf("use shared memory\n");

	bt = BenchTicksGetCurrent();
	InterpolationFilter->Update();
	bt = BenchTicksGetCurrent() - bt;
	printf("Success, compute time = %s\n", BenchTicksToString(bt, true));

	if (noOutput)
		return 0;
	outf = fopen(ofilename, "w");
	if (!outf)
	{
		printf("Output file error!\n");
		return 0;
	}
	fprintf(outf, "%d %d %d\n", seg[0], seg[1], seg[2]);
	outdata = InterpolationFilter->GetOutput();
	outScalars = (vtkDoubleArray*)outdata->GetPointData()->GetScalars();
	double p[3], s;
	for (int i=0; i<outdata->GetNumberOfPoints(); i++)
	{
		outdata->GetPoint(i, p);
		s = outScalars->GetValue(i);
		fprintf(outf, "%f\t%f\t%f\t%f\n", p[0], p[1], p[2], s);
	}
	fclose(outf);
}

// athour: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a, ³¯¥ú«³
// In academic purposes only