
#include "vtkBounds.h"
#include <cstdio>
#include <cctype>
#include <cassert>
#include <auto_link_vtk.hpp>

#include "main.h"
#include "Win32Window.h"
#include "GetDirectXInput.h"
#include "InputState.h"
#include "CoordinateTransform.h"

vtkRenderWindow_Sptr	g_RenderWindow;
vtkRenderer_Sptr	g_Renderer;
vtkAxesActor_Sptr	g_Axes;
vtkCamera_Sptr		g_Camera;
vtkRenderWindowInteractor_Sptr	g_WindowInteractor;
vtkOrientationMarkerWidget_Sptr	g_Axes_widget;
vtkInteractorStyleTrackballCamera_Sptr	g_style;
VectexDatas		g_Vectexs;
VolumeDatas		g_Volumes;
ContourDatas		g_Contours;
PlaneDatas		g_Planes;
CubeAxesDatas		g_CubeAxes;
Win32Window		g_window;
vtkScalarBarActor_Sptr		g_ScalarBarActor;
vtkPolyData_Sptr input_polydata;
vtkImageData_Sptr input_imagedata;
int nx, ny, nz;
double g_max, g_min;
float g_alpha = 1.0;
float g_colormax = 0, g_colormin = 0;
float g_offset = 0;

void printInfo()
{
	printf("V \t: Show Vertex\n");
	printf("CV \t: Clear Vertex\n");
	printf("VR \t: Show Volume Rendering\n");
	printf("CVR \t: Clear Volume Rendering\n");
	printf("S%%f \t: Show Isosurface\n");
	printf("CS \t: Clear Isosurface\n");
	printf("PX or PY or PZ \t: Show Plane on X or Y or Z\n");
	printf("CP \t: Clear Plane\n");
	printf("C%%f %%f \t: All Color table\n");
	printf("A%%f \t: Volume Rendering alpha\n");
	printf("EXIT \t: Exit\n");
}

int main(int argc, char *argv[])
{
	char *ifilename=NULL;
	char ofilename[256] = "a.out";
	FILE *inf = NULL;
	double seg[3] = {50, 50, 50};
	double bound[6];
	double interval[3];
	enum{ USE_NEARESTNEIGHBOR_FILTER, USE_INVERSEDISTANCE_FILTER, USE_KRIGING_FILTER, ONLY_PRINT_INFO, ONLY_PRINT_INSTRUCTION };
	enum{ USE_GPU_COMPUTE, USE_CPU_COMPUTE};
	int interpolater = USE_NEARESTNEIGHBOR_FILTER;
	int computerunit = USE_GPU_COMPUTE;
	bool usedefaultbound = true;
	bool autorun = false;
	input_polydata = vtkSmartNew;
	input_imagedata = vtkSmartNew;
	bool cmderr = false;
	if (argc < 2)
	{
		printf("no input file!\n");
		printf("use vtkDataRender [input]\n");
		printf("-s%%f,%%f,%%f scale\n");
		return 0;
	}
	else if (argc == 3 && memcmp("-v", argv[2], 2)==0 )
	{
		ifilename = argv[1];
		LoadInputData2(ifilename, input_polydata);
		InitVTK();
		AddVertex(input_polydata);
		AddCubeAxes(input_polydata);
		for (;;)
		{
			g_window.HandlePeekMessage();
		}
	}
	else
	{
		ifilename = argv[1];
		float s[3] = {1, 1, 1};
		for (int i=2;i < argc;++i)
		{
			if (memcmp("-s", argv[i], 2)==0)
				sscanf(argv[i], "-s%f,%f,%f", s, s+1, s+2);
		}		
		if (!LoadInputData(ifilename, input_polydata))
		{
			printf("load file error!\n");
			return 0;
		}
		ConvertPolyToImage2(input_polydata, input_imagedata, nx, ny, nz, s);
		printInfo();
		InitVTK();
		MainLoop();
	}
}

void MainLoop()
{
	DirectXIS::instance().InputInit(g_window.GetHandle(), g_window.GetInstance());
	InputStateS::instance().SetDirectXInput(&DirectXIS::instance());
	static char input[1024] = {0};
	static int index = 0;
	float viewup[3] = {1,0,0};
	for (;;)
	{
		g_window.HandlePeekMessage();
		InputStateS::instance().GetInput();
		if (InputStateS::instance().isKeyDown(KEY_RETURN)||
			InputStateS::instance().isKeyDown(KEY_NUMPADENTER))
		{
			index = 0;
			
			if (memcmp(input, "CVR", 3)==0)
			{
				ClearVolume();
			}
			else if (memcmp(input, "EXIT", 4)==0)
			{
				exit(0);
			}
			else if (memcmp(input, "CV", 2)==0)
			{
				ClearVertex();
			}
			else if (memcmp(input, "CV", 2)==0)
			{
				ClearVertex();
			}
			else if (memcmp(input, "CS", 2)==0)
			{
				ClearContour();
			}
			else if (memcmp(input, "CA", 2)==0)
			{
				ClearCubeAxes();
			}
			else if (memcmp(input, "VR", 2)==0)
			{
				AddVolume(input_imagedata);
			}
			else if (memcmp(input, "AC", 2)==0)
			{
				AddCubeAxes(input_imagedata);
			}
			else if (memcmp(input, "PX", 2)==0)
			{
				AddPlane(input_imagedata, 1);
			}
			else if (memcmp(input, "PY", 2)==0)
			{
				AddPlane(input_imagedata, 2);
			}
			else if (memcmp(input, "PZ", 2)==0)
			{
				AddPlane(input_imagedata, 3);
			}
			else if (memcmp(input, "CP", 2)==0)
			{
				ClearPlane();
			}
			else if (memcmp(input, "V", 1)==0)
			{
				AddVertex(input_polydata);
			}
			else if (memcmp(input, "S", 1)==0)
			{
				double v;
				sscanf(input, "S%lf", &v);
				AddContour(input_imagedata, v);
			}
			else if (memcmp(input, "-M", 2)==0)
			{
				sscanf(input, "-M%f", &g_offset);
			}
			else if (memcmp(input, "N1", 2)==0)
			{
				vtkBounds bounding;
				double tmp[6];
				input_polydata->GetBounds(tmp);
				bounding.SetBounds(tmp);
				double pos[3];
				g_Camera->SetViewUp(viewup[0], viewup[1], viewup[2]);
				g_Camera->GetPosition(pos);
				g_Camera->SetPosition(bounding.Xmid()+60000, bounding.Ymid()-60000, bounding.Zmid());
				g_Camera->SetFocalPoint(bounding.Xmid(), bounding.Ymid(), bounding.Zmid());
				g_Camera->UpdateViewport(g_Renderer);
			}
			else if (memcmp(input, "N2", 2)==0)
			{
				vtkBounds bounding;
				double tmp[6];
				input_polydata->GetBounds(tmp);
				bounding.SetBounds(tmp);
				double pos[3];
				g_Camera->SetViewUp(viewup[0], viewup[1], viewup[2]);
				g_Camera->GetPosition(pos);
				g_Camera->SetPosition(bounding.Xmid()-60000, bounding.Ymid()-60000, bounding.Zmid());
				g_Camera->SetFocalPoint(bounding.Xmid(), bounding.Ymid(), bounding.Zmid());
				g_Camera->UpdateViewport(g_Renderer);
			}
			else if (memcmp(input, "N3", 2)==0)
			{
				vtkBounds bounding;
				double tmp[6];
				input_polydata->GetBounds(tmp);
				bounding.SetBounds(tmp);
				double pos[3];
				g_Camera->SetViewUp(viewup[0], viewup[1], viewup[2]);
				g_Camera->GetPosition(pos);
				g_Camera->SetPosition(bounding.Xmid()-60000, bounding.Ymid()+60000, bounding.Zmid());
				g_Camera->SetFocalPoint(bounding.Xmid(), bounding.Ymid(), bounding.Zmid());
				g_Camera->UpdateViewport(g_Renderer);
			}
			else if (memcmp(input, "N4", 2)==0)
			{
				vtkBounds bounding;
				double tmp[6];
				input_polydata->GetBounds(tmp);
				bounding.SetBounds(tmp);
				double pos[3];
				g_Camera->SetViewUp(viewup[0], viewup[1], viewup[2]);
				g_Camera->GetPosition(pos);
				g_Camera->SetPosition(bounding.Xmid()+60000, bounding.Ymid()+60000, bounding.Zmid());
				g_Camera->SetFocalPoint(bounding.Xmid(), bounding.Ymid(), bounding.Zmid());
				g_Camera->UpdateViewport(g_Renderer);
			}
			else if (memcmp(input, "N5", 2)==0)
			{
				vtkBounds bounding;
				double tmp[6];
				input_polydata->GetBounds(tmp);
				bounding.SetBounds(tmp);
				double pos[3];
				g_Camera->SetViewUp(viewup[0], viewup[1], viewup[2]);
				g_Camera->GetPosition(pos);
				g_Camera->SetPosition(bounding.Xmid(), bounding.Ymid()+bounding.Ylen()+g_offset, bounding.Zmid()+bounding.Zlen()+g_offset);
				g_Camera->SetFocalPoint(bounding.Xmid(), bounding.Ymid(), bounding.Zmid());
				g_Camera->UpdateViewport(g_Renderer);
			}
			else if (memcmp(input, "N6", 2)==0)
			{
				vtkBounds bounding;
				double tmp[6];
				input_polydata->GetBounds(tmp);
				bounding.SetBounds(tmp);
				double pos[3];
				g_Camera->SetViewUp(viewup[0], viewup[1], viewup[2]);
				g_Camera->GetPosition(pos);
				g_Camera->SetPosition(bounding.Xmid(), bounding.Ymid()+bounding.Ylen()+g_offset, bounding.Zmid()-bounding.Zlen()-g_offset);
				g_Camera->SetFocalPoint(bounding.Xmid(), bounding.Ymid(), bounding.Zmid());
				g_Camera->UpdateViewport(g_Renderer);
			}
			else if (memcmp(input, "N7", 2)==0)
			{
				vtkBounds bounding;
				double tmp[6];
				input_polydata->GetBounds(tmp);
				bounding.SetBounds(tmp);
				double pos[3];
				g_Camera->SetViewUp(viewup[0], viewup[1], viewup[2]);
				g_Camera->GetPosition(pos);
				g_Camera->SetPosition(bounding.Xmid(), bounding.Ymid()-bounding.Ylen()-g_offset, bounding.Zmid()+bounding.Zlen()+g_offset);
				g_Camera->SetFocalPoint(bounding.Xmid(), bounding.Ymid(), bounding.Zmid());
				g_Camera->UpdateViewport(g_Renderer);
			}
			else if (memcmp(input, "N8", 2)==0)
			{
				vtkBounds bounding;
				double tmp[6];
				input_polydata->GetBounds(tmp);
				bounding.SetBounds(tmp);
				double pos[3];
				g_Camera->SetViewUp(viewup[0], viewup[1], viewup[2]);
				g_Camera->GetPosition(pos);
				g_Camera->SetPosition(bounding.Xmid(), bounding.Ymid()-bounding.Ylen()-g_offset, bounding.Zmid()-bounding.Zlen()-g_offset);
				g_Camera->SetFocalPoint(bounding.Xmid(), bounding.Ymid(), bounding.Zmid());
				g_Camera->UpdateViewport(g_Renderer);
			}
			else if (memcmp(input, "NX", 2)==0)
			{
				float nn[3];
				vtkBounds bounding;
				double tmp[6];
				sscanf(input, "NX%fX%fX%f", nn, nn+1, nn+2);
				input_polydata->GetBounds(tmp);
				bounding.SetBounds(tmp);
				double pos[3];
				g_Camera->SetViewUp(viewup[0], viewup[1], viewup[2]);
				g_Camera->GetPosition(pos);
				g_Camera->SetPosition(bounding.Xmid()+nn[0], bounding.Ymid()+nn[1], bounding.Zmid()+nn[2]);
				g_Camera->SetFocalPoint(bounding.Xmid(), bounding.Ymid(), bounding.Zmid());
				g_Camera->UpdateViewport(g_Renderer);
			}
			else if (memcmp(input, "NN", 2)==0)
			{
				vtkBounds bounding;
				double tmp[6];
				input_polydata->GetBounds(tmp);
				bounding.SetBounds(tmp);
				double pos[3];
				g_Camera->SetViewUp(viewup[0], viewup[1], viewup[2]);
				g_Camera->GetPosition(pos);
				g_Camera->SetPosition(bounding.Xmid() , bounding.Ymid()-g_offset, bounding.Zmid());
				g_Camera->SetFocalPoint(bounding.Xmid(), bounding.Ymid(), bounding.Zmid());
				g_Camera->UpdateViewport(g_Renderer);
			}
			else if (memcmp(input, "UP", 2)==0)
			{
				sscanf(input, "UP%fX%fX%f", viewup, viewup+1, viewup+2);
				g_Camera->SetViewUp(viewup[0], viewup[1], viewup[2]);
				g_Camera->UpdateViewport(g_Renderer);
			}
			else if (memcmp("-A", input, 2)==0)
				sscanf(input, "-A%f", &g_alpha);
			else if (memcmp("-C", input, 2)==0)
			{
				sscanf(input, "-C%f %f", &g_colormax, &g_colormin);
				if (g_colormax>g_colormin)
				{
					g_max = g_colormax;
					g_min = g_colormin;
				}
			}
			memset(input, 0, sizeof(input));
			printf(" .\n");
		}
		else
		{
			if (InputStateS::instance().isKeyDown(KEY_A)){ input[index++] = 65; printf("%c", 65); }
			if (InputStateS::instance().isKeyDown(KEY_B)){ input[index++] = 66; printf("%c", 66); }
			if (InputStateS::instance().isKeyDown(KEY_C)){ input[index++] = 67; printf("%c", 67); }
			if (InputStateS::instance().isKeyDown(KEY_D)){ input[index++] = 68; printf("%c", 68); }
			if (InputStateS::instance().isKeyDown(KEY_E)){ input[index++] = 69; printf("%c", 69); }
			if (InputStateS::instance().isKeyDown(KEY_F)){ input[index++] = 70; printf("%c", 70); }
			if (InputStateS::instance().isKeyDown(KEY_G)){ input[index++] = 71; printf("%c", 71); }
			if (InputStateS::instance().isKeyDown(KEY_H)){ input[index++] = 72; printf("%c", 72); }
			if (InputStateS::instance().isKeyDown(KEY_I)){ input[index++] = 73; printf("%c", 73); }
			if (InputStateS::instance().isKeyDown(KEY_J)){ input[index++] = 74; printf("%c", 74); }
			if (InputStateS::instance().isKeyDown(KEY_K)){ input[index++] = 75; printf("%c", 75); }
			if (InputStateS::instance().isKeyDown(KEY_L)){ input[index++] = 76; printf("%c", 76); }
			if (InputStateS::instance().isKeyDown(KEY_M)){ input[index++] = 77; printf("%c", 77); }
			if (InputStateS::instance().isKeyDown(KEY_N)){ input[index++] = 78; printf("%c", 78); }
			if (InputStateS::instance().isKeyDown(KEY_O)){ input[index++] = 79; printf("%c", 79); }
			if (InputStateS::instance().isKeyDown(KEY_P)){ input[index++] = 80; printf("%c", 80); }
			if (InputStateS::instance().isKeyDown(KEY_Q)){ input[index++] = 81; printf("%c", 81); }
			if (InputStateS::instance().isKeyDown(KEY_R)){ input[index++] = 82; printf("%c", 82); }
			if (InputStateS::instance().isKeyDown(KEY_S)){ input[index++] = 83; printf("%c", 83); }
			if (InputStateS::instance().isKeyDown(KEY_T)){ input[index++] = 84; printf("%c", 84); }
			if (InputStateS::instance().isKeyDown(KEY_U)){ input[index++] = 85; printf("%c", 85); }
			if (InputStateS::instance().isKeyDown(KEY_V)){ input[index++] = 86; printf("%c", 86); }
			if (InputStateS::instance().isKeyDown(KEY_W)){ input[index++] = 87; printf("%c", 87); }
			if (InputStateS::instance().isKeyDown(KEY_X)){ input[index++] = 88; printf("%c", 88); }
			if (InputStateS::instance().isKeyDown(KEY_Y)){ input[index++] = 89; printf("%c", 89); }
			if (InputStateS::instance().isKeyDown(KEY_Z)){ input[index++] = 90; printf("%c", 90); }

			if (InputStateS::instance().isKeyDown(KEY_SUBTRACT)){ input[index++] = 45; printf("%c", 45); }
			if (InputStateS::instance().isKeyDown(KEY_SPACE)){ input[index++] = 32; printf("%c", 32); }
			if (InputStateS::instance().isKeyDown(KEY_NUMPAD0)){ input[index++] = 48; printf("%c", 48); }
			if (InputStateS::instance().isKeyDown(KEY_NUMPAD1)){ input[index++] = 49; printf("%c", 49); }
			if (InputStateS::instance().isKeyDown(KEY_NUMPAD2)){ input[index++] = 50; printf("%c", 50); }
			if (InputStateS::instance().isKeyDown(KEY_NUMPAD3)){ input[index++] = 51; printf("%c", 51); }
			if (InputStateS::instance().isKeyDown(KEY_NUMPAD4)){ input[index++] = 52; printf("%c", 52); }
			if (InputStateS::instance().isKeyDown(KEY_NUMPAD5)){ input[index++] = 53; printf("%c", 53); }
			if (InputStateS::instance().isKeyDown(KEY_NUMPAD6)){ input[index++] = 54; printf("%c", 54); }
			if (InputStateS::instance().isKeyDown(KEY_NUMPAD7)){ input[index++] = 55; printf("%c", 55); }
			if (InputStateS::instance().isKeyDown(KEY_NUMPAD8)){ input[index++] = 56; printf("%c", 56); }
			if (InputStateS::instance().isKeyDown(KEY_NUMPAD9)){ input[index++] = 57; printf("%c", 57); }
			if (InputStateS::instance().isKeyDown(KEY_DECIMAL)){ input[index++] = 46; printf("%c", 46); }
			if (InputStateS::instance().isKeyDown(KEY_PERIOD)){ input[index++] = 46; printf("%c", 46); }
		}
		Sleep(1);
	}
}

bool LoadInputData(char* filename, vtkPolyData* polydata)
{
	g_max=-VTK_FLOAT_MAX;
	g_min=VTK_FLOAT_MAX;
	vtkPoints_Sptr input_points = vtkSmartNew;
	vtkDoubleArray_Sptr input_scalars = vtkSmartNew;
	double x,y,z,s;
	char c = 0;
	FILE* inf = fopen(filename, "r");
	if (!inf)
	{
		printf("Input file error!\n");
		return false;
	}
	fscanf(inf, "%d %d %d", &nx, &ny, &nz);
	while(true)
	{
		if( fscanf(inf, "%lf", &x) == EOF)
			break;
		fscanf(inf, "%lf%lf%lf", &y, &z, &s);
		input_points->InsertNextPoint(x, y, z);
		input_scalars->InsertNextTuple1(s);
		if (s>g_max) g_max=s;
		if (s<g_min) g_min=s;
	}
	if (g_colormax>g_colormin)
	{
		g_max = g_colormax;
		g_min = g_colormin;
	}
	polydata->SetPoints(input_points);
	polydata->GetPointData()->SetScalars(input_scalars);
	fclose(inf);
	
	return true;
}

bool ConvertPolyToImage2(const vtkPolyData_Sptr poly, vtkImageData_Sptr image, int nx, int ny, int nz, float* s)
{
	vtkPoints_Sptr	points = poly->GetPoints();
	poly->GetPointData()->GetScalars()->SetName("value");

	vtkIdType count = poly->GetPointData()->GetScalars()->GetNumberOfTuples();
	bool	isGrided = nx * ny * nz == count;
	vtkBounds bounding;
	bounding.SetBounds(poly->GetBounds());

	double	orgin[3];
	poly->GetPoint(0, orgin);
	if (isGrided)
	{
		image->SetSpacing
			(
			bounding.Xlen() / (nx - 1)*s[0],
			bounding.Ylen() / (ny - 1)*s[1],
			bounding.Zlen() / (nz - 1)*s[2]
			);
		image->SetDimensions(nx, ny, nz);
		image->GetPointData()->SetScalars(poly->GetPointData()->GetScalars());
		image->SetOrigin(bounding[0], bounding[2], bounding[4]);
		image->Update();
	}
	else
	{
		//assert(0 && "error: nx* ny* nz == count ");
		return 1;
	}
}

bool ConvertPolyToImage(const vtkPolyData_Sptr poly, vtkImageData_Sptr image, int nx, int ny, int nz)
{
	vtkPoints_Sptr	points = poly->GetPoints();
	poly->GetPointData()->GetScalars()->SetName("value");

	vtkIdType count = poly->GetPointData()->GetScalars()->GetNumberOfTuples();
	bool	isGrided = nx * ny * nz == count;
	vtkBounds bounding;
	bounding.SetBounds(poly->GetBounds());

	double	orgin[3];
	poly->GetPoint(0, orgin);
	if (isGrided)
	{
		image->SetSpacing
			(
			bounding.Xlen() / (nx - 1),
			bounding.Ylen() / (ny - 1),
			bounding.Zlen() / (nz - 1)
			);
		image->SetDimensions(nx, ny, nz);
		image->GetPointData()->SetScalars(poly->GetPointData()->GetScalars());
		image->SetOrigin(bounding[0], bounding[2], bounding[4]);
		image->Update();
	}
	else
	{
		//assert(0 && "error: nx* ny* nz == count ");
		return 1;
	}
}

bool start = false;
static LRESULT CALLBACK MyProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	if (start)
		g_RenderWindow->Render();
	

	switch (message)
	{
	case WM_DESTROY:
		PostQuitMessage(WM_QUIT);
		break;
	case WM_SIZE:
		{
			RECT window_rect = g_window.GetRect();
			g_RenderWindow->SetSize(window_rect.right-window_rect.left, window_rect.bottom-window_rect.top);
		}
		break;
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}
	return 0;
}

void InitVTK()
{
	g_RenderWindow = vtkSmartNew;
	g_Renderer = vtkSmartNew;
	g_WindowInteractor = vtkOnlyNew;
	g_Camera = vtkSmartNew;
	g_Axes_widget = vtkSmartNew;
	g_Axes = vtkSmartNew;
	g_style = vtkSmartNew;
	g_RenderWindow->AddRenderer(g_Renderer);
	g_WindowInteractor->SetRenderWindow(g_RenderWindow);
	g_WindowInteractor->SetInteractorStyle( g_style );
	g_WindowInteractor->EnableRenderOn();
	g_Renderer->SetActiveCamera(g_Camera);
	g_Renderer->SetBackground(.0, .0, .0);
	g_Axes_widget->SetOutlineColor( 0.8300, 0.6700, 0.5300 );
	g_Axes_widget->SetOrientationMarker( g_Axes );
	g_Axes_widget->SetInteractor( g_WindowInteractor );
	g_Axes_widget->On();
	g_window.ToCreateWindow(0,0,1920,1080, L"vtkDataRender", MyProc);
	g_window.ToShow();
	g_window.ToMoveCenter();
	g_RenderWindow->SetParentId(g_window.GetHandle());
	g_RenderWindow->Render();
	g_RenderWindow->SetSize(1920,1080);
	vtkBounds bounding;
	bounding.SetBounds(input_polydata->GetBounds());
	g_Camera->SetPosition(0, 0, (bounding.Xmid() + bounding.Ymid() + bounding.Zmid()) / 2);
	g_Camera->SetFocalPoint(bounding.Xmid(), bounding.Ymid(), bounding.Zmid());
	start = true;
}

void AddVertex(vtkPolyData_Sptr poly)
{
	VectexData vd;
	vtkUnsignedCharArray_Sptr	colors = vtkSmartNew;
	colors->SetNumberOfComponents(3);
	colors->SetName("Colors");
	vtkDoubleArray* scalars = (vtkDoubleArray*)poly->GetPointData()->GetScalars();
	vtkColorTransferFunction_Sptr	colorTransferFunction = vtkSmartNew;
	
	double step = (g_max-g_min)/6;
	colorTransferFunction->AddRGBPoint(g_min+step*6, 1.0 / 2, 0.0, 0.0);
	colorTransferFunction->AddRGBPoint(g_min+step*5, 1.0 / 2, 165 / 255 / 2.0, 0.0);
	colorTransferFunction->AddRGBPoint(g_min+step*4, 1.0 / 2, 1.0 / 2, 0.0);
	colorTransferFunction->AddRGBPoint(g_min+step*3, 0.0, 1.0 / 2, 0.0);
	colorTransferFunction->AddRGBPoint(g_min+step*2, 0.0, 0.5 / 2, 1.0 / 2);
	colorTransferFunction->AddRGBPoint(g_min+step*1.6, 0.0, 0.0, 1.0 / 2);
	colorTransferFunction->AddRGBPoint(g_min+step*1.4, 139 / 255.0 / 2, 0.0, 1.0 / 2);
	colorTransferFunction->AddRGBPoint(g_min+step*1.2, 0.0, 0.5 / 2, 1.0 / 2);
	colorTransferFunction->AddRGBPoint(g_min+step*1, 0.0, 0.0, 1.0 / 2);
	colorTransferFunction->AddRGBPoint(g_min+step*0, 139 / 255.0 / 2, 0.0, 1.0 / 2);

	double	p[6];
	poly->GetBounds(p);

	int	point_size = poly->GetNumberOfPoints();
	for (int i = 0; i < point_size; i++)
	{
		double	dcolor[3];
		colorTransferFunction->GetColor(scalars->GetValue(i), dcolor);
		unsigned char	color[3];
		for (unsigned int j = 0; j < 3; j++)
		{
			color[j] = static_cast<unsigned char>(255.0 * dcolor[j]);
		}

		colors->InsertNextTupleValue(color);
	}

	vtkVertexGlyphFilter_Sptr	vertexGlyphFilter = vtkSmartNew;
	vtkPolyData_Sptr		colorpolydata = vtkSmartNew;
	colorpolydata->SetPoints(poly->GetPoints());
	colorpolydata->GetPointData()->SetScalars(colors);
	vertexGlyphFilter->SetInput(colorpolydata);
	vd.m_polydataMapper->SetInputConnection(vertexGlyphFilter->GetOutputPort());
	vd.m_polydataMapper->SetLookupTable(colorTransferFunction);
	vd.m_actor->SetMapper(vd.m_polydataMapper);
	vd.m_actor->GetProperty()->SetPointSize(4);
	g_Renderer->AddActor(vd.m_actor);
	g_Vectexs.push_back(vd);

	g_Renderer->RemoveActor(g_ScalarBarActor);
	g_ScalarBarActor = vtkSmartNew;
	g_ScalarBarActor->SetLookupTable(colorTransferFunction);
	g_ScalarBarActor->SetNumberOfLabels(5);
	g_ScalarBarActor->SetMaximumWidthInPixels(150);
	g_ScalarBarActor->SetMaximumHeightInPixels(400);

	g_Renderer->AddActor2D(g_ScalarBarActor);
}

void AddCubeAxes(vtkPolyData_Sptr poly)
{
	CubeAxesData ad;
	double boundingTWD[6];
	double boundingWGS[6];
	poly->GetBounds(boundingTWD);
	poly->GetBounds(boundingWGS);
	//xmin zmin
	CoordinateTransform::TWD97_To_lonlat(boundingTWD[0], boundingTWD[2], boundingWGS+0, boundingWGS+2);
	CoordinateTransform::TWD97_To_lonlat(boundingTWD[1], boundingTWD[3], boundingWGS+1, boundingWGS+3);
	ad.m_axes->SetBounds(poly->GetBounds());
	ad.m_axes->SetCamera(g_Renderer->GetActiveCamera());
	ad.m_axes->SetBounds(boundingTWD);
	ad.m_axes->SetXAxisRange(boundingWGS[0], boundingWGS[1]);
	ad.m_axes->SetYAxisRange(boundingWGS[2], boundingWGS[3]);
	ad.m_axes->SetZAxisRange(boundingWGS[4], boundingWGS[5]);
	ad.m_axes->SetXTitle("");
	ad.m_axes->SetYTitle("");
	ad.m_axes->SetZTitle("");
	ad.m_axes->SetXLabelFormat("%-#f");
	ad.m_axes->SetYLabelFormat("%-#f");
	ad.m_axes->SetZLabelFormat("%-#f");
	ad.m_axes->SetLabelScaling(false,0,0,0);
	ad.m_axes->SetTickLocationToOutside();
	g_Renderer->AddActor(ad.m_axes);
	g_CubeAxes.push_back(ad);
}

void AddCubeAxes(vtkImageData_Sptr image)
{
	CubeAxesData ad;
	double boundingTWD[6];
	double boundingWGS[6];
	image->GetBounds(boundingTWD);
	//xmin zmin
	CoordinateTransform::TWD97_To_lonlat(boundingTWD[0], boundingTWD[4], boundingWGS+0, boundingWGS+4);
	CoordinateTransform::TWD97_To_lonlat(boundingTWD[1], boundingTWD[5], boundingWGS+1, boundingWGS+5);
	ad.m_axes->SetBounds(image->GetBounds());
	ad.m_axes->SetCamera(g_Renderer->GetActiveCamera());
	ad.m_axes->SetBounds(boundingTWD);
	ad.m_axes->SetXAxisRange(boundingWGS[0], boundingWGS[1]);
	ad.m_axes->SetYAxisRange(boundingWGS[2], boundingWGS[3]);
	ad.m_axes->SetZAxisRange(boundingWGS[4], boundingWGS[5]);
	ad.m_axes->SetXTitle("E,lon");
	ad.m_axes->SetYTitle("Height");
	ad.m_axes->SetZTitle("N,lot");
	ad.m_axes->SetXLabelFormat("%-#f");
	ad.m_axes->SetYLabelFormat("%-#f");
	ad.m_axes->SetZLabelFormat("%-#f");
	ad.m_axes->SetLabelScaling(false,0,0,0);
	ad.m_axes->SetFlyModeToStaticEdges();
	g_Renderer->AddActor(ad.m_axes);
	g_CubeAxes.push_back(ad);
}

void AddVolume(vtkImageData_Sptr image)
{
	VolumeData vd;
	vtkPiecewiseFunction_Sptr	compositeOpacity = vtkSmartNew;
	vtkColorTransferFunction_Sptr	colorTransferFunction = vtkSmartNew;
	vtkSmartVolumeMapper_Sptr	volumeMapper = vtkOnlyNew;
	vtkVolumeProperty_Sptr		volumeProperty = vtkSmartNew;
	
	volumeMapper->SetBlendModeToComposite();		// composite first
	volumeMapper->SetRequestedRenderMode(vtkSmartVolumeMapper::GPURenderMode);
	volumeMapper->SetInputConnection(image->GetProducerPort());
	double step = (g_max-g_min)/6;
	compositeOpacity->AddPoint(g_min+step*0, 0.001*g_alpha);
	compositeOpacity->AddPoint(g_min+step*1, 0.002*g_alpha);
	compositeOpacity->AddPoint(g_min+step*2, 0.003*g_alpha);
	compositeOpacity->AddPoint(g_min+step*3, 0.004*g_alpha);
	compositeOpacity->AddPoint(g_min+step*4, 0.004*g_alpha);
	compositeOpacity->AddPoint(g_min+step*5, 0.004*g_alpha);
	compositeOpacity->AddPoint(g_min+step*6, 0.02*g_alpha);
	volumeProperty->SetScalarOpacity(compositeOpacity);	// composite first.
	volumeProperty->SetDiffuse(0.2);
	volumeProperty->ShadeOff();
	volumeProperty->SetDisableGradientOpacity(1);
	volumeProperty->SetInterpolationType(VTK_LINEAR_INTERPOLATION);
	colorTransferFunction->AddRGBPoint(g_min+step*6, 1.0 / 2, 0.0, 0.0);
	colorTransferFunction->AddRGBPoint(g_min+step*5, 1.0 / 2, 165 / 255 / 2.0, 0.0);
	colorTransferFunction->AddRGBPoint(g_min+step*4, 1.0 / 2, 1.0 / 2, 0.0);
	colorTransferFunction->AddRGBPoint(g_min+step*3, 0.0, 1.0 / 2, 0.0);
	colorTransferFunction->AddRGBPoint(g_min+step*2, 0.0, 0.5 / 2, 1.0 / 2);
	colorTransferFunction->AddRGBPoint(g_min+step*1, 0.0, 0.0, 1.0 / 2);
	colorTransferFunction->AddRGBPoint(g_min+step*0, 139 / 255.0 / 2, 0.0, 1.0 / 2);
	volumeProperty->SetColor(colorTransferFunction);
	vd.m_volume = vtkSmartNew;
	vd.m_volume->SetMapper(volumeMapper);
	vd.m_volume->SetProperty(volumeProperty);
	g_Renderer->RemoveActor(g_ScalarBarActor);
	g_ScalarBarActor = vtkSmartNew;
	g_ScalarBarActor->SetLookupTable(colorTransferFunction);
	g_ScalarBarActor->SetNumberOfLabels(5);
	g_ScalarBarActor->SetMaximumWidthInPixels(150);
	g_ScalarBarActor->SetMaximumHeightInPixels(400);

	
	g_Renderer->AddActor2D(g_ScalarBarActor);
	g_Renderer->AddViewProp(vd.m_volume);
	g_Volumes.push_back(vd);
}

bool LoadInputData2(char* filename, vtkPolyData* polydata)
{
	g_max=-VTK_FLOAT_MAX;
	g_min=VTK_FLOAT_MAX;
	vtkPoints_Sptr input_points = vtkSmartNew;
	vtkDoubleArray_Sptr input_scalars = vtkSmartNew;
	double x,y,z,s;
	char c = 0;
	FILE* inf = fopen(filename, "r");
	if (!inf)
	{
		printf("Input file error!\n");
		return false;
	}
	while(true)
	{
		if( fscanf(inf, "%lf", &x) == EOF)
			break;
		fscanf(inf, "%lf%lf%lf", &y, &z, &s);
		input_points->InsertNextPoint(x, y, z);
		input_scalars->InsertNextTuple1(s);
		if (s>g_max) g_max=s;
		if (s<g_min) g_min=s;
	}
	polydata->SetPoints(input_points);
	polydata->GetPointData()->SetScalars(input_scalars);
	fclose(inf);
	return true;
}

void ClearContour()
{
	for (size_t i=0;i < g_Contours.size();++i)
	{
		g_Renderer->RemoveActor(g_Contours[i].m_actor);
	}
	g_Contours.clear();
}
void ClearVertex()
{
	for (size_t i=0;i < g_Vectexs.size();++i)
	{
		g_Renderer->RemoveActor(g_Vectexs[i].m_actor);
	}
	g_Vectexs.clear();
}
void ClearCubeAxes() 
{
	for (size_t i=0;i < g_CubeAxes.size();++i)
	{
		g_Renderer->RemoveViewProp(g_CubeAxes[i].m_axes);
	}
	g_CubeAxes.clear();
}
void ClearVolume() 
{
	for (size_t i=0;i < g_Volumes.size();++i)
	{
		g_Renderer->RemoveViewProp(g_Volumes[i].m_volume);
	}
	g_Volumes.clear();
}
void ClearPlane() 
{
	for (size_t i=0;i < g_Planes.size();++i)
	{
		g_Planes[i].m_ImagePlane->Off();
	}
	g_Planes.clear();
}

void AddContour( vtkImageData_Sptr image, double v )
{
	ContourData cd;
	vtkLookupTable_Sptr	lut = vtkSmartNew;
	lut->SetTableRange(g_min, g_max);
	lut->Build();
	cd.m_ContourFilter = vtkSmartNew;
	cd.m_polydataMapper = vtkSmartNew;
	cd.m_actor = vtkSmartNew;
	cd.m_ContourFilter->SetInput(image);
	cd.m_ContourFilter->SetValue(0, v);
	cd.m_polydataMapper->SetInputConnection(cd.m_ContourFilter->GetOutputPort());

	vtkColorTransferFunction_Sptr	colorTransferFunction = vtkSmartNew;
	double step = (g_max-g_min)/6;
	colorTransferFunction->AddRGBPoint(g_min+step*6, 1.0 / 2, 0.0, 0.0);
	colorTransferFunction->AddRGBPoint(g_min+step*5, 1.0 / 2, 165 / 255 / 2.0, 0.0);
	colorTransferFunction->AddRGBPoint(g_min+step*4, 1.0 / 2, 1.0 / 2, 0.0);
	colorTransferFunction->AddRGBPoint(g_min+step*3, 0.0, 1.0 / 2, 0.0);
	colorTransferFunction->AddRGBPoint(g_min+step*2, 0.0, 0.5 / 2, 1.0 / 2);
	colorTransferFunction->AddRGBPoint(g_min+step*1, 0.0, 0.0, 1.0 / 2);
	colorTransferFunction->AddRGBPoint(g_min+step*0, 139 / 255.0 / 2, 0.0, 1.0 / 2);
	cd.m_polydataMapper->SetLookupTable(colorTransferFunction);

	vtkPolyDataNormals_Sptr sGridPolyDataNormal = vtkSmartNew;
	sGridPolyDataNormal->SetInput(cd.m_ContourFilter->GetOutput());
	sGridPolyDataNormal->Update();
	cd.m_polydataMapper->SetInput(sGridPolyDataNormal->GetOutput());
	cd.m_polydataMapper->Update();
	cd.m_actor->SetMapper(cd.m_polydataMapper);
	g_Renderer->AddActor(cd.m_actor);
	g_Contours.push_back(cd);
}


void AddPlane( vtkImageData_Sptr image, int type )
{
	PlaneData pd;
	pd.m_ImagePlane = vtkSmartNew;
	pd.m_ImagePlane->SetLeftButtonAction(vtkImagePlaneWidget::VTK_CURSOR_ACTION);
	pd.m_ImagePlane->SetRightButtonAction(vtkImagePlaneWidget::VTK_CURSOR_ACTION);
	pd.m_ImagePlane->SetInteractor(g_WindowInteractor);
	pd.m_ImagePlane->RestrictPlaneToVolumeOn();
	pd.m_ImagePlane->SetInput(image);
	switch(type)
	{
	case 1: pd.m_ImagePlane->SetPlaneOrientationToXAxes();
		break;
	case 2: pd.m_ImagePlane->SetPlaneOrientationToYAxes();
		break;
	case 3: pd.m_ImagePlane->SetPlaneOrientationToZAxes();
		break;
	}
	vtkColorTransferFunction_Sptr	colorTransferFunction = vtkSmartNew;
	double step = (g_max-g_min)/6;
	colorTransferFunction->AddRGBPoint(g_min+step*6, 1.0 / 2, 0.0, 0.0);
	colorTransferFunction->AddRGBPoint(g_min+step*5, 1.0 / 2, 165 / 255 / 2.0, 0.0);
	colorTransferFunction->AddRGBPoint(g_min+step*4, 1.0 / 2, 1.0 / 2, 0.0);
	colorTransferFunction->AddRGBPoint(g_min+step*3, 0.0, 1.0 / 2, 0.0);
	colorTransferFunction->AddRGBPoint(g_min+step*2, 0.0, 0.5 / 2, 1.0 / 2);
	colorTransferFunction->AddRGBPoint(g_min+step*1, 0.0, 0.0, 1.0 / 2);
	colorTransferFunction->AddRGBPoint(g_min+step*0, 139 / 255.0 / 2, 0.0, 1.0 / 2);
	pd.m_ImagePlane->GetColorMap()->SetLookupTable(colorTransferFunction);
	pd.m_ImagePlane->On();
	g_Planes.push_back(pd);
}
