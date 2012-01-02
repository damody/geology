
#include "StdWxVtk.h"
#include "SolidCtrl.h"
#include "SolidDoc.h"
#include "SolidView.h"
#include "SEffect.h"
#include "Interpolation/vtkInverseDistanceFilter.h"
#include "Interpolation/vtkNearestNeighborFilter.h"
#include <vtkCellArray.h>
#include <vtkDataSetMapper.h>
#include <vtkTriangleStrip.h>
#include <vtkPlaneSource.h>
#include <vtkStripper.h>
#include <vtkPoints.h>
#include <vtkLabeledDataMapper.h>

VTKSMART_PTR(vtkNearestNeighborFilter)
VTKSMART_PTR(vtkInverseDistanceFilter)
SolidDoc_Sptr SolidCtrl::NewDoc()	// 新資料集
{
	SolidDoc_Sptr	tmpPtr(new SolidDoc(m_bounds));
	assert(m_polydata->GetNumberOfPoints() != 0);
	tmpPtr->SetPolyData(m_polydata);
	if (m_imagedata.GetPointer() != 0)
	{
		tmpPtr->SetImageData(m_imagedata);
	}

	if (m_sf3d)
		tmpPtr->m_histogram = Histogramd(m_sf3d->begin(), m_sf3d->size());
	else if (!m_SolidDocPtrs.empty())
		tmpPtr->m_histogram = m_SolidDocPtrs.back()->m_histogram;
	tmpPtr->m_ParentCtrl = this;
	m_SolidDocPtrs.push_back(tmpPtr);
	return tmpPtr;
}

SolidView_Sptr SolidCtrl::NewView(SEffect_Sptr &effect, SolidDoc_Sptr &doc)	// 新的資料虛擬化View
{
	SolidView_Sptr	tmpPtr(new SolidView(this, doc));
	tmpPtr->SetEffect(effect);
	m_SolidViewPtrs.push_back(tmpPtr);
	return tmpPtr;
}

SolidView_Sptr SolidCtrl::NewSEffect(SEffect_Sptr effect)
{
	SolidDoc_Sptr	spDoc = NewDoc();
	assert(m_polydata->GetNumberOfPoints() != 0);
	spDoc->SetPolyData(m_polydata);
	spDoc->SetImageData(m_imagedata);

	SolidView_Sptr	spView = NewView(effect, spDoc);
	return spView;
}

void SolidCtrl::ReSetViewDirection()
{
	m_Camera->SetPosition(0, 0, (m_bounds.Xmid() + m_bounds.Ymid() + m_bounds.Zmid()) / 2);
	m_Camera->SetFocalPoint(m_bounds.Xmid(), m_bounds.Ymid(), m_bounds.Zmid());
}

void SolidCtrl::RmView(SolidView_Sptr view)
{
	for (SolidView_Sptrs::iterator it = m_SolidViewPtrs.begin(); it != m_SolidViewPtrs.end(); it++)
	{
		if (*it == view)
		{
			(*it)->SetVisable(false);
			m_SolidViewPtrs.erase(it);

			SolidView_Sptr	tmp;
			tmp.swap(view);
			break;
		}
	}
}

void SolidCtrl::RmDoc(SolidDoc_Sptr doc)
{
	for (SolidDoc_Sptrs::iterator it = m_SolidDocPtrs.begin(); it != m_SolidDocPtrs.end(); it++)
	{
		if (*it == doc)
		{
			m_SolidDocPtrs.erase(it);
			doc->RmAllView();

			SolidDoc_Sptr	tmp;
			tmp.swap(doc);
			break;
		}
	}
}

void SolidCtrl::RmAllView()
{
	for (SolidView_Sptrs::iterator it = m_SolidViewPtrs.begin(); it != m_SolidViewPtrs.end(); it++)
	{
		(*it)->SetVisable(false);

		SolidView_Sptr	tmp;
		tmp.swap(*it);
	}

	m_SolidViewPtrs.clear();
}

int SolidCtrl::SetUnGridData(vtkPolyData_Sptr polydata, InterpolationMethod method /*= NEAREST_NEIGHBOR*/ )
{
	m_sf3d = NULL;

	const float	distance = 1.0f;
	RmAllView();

	// 先載入資料
	m_polydata = vtkSmartNew;
	m_imagedata = vtkSmartNew;

	vtkPoints_Sptr		points = vtkSmartNew;
	vtkDoubleArray_Sptr	point_array = (vtkDoubleArray *) polydata->GetPointData()->GetScalars();
	point_array->SetName("value");

	// Griding資料
	switch (method)
	{
	case NEAREST_NEIGHBOR:
		{
			vtkNearestNeighborFilter_Sptr	NearestNeighbor = vtkSmartNew;
			NearestNeighbor->SetBounds(polydata->GetBounds());
			NearestNeighbor->SetInterval(distance, distance, distance);
			NearestNeighbor->SetInput(polydata);
			NearestNeighbor->Update();
			m_polydata = NearestNeighbor->GetOutput();
		}
		break;

	case INVERSE_DISTANCE:
		{
			vtkInverseDistanceFilter_Sptr	InverseDistance = vtkSmartNew;
			InverseDistance->SetBounds(polydata->GetBounds());
			InverseDistance->SetInterval(distance, distance, distance);
			InverseDistance->SetInput(polydata);
			InverseDistance->Update();
			m_polydata = InverseDistance->GetOutput();
		}
		break;
	}

	vtkBounds	tbounds;
	m_polydata->GetBounds(tbounds);
	m_polydata->GetPointData()->GetScalars()->SetName("value");
	m_bounds.SetBounds(m_polydata->GetBounds());
	m_imagedata->SetDimensions(tbounds.Xlen(), tbounds.Ylen(), tbounds.Zlen());
	m_imagedata->SetExtent(tbounds[0], tbounds[1], tbounds[2], tbounds[3], tbounds[4], tbounds[5]);

	//m_imagedata->SetOrigin(tbounds[0], tbounds[2], tbounds[4]);
	//m_imagedata->SetSpacing(0.8, 0.8, 0.8);
	m_imagedata->GetPointData()->SetScalars(m_polydata->GetPointData()->GetScalars());

	// 把資料跟bounding box建出來
	SolidDoc_Sptr	spDoc = NewDoc();
	assert(m_polydata->GetNumberOfPoints() != 0);
	spDoc->SetPolyData(m_polydata);
	spDoc->SetImageData(m_imagedata);
	point_array = (vtkDoubleArray *) m_polydata->GetPointData()->GetScalars();

	Histogramd	histg;
	for (vtkIdType i = 0; i < point_array->GetNumberOfTuples(); i++)
		histg.Append(point_array->GetValue(i));
	histg.Sort();
	spDoc->m_histogram = histg;
	ReSetViewDirection();
	return 0;
}

int SolidCtrl::SetGridedData(vtkImageData_Sptr image)
{
	m_sf3d = NULL;
	RmAllView();

	// 先載入資料
	m_polydata = vtkSmartNew;
	m_imagedata = image;
	vtkPoints_Sptr		points = vtkSmartNew;
	vtkDoubleArray_Sptr	point_array = (vtkDoubleArray *) image->GetPointData()->GetScalars();
	point_array->SetName("value");

	vtkBounds	tbounds;
	m_imagedata->GetBounds(tbounds);
	m_imagedata->GetPointData()->GetScalars()->SetName("value");
	m_bounds.SetBounds(m_imagedata->GetBounds());
	for (vtkIdType i = 0; i < m_imagedata->GetNumberOfPoints(); i++)
	{
		double	p[3];
		m_imagedata->GetPoint(i, p);
		points->InsertNextPoint(p);
	}

	m_polydata->SetPoints(points);
	m_polydata->GetPointData()->SetScalars(m_imagedata->GetPointData()->GetScalars());

	// 把資料跟bounding box建出來
	SolidDoc_Sptr	spDoc = NewDoc();
	assert(m_polydata->GetNumberOfPoints() != 0);
	spDoc->SetPolyData(m_polydata);
	spDoc->SetImageData(m_imagedata);
	point_array = (vtkDoubleArray *) m_polydata->GetPointData()->GetScalars();

	Histogramd	histg;
	for (vtkIdType i = 0; i < point_array->GetNumberOfTuples(); i++)
		histg.Append(point_array->GetValue(i));
	histg.Sort();
	spDoc->m_histogram = histg;
	ReSetViewDirection();
	return 0;
}

int SolidCtrl::SetGridedData(SJCScalarField3d *sf3d)
{
	RmAllView();
	m_sf3d = sf3d;

	// 先載入資料
	m_polydata = vtkSmartNew;
	m_imagedata = vtkSmartNew;

	vtkPoints_Sptr		points = vtkSmartNew;
	vtkDoubleArray_Sptr	point_array = vtkSmartNew;
	point_array->SetName("value");

	const uint	x_len = sf3d->NumX(), y_len = sf3d->NumY(), z_len = sf3d->NumZ();
	uint		i, j, k, offset = 0;
	for (k = 0; k < z_len; k++)
	{
		for (j = 0; j < y_len; j++)
		{
			for (i = 0; i < x_len; i++)
			{

				//printf("%d %d %d %f\t", dpvec[j][i]);
				point_array->InsertTuple1
					(
						offset++,
						*(sf3d->begin() + i + j * x_len + k * x_len * y_len)
					);
				points->InsertNextPoint(i * sf3d->DX(), j * sf3d->DY(), k * sf3d->DZ());
			}
		}
	}

	uint	count = point_array->GetNumberOfTuples();

	// 如果資料被Griding過了就直接放到imagedata
	bool	isGrided = x_len * y_len * z_len == count;
	m_polydata->SetPoints(points);
	m_polydata->GetPointData()->SetScalars(point_array);
	m_bounds.SetBounds(m_polydata->GetBounds());

	vtkBounds	tbounds;
	m_polydata->GetBounds(tbounds);
	m_imagedata->SetExtent(tbounds[0], tbounds[1], tbounds[2], tbounds[3], tbounds[4], tbounds[5]);
	if (isGrided)
	{
		m_imagedata->SetSpacing
			(
			m_bounds.Xlen() / (x_len - 1),
			m_bounds.Ylen() / (y_len - 1),
			m_bounds.Zlen() / (z_len - 1)
			);
		m_imagedata->SetDimensions(x_len, y_len, z_len);
		m_imagedata->GetPointData()->SetScalars(point_array);
	}
	else
	{
		assert(0 && "error: x_len* y_len* z_len == count ");
		return 1;
	}
	// 把資料建出來
	SolidDoc_Sptr	spDoc = NewDoc();
	assert(m_polydata->GetNumberOfPoints() != 0);
	spDoc->SetPolyData(m_polydata);
	if (isGrided)
	{
		spDoc->SetImageData(m_imagedata);
	}
	else
		assert(0 && "not is Grided");

	Histogramd	histg;
	for (vtkIdType i = 0; i < point_array->GetNumberOfTuples(); i++)
		histg.Append(point_array->GetValue(i));
	histg.Sort();
	spDoc->m_histogram = histg;
	ReSetViewDirection();
	return 0;
}

int SolidCtrl::SetGridedData(vtkPolyData_Sptr poly, int nx, int ny, int nz)
{
	RmAllView();
	m_sf3d = NULL;

	// 先載入資料
	m_polydata = poly;
	m_imagedata = vtkSmartNew;

	vtkPoints_Sptr	points = m_polydata->GetPoints();
	m_polydata->GetPointData()->GetScalars()->SetName("value");

	uint	count = m_polydata->GetPointData()->GetScalars()->GetNumberOfTuples();
	bool	isGrided = nx * ny * nz == count;
	m_bounds.SetBounds(m_polydata->GetBounds());

	double	orgin[3];
	m_polydata->GetPoint(0, orgin);
	if (isGrided)
	{
		m_imagedata->SetSpacing
			(
				m_bounds.Xlen() / (nx - 1),
				m_bounds.Ylen() / (ny - 1),
				m_bounds.Zlen() / (nz - 1)
			);
		m_imagedata->SetDimensions(nx, ny, nz);
		m_imagedata->GetPointData()->SetScalars(m_polydata->GetPointData()->GetScalars());
		m_imagedata->SetOrigin(m_bounds[0], m_bounds[2], m_bounds[4]);
	}
	else
	{
		MessageBoxA(0, "error: nx* ny* nz == count ", "", 0);
		assert(0 && "error: nx* ny* nz == count ");
		return 1;
	}
	// 把資料建出來
	SolidDoc_Sptr	spDoc = NewDoc();
	assert(m_polydata->GetNumberOfPoints() != 0);
	spDoc->SetPolyData(m_polydata);
	if (isGrided)
	{
		spDoc->SetImageData(m_imagedata);
	}
	else
	{
		MessageBoxA(0, "error: nx* ny* nz == count ", "", 0);
		assert(0 && "not is Grided");
	}

	vtkDoubleArray_Sptr	point_array = (vtkDoubleArray *) m_polydata->GetPointData()->GetScalars();
	Histogramd		histg;
	for (vtkIdType i = 0; i < point_array->GetNumberOfTuples(); i++)
		histg.Append(point_array->GetValue(i));
	histg.Sort();
	spDoc->m_histogram = histg;
	ReSetViewDirection();
	return 0;
}

void SolidCtrl::Render()
{
	m_RenderWindow->Render();
}
void SolidCtrl::AddTaiwan()
{
	AddTaiwan("TW100m.dat", 0, 503);
}

void SolidCtrl::AddTaiwan(char *datafilename, int col, int row)
{
	vtkPoints_Sptr	points = vtkSmartNew;
	std::ifstream istr1(datafilename);
	vtkSmartPointer<vtkDoubleArray> Scalars =
		vtkSmartPointer<vtkDoubleArray>::New();
	Scalars->SetNumberOfComponents(1);
	Scalars->SetName("Isovalues");

	int	i = 0;
	for (; !istr1.eof();)
	{
		double	x, y, z;
		istr1 >> x >> y >> z;
		points->InsertNextPoint(x, y*0.2, z);
		Scalars->InsertNextTuple1(y*0.1);
		i++;
	}

	assert(row + col > 0);
	if (row == 0)
		row = i / col;
	if (col == 0)
		col = i / row;

	int	count = 0;
	vtkSmartPointer<vtkCellArray>	cells = vtkSmartNew;
	for (int c = 0; c < col - 1; c++)
	{
		vtkSmartPointer<vtkTriangleStrip>	triangle = vtkSmartNew;
			for (int w = 0; w < row - 1; w++)
			{
				triangle->GetPointIds()->InsertNextId(c * row + w);
				triangle->GetPointIds()->InsertNextId((c + 1) * row + w);
			}
		cells->InsertNextCell(triangle);
	}
	vtkPolyData_Sptr	polydata = vtkSmartNew;
	polydata->SetPoints(points);
	polydata->SetStrips(cells);
	polydata->GetPointData()->SetScalars(Scalars);
	vtkPolyData_Sptr	polydata2 = vtkSmartNew;
	polydata2->DeepCopy(polydata);
	vtkPoints *ps = polydata2->GetPoints();
	double bound[6];
	ps->GetBounds(bound);
	for (int c = 0; c < ps->GetNumberOfPoints(); c++)
	{
		double pos[3];
		ps->GetPoint(c, pos);
		ps->SetPoint(c, pos[0], pos[1], pos[2]);
	}
	polydata2->SetPoints(ps);
	vtkPolyDataMapper_Sptr	mapper = vtkSmartNew;
	mapper->SetInput(polydata);

	vtkActor_Sptr	actor = vtkSmartNew;
	actor->SetMapper(mapper);
	//actor->GetProperty()->SetOpacity(0.7);
	//m_Renderer->AddActor(actor);

	vtkSmartPointer<vtkContourFilter> contours =
		vtkSmartPointer<vtkContourFilter>::New();
	contours->SetInput(polydata2);
	//contours->GenerateValues(7, 0, 3000);
	contours->GenerateValues(10, 0, 4000);

	// Connect the segments of the conours into polylines
	vtkSmartPointer<vtkStripper> contourStripper =
		vtkSmartPointer<vtkStripper>::New();
	contourStripper->SetInputConnection(contours->GetOutputPort());
	contourStripper->Update();

	int numberOfContourLines = contourStripper->GetOutput()->GetNumberOfLines();

	std::cout << "There are "
		<< numberOfContourLines << " contours lines."
		<< std::endl;

	vtkPoints *points2     =
		contourStripper->GetOutput()->GetPoints();
	vtkCellArray *cells2   =
		contourStripper->GetOutput()->GetLines();
	vtkSmartPointer<vtkCellArray> newcells = vtkSmartNew;
	newcells->Initialize();
	vtkDataArray *scalars =
		contourStripper->GetOutput()->GetPointData()->GetScalars();

	// Create a polydata that contains point locations for the contour
	// line labels
	vtkSmartPointer<vtkPolyData> labelPolyData =
		vtkSmartPointer<vtkPolyData>::New();
	vtkSmartPointer<vtkPoints> labelPoints =
		vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkDoubleArray> labelScalars =
		vtkSmartPointer<vtkDoubleArray>::New();
	labelScalars->SetNumberOfComponents(1);
	labelScalars->SetName("Isovalues");

	vtkIdType *indices;
	vtkIdType numberOfPoints;
	unsigned int lineCount = 0;
	
	for (cells2->InitTraversal();
		count = cells2->GetNextCell(numberOfPoints, indices);
		lineCount++)
	{
		if (numberOfPoints > 100)
		{
			newcells->InsertNextCell(numberOfPoints, indices);
			vtkIdType midPointId = indices[numberOfPoints / 2];
			midPointId = indices[static_cast<vtkIdType>(vtkMath::Random(0, numberOfPoints))];
			double midPoint[3];
			points2->GetPoint(midPointId, midPoint);
			labelPoints->InsertNextPoint(midPoint);
			labelScalars->InsertNextTuple1(scalars->GetTuple1(midPointId));
		}
	}
	cells2->DeepCopy(newcells);
	labelPolyData->SetPoints(labelPoints);
	labelPolyData->GetPointData()->SetScalars(labelScalars);

	vtkSmartPointer<vtkLookupTable> surfaceLUT =
		vtkSmartPointer<vtkLookupTable>::New();
	surfaceLUT->SetRange(
		polydata2->GetPointData()->GetScalars()->GetRange());
	surfaceLUT->SetNumberOfTableValues(7);
	surfaceLUT->Build();

	double range[2];
	polydata2->GetPointData()->GetScalars()->GetRange(range);
	double step = range[1]-range[0];
	vtkColorTransferFunction_Sptr	colorTransferFunction = vtkSmartNew;
	colorTransferFunction->AddRGBPoint(range[0]+step, 1.0 / 2, 0.0, 0.0);
	colorTransferFunction->AddRGBPoint(range[0]+step/6*5, 1.0 / 2, 165 / 255 / 2.0, 0.0);
	colorTransferFunction->AddRGBPoint(range[0]+step/6*4, 1.0 / 2, 1.0 / 2, 0.0);
	colorTransferFunction->AddRGBPoint(range[0]+step/6*3, 0.0, 1.0 / 2, 0.0);
	colorTransferFunction->AddRGBPoint(range[0]+step/6*2, 0.0, 0.5 / 2, 1.0 / 2);
	colorTransferFunction->AddRGBPoint(range[0]+step/6*1, 0.0, 0.0, 1.0 / 2);
	colorTransferFunction->AddRGBPoint(range[0], 139 / 255.0 / 2, 0.0, 1.0 / 2);

	vtkScalarBarActor_Sptr ScalarBarActor = vtkSmartNew;
	ScalarBarActor->SetLookupTable(colorTransferFunction);
	ScalarBarActor->SetNumberOfLabels(4);
	ScalarBarActor->SetMaximumWidthInPixels(60);
	ScalarBarActor->SetMaximumHeightInPixels(300);
	ScalarBarActor->SetLabelFormat("%-.0f");
	m_Renderer->AddActor2D(ScalarBarActor);

	vtkSmartPointer<vtkPolyDataMapper> contourMapper =
		vtkSmartPointer<vtkPolyDataMapper>::New();
	contourMapper->SetInputConnection(contourStripper->GetOutputPort());
	contourMapper->SetLookupTable(colorTransferFunction);
	contourMapper->ScalarVisibilityOn();
	contourMapper->SetScalarRange(
		polydata->GetPointData()->GetScalars()->GetRange());


	vtkSmartPointer<vtkActor> isolines =
		vtkSmartPointer<vtkActor>::New();
	isolines->SetMapper(contourMapper);

	vtkSmartPointer<vtkPolyDataMapper> surfaceMapper =
		vtkSmartPointer<vtkPolyDataMapper>::New();
	surfaceMapper->SetInput(polydata);  
	surfaceMapper->ScalarVisibilityOn();
	surfaceMapper->SetScalarRange(
		polydata->GetPointData()->GetScalars()->GetRange());
	surfaceMapper->SetLookupTable(colorTransferFunction);
	
	vtkSmartPointer<vtkActor> surface =
		vtkSmartPointer<vtkActor>::New();
	surface->SetMapper(surfaceMapper);
	//surface->GetProperty()->SetOpacity(0.5);

	// The labeled data mapper will place labels at the points
	vtkSmartPointer<vtkLabeledDataMapper> labelMapper =
		vtkSmartPointer<vtkLabeledDataMapper>::New();
	labelMapper->SetFieldDataName("Isovalues");
	labelMapper->SetInput(labelPolyData);
	labelMapper->SetLabelModeToLabelScalars();
	labelMapper->SetLabelFormat("%6.0f");

	vtkSmartPointer<vtkActor2D> isolabels =
		vtkSmartPointer<vtkActor2D>::New();
	isolabels->SetMapper(labelMapper);

	// Add the actors to the scene
	m_Renderer->AddActor(isolines);
	//m_Renderer->AddActor(isolabels);
	m_Renderer->AddActor(surface);
}
