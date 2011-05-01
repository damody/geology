
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
VTK_SMART_POINTER(vtkNearestNeighborFilter)
VTK_SMART_POINTER(vtkInverseDistanceFilter)
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
	int		num = m_polydata->GetNumberOfPoints();
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

void SolidCtrl::AddTaiwan(char *datafilename, int col, int raw)
{
	vtkPoints_Sptr	points = vtkSmartNew;
	std::ifstream istr1(datafilename);

	vtkSmartPointer<vtkTriangleStrip>	triangle = vtkSmartNew;
	int					i = 0;
	for (; !istr1.eof();)
	{
		double	x, y, z;
		istr1 >> x >> y >> z;
		points->InsertNextPoint(x, y, z);
		i++;
	}

	assert(raw + col > 0);
	if (raw == 0)
		raw = i / col;
	if (col == 0)
		col = i / raw;

	int	count = 0;
	for (int c = 0; c < col - 1; c++)
	{
		if (c % 2 == 0)
		{
			for (int w = 0; w < raw - 1; w++)
			{
				triangle->GetPointIds()->InsertNextId(c * raw + w);
				triangle->GetPointIds()->InsertNextId((c + 1) * raw + w);
			}
		}
		else
		{
			for (int w = raw - 1; w > 0; w--)
			{
				triangle->GetPointIds()->InsertNextId((c + 1) * raw + w);
				triangle->GetPointIds()->InsertNextId(c * raw + w - 1);
			}
		}
	}

	vtkSmartPointer<vtkCellArray>	cells = vtkSmartNew;
	cells->InsertNextCell(triangle);

	vtkPolyData_Sptr	polydata = vtkSmartNew;
	polydata->SetPoints(points);
	polydata->SetStrips(cells);

	vtkPolyDataMapper_Sptr	mapper = vtkSmartNew;
	mapper->SetInput(polydata);

	vtkActor_Sptr	actor = vtkSmartNew;
	actor->SetMapper(mapper);
	actor->GetProperty()->SetOpacity(0.5);
	m_Renderer->AddActor(actor);
}
