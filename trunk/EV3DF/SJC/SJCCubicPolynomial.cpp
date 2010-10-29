/*
** $Header: /p/graphics/CVS/yu-chi/SJC/SJCCubicPolynomial.cpp,v 1.1.1.1 2006/04/25 20:21:43 yu-chi Exp $
**
** (c) 2003-2005 Stephen Chenney
**
** $Log: SJCCubicPolynomial.cpp,v $
** Revision 1.1.1.1  2006/04/25 20:21:43  yu-chi
**
**
** Revision 1.1.1.1  2005/08/29 20:24:14  schenney
** New libSJC source repository.
**
*/

#include <SJCCubicPolynomial.h>
#include <math.h>

static int  LU_Decomp(double **a, int n, int *indx, int *d, double *vv);
static void LU_Back_Subst(double **a, int n, int *indx, double *b);


SJCCubicPolynomial::SJCCubicPolynomial(const std::pair<double,double> points[4])
{
    double  **m = new double*[4];
    for ( uint i = 0 ; i < 4 ; i++ )
	m[i] = new double[4];
    int	    *index = new int[4];
    double  *vv = new double[4];

    for ( uint i = 0 ; i < 4 ; i++ )
    {
	m[i][3] = 1.0;
	m[i][2] = points[i].first;
	m[i][1] = points[i].first * points[i].first;
	m[i][0] = m[i][1] * points[i].first;
	coefficients[i] = points[i].second;
    }

    int	flip;
    LU_Decomp(m, 4, index, &flip, vv);
    LU_Back_Subst(m, 4, index, coefficients);

    for ( uint i = 0 ; i < 4 ; i++ )
	delete m[i];
    delete[] m;
    delete[] index;
    delete[] vv;
}


double
SJCCubicPolynomial::operator()(const double u) const
{
    double  u_sq = u * u;

    return u_sq * u * coefficients[0] + u_sq * coefficients[1]
	 +        u * coefficients[2] +        coefficients[3];
}


std::ostream&
operator<<(std::ostream &o, const SJCCubicPolynomial &b)
{
    o << "< " << b.coefficients[0] << ", "
	      << b.coefficients[1] << ", "
	      << b.coefficients[2] << ", "
	      << b.coefficients[3] << " >";

    return o;
}


static int
LU_Decomp(double **a, int n, int *indx, int *d, double *vv)
{
    int     i, imax, j, k;
    double  big, dum, sum, temp;

    *d = 1;
    for ( i = 0 ; i < n ; i++ )
    {
        big = 0.0;
        for ( j = 0 ; j < n ; j++ )
            if ( ( temp = fabs(a[i][j]) ) > big )
                big = temp;
        if ( big == 0.0 )
        {
            fprintf(stderr, "Singular matrix in LU_Decomp\n");
            return 0;
        }
        vv[i] = 1.0 / big;
    }
    for ( j = 0 ; j < n ; j++ )
    {
        for ( i = 0 ; i < j ; i++ )
        {
            sum = a[i][j];
            for ( k = 0 ; k < i ; k++ )
                sum -= a[i][k] * a[k][j];
            a[i][j] = sum;
        }
        big = 0.0;
        for ( i = j ; i < n ; i++ )
        {
            sum = a[i][j];
            for ( k = 0 ; k < j ; k++ )
                sum -= a[i][k] * a[k][j];
            a[i][j] = sum;
            if ( ( dum = vv[i] * fabs(sum) ) >= big )
            {
                big = dum;
                imax = i;
            }
        }
        if ( j != imax )
        {
            for ( k =0 ; k < n ; k++ )
            {
                dum = a[imax][k];
                a[imax][k] = a[j][k];
                a[j][k] = dum;
            }
            *d = - (*d);
            vv[imax] = vv[j];
        }
        indx[j] = imax;
        if ( a[j][j] == 0.0 )
            a[j][j] = 1.0e-16;
        if ( j != n - 1 )
        {
            dum = 1.0 / a[j][j];
            for ( i = j + 1 ; i < n ; i++ )
                a[i][j] *= dum;
        }
    }

    return 1;
}


static void
LU_Back_Subst(double **a, int n, int *indx, double *b)
{
    int     i, ii = -1, ip, j;
    double  sum;

    for ( i = 0 ; i < n ; i++ )
    {
        ip = indx[i];
        sum = b[ip];
        b[ip] = b[i];
        if ( ii != -1 )
            for ( j = ii ; j <= i - 1 ; j++ )
                sum -= a[i][j] * b[j];
        else if ( sum )
            ii = i;
        b[i] = sum;
    }
    for ( i = n - 1 ; i >= 0 ; i-- )
    {
        sum = b[i];
        for ( j = i + 1 ; j < n ; j++ )
            sum -= a[i][j] * b[j];
        b[i] = sum / a[i][i];
    }
}

