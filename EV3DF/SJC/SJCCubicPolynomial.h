/*
** $Header: /p/graphics/CVS/yu-chi/SJC/SJCCubicPolynomial.h,v 1.1.1.1 2006/04/25 20:21:50 yu-chi Exp $
**
** (c) 2003-2005 Stephen Chenney
**
** $Log: SJCCubicPolynomial.h,v $
** Revision 1.1.1.1  2006/04/25 20:21:50  yu-chi
**
**
** Revision 1.1.1.1  2005/08/29 20:24:14  schenney
** New libSJC source repository.
**
**
*/

#ifndef SJCCUBICPOLYNOMIAL_H_
#define SJCCUBICPOLYNOMIAL_H_

#include <iostream>
#include <utility>


class SJCCubicPolynomial {
  public:
    SJCCubicPolynomial(const std::pair<double,double> points[4]);

    double  operator()(const double u) const;

    friend  std::ostream& operator<<(std::ostream &o,
				     const SJCCubicPolynomial &b);

  private:
    double  coefficients[4];
};


#endif

