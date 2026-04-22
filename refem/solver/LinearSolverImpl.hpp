/**
 * @file   LinearSolverImpl.h
 * @author Kakeru Ueda (ueda.k.2290@m.isct.ac.jp)
 *
 */

#pragma once

#include <string>

namespace refem
{

class SparseMatrix;
class Vector;

class LinearSolverImpl
{
public:
  virtual ~LinearSolverImpl() = default;

  virtual void setOperator(const SparseMatrix& A)           = 0;
  virtual void setPreconditioner(const std::string& method) = 0;
  virtual void solve(const Vector& b, Vector& x)            = 0;
};

} // namespace refem
