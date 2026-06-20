#pragma once

#include <femx/algebra/LinearOperator.hpp>
#include <femx/algebra/MatrixBuilder.hpp>

namespace femx
{
namespace algebra
{

/** @brief Matrix that can be assembled and applied as a linear operator. */
class MatrixOperator : public LinearOperator, public MatrixBuilder
{
public:
  ~MatrixOperator() override = default;

  Index numRows() const override = 0;
  Index numCols() const override = 0;
};

} // namespace algebra
} // namespace femx
