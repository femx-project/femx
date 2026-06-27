#pragma once

#include <femx/linalg/operator/LinearOperator.hpp>
#include <femx/linalg/operator/MatrixBuilder.hpp>

namespace femx
{
namespace linalg
{

/** @brief Matrix that can be assembled and applied as a linear operator. */
class MatrixOperator : public LinearOperator, public MatrixBuilder
{
public:
  ~MatrixOperator() override = default;

  Index numRows() const override = 0;
  Index numCols() const override = 0;
};

} // namespace linalg
} // namespace femx
