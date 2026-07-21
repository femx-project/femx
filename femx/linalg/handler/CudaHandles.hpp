#pragma once

#include <cublas_v2.h>
#include <cusparse.h>

namespace femx::linalg::detail
{

void checkCublas(cublasStatus_t status, const char* operation);
void checkCusparse(cusparseStatus_t status, const char* operation);

cublasHandle_t   cublasHandle(void* stream);
cusparseHandle_t cusparseHandle(void* stream);

} // namespace femx::linalg::detail
