add_library(femx_linalg)
add_library(femx::linalg ALIAS femx_linalg)

target_sources(
  femx_linalg
  PRIVATE ../common/Math.cpp handler/MatrixHandler.cpp handler/VectorHandler.cpp
          native/DenseLinearSolver.cpp
  PUBLIC Backend.hpp
         View.hpp
         CsrPattern.hpp
         CsrMatrix.hpp
         DenseMatrix.hpp
         handler/MatrixHandler.hpp
         handler/VectorHandler.hpp
         LinearSolver.hpp
         native/DenseLinearSolver.hpp
         Vector.hpp)

target_link_libraries(femx_linalg PUBLIC femx_config femx_common)

if(FEMX_HAS_CUDA)
  target_sources(
    femx_linalg PRIVATE ../common/Cuda.cu handler/CudaHandles.cu
                        handler/VectorHandler.cu handler/MatrixHandler.cu)
  target_link_libraries(femx_linalg PUBLIC CUDA::cudart CUDA::cusparse
                                           CUDA::cublas)
  set_target_properties(femx_linalg PROPERTIES CUDA_STANDARD 17
                                               CUDA_STANDARD_REQUIRED ON)
else()
  target_sources(femx_linalg PRIVATE ../common/Cuda.cpp)
endif()

if(FEMX_ENABLE_RESOLVE)
  add_subdirectory(resolve)
endif()
