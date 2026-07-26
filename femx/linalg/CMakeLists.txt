add_library(femx_linalg)
add_library(femx::linalg ALIAS femx_linalg)

target_sources(
  femx_linalg
  PRIVATE ../common/Math.cpp
          cuda/CudaContext.cpp
          cuda/CudaJacobian.cpp
          cuda/CudaLinearSystem.cpp
          cuda/CudaVectorHandler.cpp
          native/DenseLinearSolver.cpp
          native/HostJacobian.cpp
          native/HostLinearSystem.cpp
          native/HostVectorHandler.cpp
  PUBLIC Context.hpp
         Jacobian.hpp
         LinearSystem.hpp
         View.hpp
         CsrPattern.hpp
         CsrMatrix.hpp
         DenseMatrix.hpp
         cuda/CudaContext.hpp
         cuda/CudaJacobian.hpp
         cuda/CudaLinearSystem.hpp
         cuda/CudaVectorHandler.hpp
         LinearSolver.hpp
         native/DenseLinearSolver.hpp
         native/HostContext.hpp
         native/HostJacobian.hpp
         native/HostLinearSystem.hpp
         native/HostVectorHandler.hpp
         Vector.hpp)

target_link_libraries(femx_linalg PUBLIC femx_config femx_common)

if(FEMX_HAS_CUDA)
  target_sources(
    femx_linalg PRIVATE ../common/Cuda.cu handler/CudaHandles.cu
                        cuda/CudaJacobian.cu cuda/CudaVectorHandler.cu)
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
