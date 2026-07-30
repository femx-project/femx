add_library(femx_linalg)
add_library(femx::linalg ALIAS femx_linalg)

target_sources(
  femx_linalg
  PRIVATE cuda/CudaContext.cpp
          cuda/CudaSystemMatrix.cpp
          cuda/CudaLinearSystem.cpp
          cuda/CudaVectorHandler.cpp
          native/DenseLinearSolver.cpp
          native/HostSystemMatrix.cpp
          native/HostLinearSystem.cpp
          native/HostVectorHandler.cpp
  PUBLIC Context.hpp
         SystemMatrix.hpp
         LinearSystem.hpp
         CsrPattern.hpp
         CsrMatrix.hpp
         DenseMatrix.hpp
         cuda/CudaContext.hpp
         cuda/CudaSystemMatrix.hpp
         cuda/CudaLinearSystem.hpp
         cuda/CudaVectorHandler.hpp
         LinearSolver.hpp
         native/DenseLinearSolver.hpp
         native/HostContext.hpp
         native/HostSystemMatrix.hpp
         native/HostLinearSystem.hpp
         native/HostVectorHandler.hpp)

target_link_libraries(femx_linalg PUBLIC femx_config femx_common)

if(FEMX_HAS_CUDA)
  target_sources(
    femx_linalg PRIVATE cuda/CudaHandles.cu cuda/CudaSystemMatrix.cu
                        cuda/CudaVectorHandler.cu)
  target_link_libraries(femx_linalg PUBLIC CUDA::cusparse CUDA::cublas)
  set_target_properties(femx_linalg PROPERTIES CUDA_STANDARD 17
                                               CUDA_STANDARD_REQUIRED ON)
endif()

if(FEMX_ENABLE_RESOLVE)
  add_subdirectory(resolve)
endif()
