add_library(femx_io)
add_library(femx::io ALIAS femx_io)

target_sources(
  femx_io
  PRIVATE DataOut.cpp Hdf5Writer.cpp TimeSeriesDataIn.cpp
          TimeSeriesDataOut.cpp VtiWriter.cpp XdmfWriter.cpp
  PUBLIC DataOut.hpp Hdf5Writer.hpp TimeSeriesDataIn.hpp
         TimeSeriesDataOut.hpp VtiWriter.hpp XdmfWriter.hpp)

target_link_libraries(femx_io PUBLIC femx_config femx_core femx_algebra
                                     femx_fem)

if(HDF5_FOUND)
  target_include_directories(femx_io PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(femx_io PRIVATE ${HDF5_LIBRARIES})

  if(HDF5_IS_PARALLEL AND MPI_CXX_FOUND)
    target_link_libraries(femx_io PRIVATE MPI::MPI_CXX)
  endif()

  if(FEMX_MPI_INCLUDE_DIR)
    target_include_directories(femx_io PRIVATE ${FEMX_MPI_INCLUDE_DIR})
  endif()
endif()
