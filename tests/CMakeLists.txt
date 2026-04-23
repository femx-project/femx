add_executable(runLagrangeQuadQ1Tests unit/fe/runLagrangeQuadQ1Tests.cpp)

target_link_libraries(runLagrangeQuadQ1Tests PRIVATE refem::refem)

add_test(NAME LagrangeQuadQ1Tests COMMAND runLagrangeQuadQ1Tests)
