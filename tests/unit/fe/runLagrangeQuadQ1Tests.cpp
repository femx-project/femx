#include <iostream>

#include "LagrangeQuadQ1Tests.hpp"

int main(int, char**)
{
  std::cout << "Running LagrangeQuadQ1 tests:\n";

  refem::tests::LagrangeQuadQ1Tests test;

  refem::tests::TestingResults result;
  result += test.elementMetadata();
  result += test.shapeAtReferenceNodes();
  result += test.shapePartitionOfUnity();
  result += test.shapeGradientAtCenter();

  return result.summary();
}
