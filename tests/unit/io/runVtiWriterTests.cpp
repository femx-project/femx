#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

#include <femx/io/VtiWriter.hpp>
#include <femx/linalg/Vector.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class VtiWriterTests : public TestBase
{
public:
  TestOutcome writeBinaryCellField()
  {
    TestStatus status;
    status = true;

    const auto dir =
        std::filesystem::temp_directory_path() / "femx_vti_writer_tests";
    std::filesystem::create_directories(dir);
    const auto path = dir / "cell-data.vti";

    VtiWriter::Image image;
    image.cell_counts = {2, 2, 1};
    image.origin      = {-0.5, -0.5, 0.0};
    image.spacing     = {1.0, 1.0, 1.0};
    image.time        = 1.25;

    Vector<Real> velocity(12);
    for (Index i = 0; i < velocity.size(); ++i)
    {
      velocity[i] = static_cast<Real>(i);
    }

    VtiWriter writer;
    writer.writeCellData(
        path.string(), image, {{"velocity", 3, &velocity}});

    status *= std::filesystem::exists(path);

    std::ifstream in(path, std::ios::binary);
    const std::string contents((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());

    status *= contents.find("type=\"ImageData\"") != std::string::npos;
    status *= contents.find("CellData") != std::string::npos;
    status *= contents.find("Name=\"velocity\"") != std::string::npos;
    status *= contents.find("format=\"appended\"") != std::string::npos;
    status *= contents.find("TimeValue") != std::string::npos;

    const auto marker = contents.find("\n_");
    status *= marker != std::string::npos;
    if (marker != std::string::npos
        && marker + 2 + sizeof(std::uint64_t) <= contents.size())
    {
      std::uint64_t bytes = 0;
      std::memcpy(&bytes, contents.data() + marker + 2, sizeof(bytes));
      status *= bytes == static_cast<std::uint64_t>(velocity.size())
                           * sizeof(Real);
    }
    else
    {
      status = false;
    }

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running VTI writer tests:\n";

  femx::tests::VtiWriterTests test;

  femx::tests::TestingResults result;
  result += test.writeBinaryCellField();

  return result.summary();
}
