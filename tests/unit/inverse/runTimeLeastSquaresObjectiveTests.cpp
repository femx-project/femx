#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <femx/eq/TimeStateTrajectory.hpp>
#include <femx/inverse/TimeLeastSquaresObjective.hpp>
#include <femx/inverse/TimeObservationData.hpp>
#include <femx/inverse/TimeObservationOperator.hpp>
#include <femx/linalg/Vector.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

namespace
{

std::string readTextFile(const std::filesystem::path& path)
{
  std::ifstream in(path);
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

void resize(Vector<Real>& out,
            Index         size)
{
  if (out.size() != size)
  {
    out.resize(size);
  }
  else
  {
    out.setZero();
  }
}

} // namespace

class LinearTimeObservation final : public inverse::TimeObservationOperator
{
public:
  Index numSteps() const override
  {
    return 2;
  }

  Index numStates() const override
  {
    return 2;
  }

  Index numParams() const override
  {
    return 2;
  }

  Index numObservations() const override
  {
    return 2;
  }

  void observe(Index               time_level,
               const Vector<Real>& state,
               const Vector<Real>& prm,
               Vector<Real>&       out) const override
  {
    resize(out, numObservations());
    out[0] = state[0] + static_cast<Real>(time_level + 1) * prm[0];
    out[1] = 2.0 * state[1] - prm[1];
  }

  void applyStateJac(Index               time_level,
                     const Vector<Real>& state,
                     const Vector<Real>& prm,
                     const Vector<Real>& dir,
                     Vector<Real>&       out) const override
  {
    (void) time_level;
    (void) state;
    (void) prm;
    resize(out, numObservations());
    out[0] = dir[0];
    out[1] = 2.0 * dir[1];
  }

  void applyStateJacT(Index               time_level,
                      const Vector<Real>& state,
                      const Vector<Real>& prm,
                      const Vector<Real>& dir,
                      Vector<Real>&       out) const override
  {
    (void) time_level;
    (void) state;
    (void) prm;
    resize(out, numStates());
    out[0] = dir[0];
    out[1] = 2.0 * dir[1];
  }

  void applyParamJac(Index               time_level,
                     const Vector<Real>& state,
                     const Vector<Real>& prm,
                     const Vector<Real>& dir,
                     Vector<Real>&       out) const override
  {
    (void) state;
    (void) prm;
    resize(out, numObservations());
    out[0] = static_cast<Real>(time_level + 1) * dir[0];
    out[1] = -dir[1];
  }

  void applyParamJacT(Index               time_level,
                      const Vector<Real>& state,
                      const Vector<Real>& prm,
                      const Vector<Real>& dir,
                      Vector<Real>&       out) const override
  {
    (void) state;
    (void) prm;
    resize(out, numParams());
    out[0] = static_cast<Real>(time_level + 1) * dir[0];
    out[1] = -dir[1];
  }
};

inverse::TimeObservationData makeData()
{
  inverse::TimeObservationData data(3, 2);
  data.setLayout(
      "point",
      std::vector<Point3>{Point3{0.25, 0.5, 0.0}},
      Vector<Index>{0, 1});

  data[0][0] = 100.0;
  data[0][1] = -100.0;
  data[1][0] = 1.0;
  data[1][1] = -2.0;
  data[2][0] = 2.0;
  data[2][1] = 3.0;
  return data;
}

inverse::TimeObservationData makeSparseData()
{
  inverse::TimeObservationData data(2, 2);
  data.setLayout(
      "point",
      std::vector<Point3>{Point3{0.25, 0.5, 0.0}},
      Vector<Index>{0, 1});
  data.setTimeLevels(Vector<Index>{1, 2});

  data[0][0] = 1.0;
  data[0][1] = -2.0;
  data[1][0] = 2.0;
  data[1][1] = 3.0;
  return data;
}

inverse::TimeObservationData makeInterpolatedData()
{
  inverse::TimeObservationData data(1, 2);
  data.setLayout(
      "point",
      std::vector<Point3>{Point3{0.25, 0.5, 0.0}},
      Vector<Index>{0, 1});
  data.setTimeValues(Vector<Real>{1.5});

  data[0][0] = 1.0;
  data[0][1] = 1.0;
  return data;
}

Vector<Real> makeWeights()
{
  Vector<Real> weights(3);
  weights[0] = 0.0;
  weights[1] = 2.0;
  weights[2] = 0.5;
  return weights;
}

eq::TimeStateTrajectory makeTrajectory()
{
  eq::TimeStateTrajectory tr(2, 2);
  tr[0][0] = 0.0;
  tr[0][1] = 0.0;
  tr[1][0] = 1.0;
  tr[1][1] = -1.0;
  tr[2][0] = 0.5;
  tr[2][1] = 2.0;
  return tr;
}

class TimeLeastSquaresObjectiveTests : public TestBase
{
public:
  TestOutcome valueAndGradientsUseTimeObservationTranspose()
  {
    TestStatus status;
    status = true;

    LinearTimeObservation                    obs;
    const inverse::TimeLeastSquaresObjective obj(
        obs, makeData(), makeWeights());

    eq::TimeStateTrajectory tr = makeTrajectory();

    Vector<Real> prm(2);
    prm[0] = 0.25;
    prm[1] = -0.5;

    status *= (obj.numSteps() == obs.numSteps());
    status *= (obj.numStates() == obs.numStates());
    status *= (obj.numParams() == obs.numParams());

    status *= isEqual(obj.value(tr, prm), 1.203125);

    Vector<Real> grad;
    obj.stateGrad(0, tr, prm, grad);
    status *= isEqual(grad[0], 0.0);
    status *= isEqual(grad[1], 0.0);

    obj.stateGrad(1, tr, prm, grad);
    status *= isEqual(grad[0], 1.0);
    status *= isEqual(grad[1], 2.0);

    obj.stateGrad(2, tr, prm, grad);
    status *= isEqual(grad[0], -0.375);
    status *= isEqual(grad[1], 1.5);

    obj.paramGrad(tr, prm, grad);
    status *= isEqual(grad[0], 0.875);
    status *= isEqual(grad[1], -1.75);

    return status.report(__func__);
  }

  TestOutcome constructorRejectsWrongTimeLevelCount()
  {
    TestStatus status;
    status = true;

    LinearTimeObservation        obs;
    inverse::TimeObservationData data(2, 2);

    bool threw = false;
    try
    {
      const inverse::TimeLeastSquaresObjective obj(obs, data);
      (void) obj;
    }
    catch (const std::runtime_error&)
    {
      threw = true;
    }

    status *= threw;
    return status.report(__func__);
  }

  TestOutcome constructorRejectsNegativeWeights()
  {
    TestStatus status;
    status = true;

    LinearTimeObservation obs;
    Vector<Real>          weights = makeWeights();
    weights[1]                    = -1.0;

    bool threw = false;
    try
    {
      const inverse::TimeLeastSquaresObjective obj(
          obs, makeData(), weights);
      (void) obj;
    }
    catch (const std::runtime_error&)
    {
      threw = true;
    }

    status *= threw;
    return status.report(__func__);
  }

  TestOutcome explicitTimeLevelsSelectTrajectoryRows()
  {
    TestStatus status;
    status = true;

    LinearTimeObservation                    obs;
    const inverse::TimeLeastSquaresObjective obj(
        obs, makeSparseData(), makeWeights());

    eq::TimeStateTrajectory tr = makeTrajectory();

    Vector<Real> prm(2);
    prm[0] = 0.25;
    prm[1] = -0.5;

    status *= isEqual(obj.value(tr, prm), 1.203125);

    Vector<Real> grad;
    obj.stateGrad(0, tr, prm, grad);
    status *= isEqual(grad[0], 0.0);
    status *= isEqual(grad[1], 0.0);

    obj.stateGrad(1, tr, prm, grad);
    status *= isEqual(grad[0], 1.0);
    status *= isEqual(grad[1], 2.0);

    obj.stateGrad(2, tr, prm, grad);
    status *= isEqual(grad[0], -0.375);
    status *= isEqual(grad[1], 1.5);

    obj.paramGrad(tr, prm, grad);
    status *= isEqual(grad[0], 0.875);
    status *= isEqual(grad[1], -1.75);

    return status.report(__func__);
  }

  TestOutcome timeValuesInterpolateTrajectoryRows()
  {
    TestStatus status;
    status = true;

    LinearTimeObservation obs;
    Vector<Real>          weights(3);
    weights[0] = 1.0;
    weights[1] = 1.0;
    weights[2] = 1.0;
    const inverse::TimeLeastSquaresObjective obj(
        obs, makeInterpolatedData(), weights, 1.0);

    eq::TimeStateTrajectory tr = makeTrajectory();

    Vector<Real> prm(2);
    prm[0] = 0.25;
    prm[1] = -0.5;

    status *= isEqual(obj.value(tr, prm), 0.1953125);

    Vector<Real> grad;
    obj.stateGrad(0, tr, prm, grad);
    status *= isEqual(grad[0], 0.0);
    status *= isEqual(grad[1], 0.0);

    obj.stateGrad(1, tr, prm, grad);
    status *= isEqual(grad[0], 0.1875);
    status *= isEqual(grad[1], 0.5);

    obj.stateGrad(2, tr, prm, grad);
    status *= isEqual(grad[0], 0.1875);
    status *= isEqual(grad[1], 0.5);

    obj.paramGrad(tr, prm, grad);
    status *= isEqual(grad[0], 0.9375);
    status *= isEqual(grad[1], -0.5);

    return status.report(__func__);
  }

  TestOutcome timeObservationDataRoundTrips()
  {
    TestStatus status;
    status = true;

    const auto path = std::filesystem::temp_directory_path()
                      / "femx-time-observation-data-test.txt";
    const inverse::TimeObservationData data = makeData();

    inverse::writeTimeObsData(path.string(), data);
    const std::string text = readTextFile(path);
    const inverse::TimeObservationData loaded =
        inverse::readTimeObsData(path.string());
    std::filesystem::remove(path);

    status *= text.find("femx_time_obs_data ") == std::string::npos;
    status *= text.find("sampler") == std::string::npos;
    status *= text.find("num_observations") == std::string::npos;
    status *= text.find("values\n  level 0\n") != std::string::npos;
    status *= (loaded.numLevels() == data.numLevels());
    status *= (loaded.numObservations() == data.numObservations());
    status *= loaded.hasLayout();
    status *= (loaded.sampler() == "point");
    status *= (loaded.points().size() == 1);
    status *= isEqual(loaded.points()[0][0], 0.25);
    status *= isEqual(loaded.points()[0][1], 0.5);
    status *= (loaded.components().size() == 2);
    status *= (loaded.components()[0] == 0);
    status *= (loaded.components()[1] == 1);
    for (Index level = 0; level < data.numLevels(); ++level)
    {
      const Vector<Real> expected = data[level];
      const Vector<Real> actual   = loaded[level];
      for (Index i = 0; i < data.numObservations(); ++i)
      {
        status *= isEqual(actual[i], expected[i]);
      }
    }

    return status.report(__func__);
  }

  TestOutcome timeObservationDataRoundTripsExplicitTimeLevels()
  {
    TestStatus status;
    status = true;

    const auto path = std::filesystem::temp_directory_path()
                      / "femx-time-observation-data-levels-test.txt";
    const inverse::TimeObservationData data = makeSparseData();

    inverse::writeTimeObsData(path.string(), data);
    const inverse::TimeObservationData loaded =
        inverse::readTimeObsData(path.string());
    std::filesystem::remove(path);

    status *= loaded.hasTimeLevels();
    status *= (loaded.numLevels() == data.numLevels());
    status *= (loaded.timeLevels().size() == 2);
    status *= (loaded.timeLevel(0) == 1);
    status *= (loaded.timeLevel(1) == 2);
    for (Index row = 0; row < data.numLevels(); ++row)
    {
      const Vector<Real> expected = data[row];
      const Vector<Real> actual   = loaded[row];
      for (Index i = 0; i < data.numObservations(); ++i)
      {
        status *= isEqual(actual[i], expected[i]);
      }
    }

    return status.report(__func__);
  }

  TestOutcome timeObservationDataRoundTripsTimeValues()
  {
    TestStatus status;
    status = true;

    const auto path = std::filesystem::temp_directory_path()
                      / "femx-time-observation-data-values-test.txt";
    const inverse::TimeObservationData data = makeInterpolatedData();

    inverse::writeTimeObsData(path.string(), data);
    const std::string text = readTextFile(path);
    const inverse::TimeObservationData loaded =
        inverse::readTimeObsData(path.string());
    std::filesystem::remove(path);

    status *= text.find("time_values\n  1.5\n") != std::string::npos;
    status *= text.find("num_observations") == std::string::npos;
    status *= loaded.hasTimeValues();
    status *= (loaded.numLevels() == data.numLevels());
    status *= (loaded.timeValues().size() == 1);
    status *= isEqual(loaded.timeValue(0), 1.5);
    status *= isEqual(loaded[0][0], data[0][0]);
    status *= isEqual(loaded[0][1], data[0][1]);

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running time least-squares objective tests:\n";

  femx::tests::TimeLeastSquaresObjectiveTests test;

  femx::tests::TestingResults result;
  result += test.valueAndGradientsUseTimeObservationTranspose();
  result += test.constructorRejectsWrongTimeLevelCount();
  result += test.constructorRejectsNegativeWeights();
  result += test.explicitTimeLevelsSelectTrajectoryRows();
  result += test.timeValuesInterpolateTrajectoryRows();
  result += test.timeObservationDataRoundTrips();
  result += test.timeObservationDataRoundTripsExplicitTimeLevels();
  result += test.timeObservationDataRoundTripsTimeValues();

  return result.summary();
}
