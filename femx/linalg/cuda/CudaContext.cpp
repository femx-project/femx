#include <femx/common/Checks.hpp>
#include <femx/common/Cuda.hpp>
#include <femx/common/Device.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/cuda/CudaHandles.hpp>

namespace femx::linalg
{

#if !defined(FEMX_HAS_CUDA)
namespace detail
{

CudaHandles::CudaHandles(void*)
{
}

CudaHandles::~CudaHandles() = default;

std::unique_ptr<CudaHandles> makeCudaHandles(void* stream)
{
  return std::make_unique<CudaHandles>(stream);
}

} // namespace detail
#endif

CudaContext::CudaContext()
  : stream_(cuda::createStream()),
    vec_handler_(*this)
{
#if defined(FEMX_HAS_CUDA)
  try
  {
    handles_ = detail::makeCudaHandles(stream_);
  }
  catch (...)
  {
    cuda::destroyStream(stream_);
    stream_ = nullptr;
    throw;
  }
#endif
}

CudaContext::~CudaContext()
{
  sparse_state_.reset();
  handles_.reset();
  cuda::destroyStream(stream_);
}

VectorHandler<MemorySpace::Device>& CudaContext::vectorHandler() noexcept
{
  return vec_handler_;
}

IndexRange CudaContext::elementRange(Index count) const
{
  require(count >= 0,
          "CudaContext element count must be nonnegative");
  return {0, count};
}

void CudaContext::sync() const
{
  device::sync(stream_);
}

void* CudaContext::stream() const noexcept
{
  return stream_;
}

bool CudaContext::available() noexcept
{
  return cuda::available();
}

CudaVectorHandler::CudaVectorHandler(CudaContext& ctx) noexcept
  : ctx_(ctx)
{
}

void* CudaVectorHandler::stream() const noexcept
{
  return ctx_.stream();
}

} // namespace femx::linalg
