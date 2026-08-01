#pragma once

#include <cstddef>

#include <femx/common/Types.hpp>

namespace femx::device
{

/**
 * @brief Allocate storage with the configured Device backend.
 *
 * @param[in] bytes - Number of bytes to allocate.
 * @return Device pointer, or `nullptr` when `bytes` is zero.
 * @throws - If no Device backend is configured or allocation fails.
 */
void* allocate(std::size_t bytes);

/**
 * @brief Release storage allocated by the configured Device backend.
 *
 * @param[in] ptr - Device pointer, which may be `nullptr`.
 */
void release(void* ptr) noexcept;

/**
 * @brief Copy bytes between Host and configured Device storage.
 *
 * @param[out] dst - Destination pointer.
 * @param[in] dst_space - Destination memory space.
 * @param[in] src - Source pointer.
 * @param[in] src_space - Source memory space.
 * @param[in] bytes - Number of bytes to copy.
 * @param[in] stream - Backend stream, or `nullptr` for its default stream.
 * @throws - If no Device backend is configured or the copy fails.
 */
void copy(void*       dst,
          MemorySpace dst_space,
          const void* src,
          MemorySpace src_space,
          std::size_t bytes,
          void*       stream = nullptr);

/**
 * @brief Set Device storage bytes to zero.
 *
 * @param[out] ptr - Device pointer.
 * @param[in] bytes - Number of bytes to set.
 * @param[in] stream - Backend stream, or `nullptr` for its default stream.
 * @throws - If no Device backend is configured or the operation fails.
 */
void zero(void* ptr, std::size_t bytes, void* stream = nullptr);

/**
 * @brief Fill Device values with one scalar value.
 */
void fill(Real* ptr, Index size, Real val, void* stream = nullptr);

/**
 * @brief Fill Device index values with one scalar value.
 */
void fill(Index* ptr, Index size, Index val, void* stream = nullptr);

/**
 * @brief Wait for work submitted to a configured Device backend stream.
 *
 * @param[in] stream - Backend stream, or `nullptr` for its default stream.
 * @throws - If no Device backend is configured or synchronization fails.
 */
void sync(void* stream);

} // namespace femx::device
