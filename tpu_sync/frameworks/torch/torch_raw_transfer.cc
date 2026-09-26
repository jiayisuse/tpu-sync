// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tpu_sync/frameworks/torch/torch_raw_transfer.h"

#include <cstddef>
#include <cstdint>
#include <cerrno>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <linux/mempolicy.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/utils.h"
#include "tpu_sync/frameworks/torch/torch_tpu_utils.h"

namespace raiden {
namespace {

using TensorList = std::vector<at::Tensor>;

using ::tpu_raiden::torch::UnpackTorchTensor;

class ScopedNumaBind {
 public:
  explicit ScopedNumaBind(int numa_node) {
#if defined(__linux__) && defined(SYS_set_mempolicy)
    unsigned long mask = 1UL << numa_node;
    if (syscall(SYS_set_mempolicy, MPOL_BIND, &mask,
                sizeof(mask) * 8) != 0) {
      throw std::runtime_error(absl::StrCat(
          "Failed to bind staging allocation to NUMA node ", numa_node,
          ": ", std::strerror(errno)));
    }
    active_ = true;
#else
    (void)numa_node;
#endif
  }

  ~ScopedNumaBind() {
#if defined(__linux__) && defined(SYS_set_mempolicy)
    if (active_) {
      (void)syscall(SYS_set_mempolicy, MPOL_DEFAULT, nullptr, 0);
    }
#endif
  }

 private:
  bool active_ = false;
};
}  // namespace

RawHostBuffer::RawHostBuffer(int64_t size_bytes) {
  if (size_bytes < 0) {
    throw std::invalid_argument(
        "RawHostBuffer size_bytes must be non-negative");
  }
  size_bytes_ = static_cast<size_t>(size_bytes);
}

RawHostBuffer::~RawHostBuffer() {
  if (c_api_ != nullptr && c_client_ != nullptr && data_ptr_ != nullptr) {
    PJRT_Client_DmaUnmap_Args args;
    args.struct_size = PJRT_Client_DmaUnmap_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.client = c_client_;
    args.data = data_ptr_;
    PJRT_Error* error = c_api_->PJRT_Client_DmaUnmap(&args);
    if (error != nullptr) {
      (void)PjrtErrorToStatusLocal(c_api_, error);
    }
  }
#if defined(__linux__)
  if (mapped_size_bytes_ != 0 && data_ptr_ != nullptr) {
    (void)munmap(data_ptr_, mapped_size_bytes_);
  }
#endif
}

uintptr_t RawHostBuffer::DataPtr() const {
  return reinterpret_cast<uintptr_t>(data_ptr_);
}

void* RawHostBuffer::MutableData() const { return data_ptr_; }

const void* RawHostBuffer::Data() const { return data_ptr_; }

size_t RawHostBuffer::SizeBytes() const { return size_bytes_; }

bool RawHostBuffer::IsPjRtBacked() const {
  return c_client_ != nullptr || host_tensor_.defined();
}

void RawHostBuffer::EnsureBoundToDevice(xla::PjRtDevice* device) {
  if (data_ptr_ != nullptr || size_bytes_ == 0) {
    return;
  }
  if (device == nullptr) {
    throw std::invalid_argument("Cannot bind RawHostBuffer to null device");
  }
  // TPU v7x staging DMA is local to NUMA node 0 on the serving hosts. Match
  // XlaHostMemoryAllocator's placement while staying on the stable ATen ABI.
  ScopedNumaBind numa_policy(/*numa_node=*/0);

  // Allocate through Torch-Tpu's registered pinned-CPU allocator.  The
  // standalone extension is built against an XLA source snapshot, while the
  // serving wheel owns the live PJRT client; calling that client's evolving
  // C++ virtual allocation API from this DSO is not ABI-safe.  ATen's tensor
  // ABI is the stable boundary already used by the serving runtime, and the
  // tensor retained here owns the registered host allocation for every DMA.
  host_tensor_ = at::empty(
      {static_cast<int64_t>(size_bytes_)},
      at::TensorOptions()
          .dtype(at::ScalarType::Byte)
          .device(at::DeviceType::CPU)
          .pinned_memory(true));
  if (!host_tensor_.defined() || host_tensor_.data_ptr() == nullptr) {
    throw std::runtime_error("Failed to allocate Torch-TPU pinned host buffer");
  }
  data_ptr_ = host_tensor_.data_ptr();

  // Match XlaHostMemoryAllocator's one-time first-touch behavior.  at::empty
  // reserves the pinned arena but leaves its pages physically uncommitted;
  // without this pass the first request to use each staging range pays page
  // faults on the latency-critical D2H/H2D path.  Touch one byte per page so
  // those faults are absorbed during initialization instead.
  volatile uint8_t* pages = static_cast<volatile uint8_t*>(data_ptr_);
  constexpr size_t kPageSize = 4096;
  for (size_t offset = 0; offset < size_bytes_; offset += kPageSize) {
    pages[offset] = 0;
  }
}

void RawHostBuffer::EnsureDmaMappedToDevice(xla::PjRtDevice* device) {
  EnsureBoundToDevice(device);
}

void RawHostBuffer::EnsureDmaMappedToBuffer(xla::PjRtBuffer* buffer) {
  if (data_ptr_ != nullptr || size_bytes_ == 0) {
    return;
  }
  if (buffer == nullptr) {
    throw std::invalid_argument("Cannot DMA-map RawHostBuffer for null buffer");
  }

  auto handles_or = GetCApiClientHandles(buffer);
  if (!handles_or.ok()) {
    // Retain a functional non-C-API fallback for CPU/unit-test buffers.
    EnsureBoundToDevice(buffer->device());
    return;
  }

#if defined(__linux__)
  constexpr size_t kPageSize = 4096;
  mapped_size_bytes_ = (size_bytes_ + kPageSize - 1) & ~(kPageSize - 1);
  ScopedNumaBind numa_policy(/*numa_node=*/0);
  data_ptr_ = mmap(nullptr, mapped_size_bytes_, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (data_ptr_ == MAP_FAILED) {
    data_ptr_ = nullptr;
    mapped_size_bytes_ = 0;
    throw std::runtime_error(absl::StrCat(
        "mmap failed for DMA staging buffer: ", std::strerror(errno)));
  }

  c_api_ = handles_or->api;
  c_client_ = handles_or->client;
  PJRT_Client_DmaMap_Args args;
  args.struct_size = PJRT_Client_DmaMap_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.client = c_client_;
  args.data = data_ptr_;
  args.size = mapped_size_bytes_;
  if (PJRT_Error* error = c_api_->PJRT_Client_DmaMap(&args); error != nullptr) {
    absl::Status status = PjrtErrorToStatusLocal(c_api_, error);
    (void)munmap(data_ptr_, mapped_size_bytes_);
    data_ptr_ = nullptr;
    mapped_size_bytes_ = 0;
    c_api_ = nullptr;
    c_client_ = nullptr;
    throw std::runtime_error(
        absl::StrCat("PJRT C-API DmaMap failed: ", status.message()));
  }

  // Match XlaHostMemoryAllocator: register first, then fault every page while
  // the allocation thread is bound to the TPU-local NUMA node.
  volatile uint8_t* pages = static_cast<volatile uint8_t*>(data_ptr_);
  for (size_t offset = 0; offset < mapped_size_bytes_; offset += kPageSize) {
    pages[offset] = 0;
  }
#else
  EnsureBoundToDevice(buffer->device());
#endif
}

namespace {
[[noreturn]] void ThrowStatus(absl::string_view context,
                              const absl::Status& status) {
  throw std::runtime_error(absl::StrCat(context, ": ", status.message()));
}

template <typename T>
T ValueOrThrow(absl::string_view context, absl::StatusOr<T> value_or) {
  if (!value_or.ok()) {
    ThrowStatus(context, value_or.status());
  }
  return std::move(value_or).value();
}

void ValidateCpuTensor(const at::Tensor& tensor, absl::string_view role) {
  if (!tensor.device().is_cpu()) {
    throw std::invalid_argument(absl::StrCat(role, " must be a CPU tensor"));
  }
  if (!tensor.is_contiguous()) {
    throw std::invalid_argument(absl::StrCat(role, " must be contiguous"));
  }
}

void AwaitReady(xla::PjRtBuffer* buffer, absl::string_view role) {
  (void)buffer;
  (void)role;
}

// Raw transfer addresses the device buffer as a flat array of equal-size
// major-dimension slices ("blocks"): block i lives at byte offset
// i * GetMajorSliceByteSize(shape). That mapping is only correct when logical
// dimension 0 is the most-major physical dimension and the buffer's physical
// size is an exact multiple of the slice size (the blocks tile it with no
// remainder). Assert both so a buffer with an unexpected on-device layout fails
// loudly here instead of silently transferring the wrong bytes.
void ValidateMajorDimLayout(const RaidenBufferHandle& buffer,
                            absl::string_view role) {
  const xla::Shape& shape = buffer.shape;
  const int rank = shape.dimensions().size();
  if (rank < 1) {
    throw std::invalid_argument(
        absl::StrCat(role, " buffer must have rank >= 1 for block transfer"));
  }
  // In xla::Layout, minor_to_major(rank - 1) is the most-major physical dim.
  if (buffer.buffer && buffer.buffer->layout() &&
      buffer.buffer->layout()->xla_layout().minor_to_major(rank - 1) != 0) {
    throw std::invalid_argument(
        absl::StrCat(role,
                     " buffer layout must place logical dimension 0 as the "
                     "most-major "
                     "physical dimension; block offsetting assumes blocks are "
                     "the "
                     "outermost, physically contiguous dimension."));
  }
  const int64_t slice = GetMajorSliceByteSize(shape);
  // Fallback to buffer->GetOnDeviceSizeInBytes() if available, but for now we
  // might not have it easily without deprecated buffer pointer if it's not
  // cached in handle.
  // Assuming shape gives enough info or buffer pointer is available as fallback
  // in handle for now.
  const int64_t physical_size =
      buffer.buffer
          ? ValueOrThrow(
                absl::StrCat(role, " physical buffer size for layout check"),
                buffer.buffer->GetOnDeviceSizeInBytes())
          : xla::ShapeUtil::ByteSizeOf(shape);  // Fallback

  if (slice <= 0 || physical_size % slice != 0) {
    throw std::invalid_argument(
        absl::StrCat(role,
                     " buffer physical size is not an exact multiple of its "
                     "major-dimension "
                     "slice size; the block-layout assumption does not hold."));
  }
}

PjRtCopyFuture IssueD2HCopy(const RaidenBufferHandle& src_buffer,
                            uint8_t* dst_data, size_t dst_size,
                            const std::vector<int64_t>& src_offsets_major_dim,
                            const std::vector<int64_t>& dst_offsets_major_dim,
                            const std::vector<int64_t>& copy_sizes_major_dim,
                            std::shared_ptr<void> user_hold = nullptr,
                            std::optional<int64_t> prepared_physical_size =
                                std::nullopt,
                            std::optional<int64_t> prepared_slice_byte_size =
                                std::nullopt) {
  if (!prepared_physical_size.has_value() ||
      !prepared_slice_byte_size.has_value()) {
    ValidateMajorDimLayout(src_buffer, "Source");
  }
  const bool is_partial =
      tpu_raiden::IsPartialCopy(src_buffer.shape, src_offsets_major_dim,
                                dst_offsets_major_dim, copy_sizes_major_dim);
  const int64_t physical_size = prepared_physical_size.has_value()
                                    ? *prepared_physical_size
                                    : src_buffer.buffer
                                          ? ValueOrThrow(
                                                "Failed to get source physical buffer size",
                                                src_buffer.buffer
                                                    ->GetOnDeviceSizeInBytes())
                                          : xla::ShapeUtil::ByteSizeOf(
                                                src_buffer.shape);
  const int64_t slice_byte_size = prepared_slice_byte_size.has_value()
                                      ? *prepared_slice_byte_size
                                      : GetMajorSliceByteSize(src_buffer.shape);

  if (is_partial) {
    tpu_raiden::ValidatePartialAlignment(src_buffer.shape, slice_byte_size);
  }

  std::vector<tpu_raiden::RawCopyChunk> chunks =
      tpu_raiden::ComputeAndValidateChunks(
          slice_byte_size, physical_size, dst_size, is_partial,
          src_offsets_major_dim, dst_offsets_major_dim, copy_sizes_major_dim,
          /*is_d2h=*/true);

  std::vector<xla::Future<>> futures;
  futures.reserve(chunks.size());
  for (const auto& chunk : chunks) {
    futures.push_back(src_buffer.CopyRawDeviceToHost(
        dst_data + chunk.dst_offset, chunk.src_offset, chunk.size_bytes));
  }
  return PjRtCopyFuture(
      xla::JoinFutures(absl::MakeSpan(futures)),
      {BufferHolder{src_buffer.c_hold, src_buffer.common_hold,
                    /*ext_hold=*/nullptr, std::move(user_hold)}});
}

PjRtCopyFuture IssueH2DCopy(const uint8_t* src_data, size_t src_size,
                            const RaidenBufferHandle& dst_buffer,
                            const std::vector<int64_t>& src_offsets_major_dim,
                            const std::vector<int64_t>& dst_offsets_major_dim,
                            const std::vector<int64_t>& copy_sizes_major_dim,
                            std::shared_ptr<void> user_hold = nullptr,
                            std::optional<int64_t> prepared_physical_size =
                                std::nullopt,
                            std::optional<int64_t> prepared_slice_byte_size =
                                std::nullopt) {
  if (!prepared_physical_size.has_value() ||
      !prepared_slice_byte_size.has_value()) {
    ValidateMajorDimLayout(dst_buffer, "Destination");
  }
  const bool is_partial =
      tpu_raiden::IsPartialCopy(dst_buffer.shape, src_offsets_major_dim,
                                dst_offsets_major_dim, copy_sizes_major_dim);
  const int64_t physical_size = prepared_physical_size.has_value()
                                    ? *prepared_physical_size
                                    : dst_buffer.buffer
                                          ? ValueOrThrow(
                                                "Failed to get destination physical buffer size",
                                                dst_buffer.buffer
                                                    ->GetOnDeviceSizeInBytes())
                                          : xla::ShapeUtil::ByteSizeOf(
                                                dst_buffer.shape);
  const int64_t slice_byte_size = prepared_slice_byte_size.has_value()
                                      ? *prepared_slice_byte_size
                                      : GetMajorSliceByteSize(dst_buffer.shape);

  if (is_partial) {
    tpu_raiden::ValidatePartialAlignment(dst_buffer.shape, slice_byte_size);
  }

  std::vector<tpu_raiden::RawCopyChunk> chunks =
      tpu_raiden::ComputeAndValidateChunks(
          slice_byte_size, physical_size, src_size, is_partial,
          src_offsets_major_dim, dst_offsets_major_dim, copy_sizes_major_dim,
          /*is_d2h=*/false);

  std::vector<xla::Future<>> futures;
  futures.reserve(chunks.size());
  for (const auto& chunk : chunks) {
    futures.push_back(dst_buffer.CopyRawHostToDevice(
        src_data + chunk.src_offset, chunk.dst_offset, chunk.size_bytes));
  }
  return PjRtCopyFuture(
      xla::JoinFutures(absl::MakeSpan(futures)),
      {BufferHolder{dst_buffer.c_hold, dst_buffer.common_hold,
                    /*ext_hold=*/nullptr, std::move(user_hold)}});
}
}  // namespace

PjRtCopyFuture TransferD2HBatchAsync(
    const TensorList& src_arrs, const TensorList& dst_arrs,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  if (src_arrs.size() != dst_arrs.size()) {
    throw std::invalid_argument("Lengths of src_arrs and dst_arrs must match");
  }
  tpu_raiden::ValidatePartialSpec(src_offsets_major_dim, dst_offsets_major_dim,
                                  copy_sizes_major_dim);
  std::vector<PjRtCopyFuture> futures;
  futures.reserve(src_arrs.size());
  for (size_t i = 0; i < src_arrs.size(); ++i) {
    ValidateCpuTensor(dst_arrs[i], "Destination");
    auto unpacked = UnpackTorchTensor(src_arrs[i]);
    const RaidenBufferHandle& src_buffer = unpacked.buffer;

    auto torch_holds = std::make_shared<std::vector<at::Tensor>>();
    torch_holds->push_back(src_arrs[i]);
    torch_holds->push_back(dst_arrs[i]);

    auto fut = IssueD2HCopy(
        src_buffer, reinterpret_cast<uint8_t*>(dst_arrs[i].data_ptr()),
        dst_arrs[i].nbytes(), src_offsets_major_dim, dst_offsets_major_dim,
        copy_sizes_major_dim, std::move(torch_holds));
    // Keep the materialized (possibly view) buffer alive until the copy is
    // done.
    if (unpacked.ref) {
      fut.AddKeepAlive(std::make_shared<torch_tpu::DeviceBufferRef>(
          std::move(*unpacked.ref)));
    }
    futures.push_back(std::move(fut));
  }
  return JoinPjRtCopyFutures(absl::MakeSpan(futures));
}

PjRtCopyFuture TransferH2DBatchAsync(
    const TensorList& src_arrs, const TensorList& dst_arrs,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  if (src_arrs.size() != dst_arrs.size()) {
    throw std::invalid_argument("Lengths of src_arrs and dst_arrs must match");
  }
  tpu_raiden::ValidatePartialSpec(src_offsets_major_dim, dst_offsets_major_dim,
                                  copy_sizes_major_dim);
  std::vector<PjRtCopyFuture> futures;
  futures.reserve(src_arrs.size());
  for (size_t i = 0; i < src_arrs.size(); ++i) {
    ValidateCpuTensor(src_arrs[i], "Source");
    auto unpacked = UnpackTorchTensor(dst_arrs[i]);
    const RaidenBufferHandle& dst_buffer = unpacked.buffer;

    auto torch_holds = std::make_shared<std::vector<at::Tensor>>();
    torch_holds->push_back(src_arrs[i]);
    torch_holds->push_back(dst_arrs[i]);

    auto fut = IssueH2DCopy(
        reinterpret_cast<const uint8_t*>(src_arrs[i].data_ptr()),
        src_arrs[i].nbytes(), dst_buffer, src_offsets_major_dim,
        dst_offsets_major_dim, copy_sizes_major_dim, std::move(torch_holds));
    // Keep the materialized (possibly view) buffer alive until the copy is
    // done.
    if (unpacked.ref) {
      fut.AddKeepAlive(std::make_shared<torch_tpu::DeviceBufferRef>(
          std::move(*unpacked.ref)));
    }
    futures.push_back(std::move(fut));
  }
  return JoinPjRtCopyFutures(absl::MakeSpan(futures));
}

PjRtCopyFuture TransferD2HAsync(
    const at::Tensor& src_arr, const at::Tensor& dst_arr,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  return TransferD2HBatchAsync({src_arr}, {dst_arr}, src_offsets_major_dim,
                               dst_offsets_major_dim, copy_sizes_major_dim);
}

PjRtCopyFuture TransferH2DAsync(
    const at::Tensor& src_arr, const at::Tensor& dst_arr,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  return TransferH2DBatchAsync({src_arr}, {dst_arr}, src_offsets_major_dim,
                               dst_offsets_major_dim, copy_sizes_major_dim);
}

PreparedTorchRawTransfer::PreparedTorchRawTransfer(
    const at::Tensor& tpu_tensor, std::shared_ptr<RawHostBuffer> host_buffer,
    bool unsafe_skip_buffer_lock)
    : host_buffer_(std::move(host_buffer)) {
  if (!host_buffer_) {
    throw std::invalid_argument("host_buffer must not be None");
  }
  auto unpacked = UnpackTorchTensor(tpu_tensor, unsafe_skip_buffer_lock);
  buffer_ = std::move(unpacked.buffer);
  buffer_ref_ = std::move(unpacked.ref);  // keep the materialized buffer alive
  host_buffer_->EnsureDmaMappedToBuffer(buffer_.buffer);
  physical_size_ = static_cast<size_t>(
      ValueOrThrow("Failed to get TPU physical buffer size",
                   buffer_.buffer ? buffer_.buffer->GetOnDeviceSizeInBytes()
                                  : xla::ShapeUtil::ByteSizeOf(buffer_.shape)));
  if (host_buffer_->SizeBytes() < physical_size_) {
    throw std::invalid_argument(
        "RawHostBuffer is smaller than TPU physical size");
  }
}

size_t PreparedTorchRawTransfer::PhysicalSizeBytes() const {
  return physical_size_;
}

std::shared_ptr<RawHostBuffer> PreparedTorchRawTransfer::HostBuffer() const {
  return host_buffer_;
}

PjRtCopyFuture PreparedTorchRawTransfer::D2HAsync() {
  xla::Future<> copy_future = buffer_.CopyRawDeviceToHost(
      host_buffer_->MutableData(), 0, physical_size_);
  return PjRtCopyFuture(
      std::move(copy_future),
      {BufferHolder{buffer_.c_hold, buffer_.common_hold, /*ext_hold=*/nullptr,
                    shared_from_this()}});
}

PjRtCopyFuture PreparedTorchRawTransfer::H2DAsync() {
  xla::Future<> copy_future =
      buffer_.CopyRawHostToDevice(host_buffer_->Data(), 0, physical_size_);
  return PjRtCopyFuture(
      std::move(copy_future),
      {BufferHolder{buffer_.c_hold, buffer_.common_hold, /*ext_hold=*/nullptr,
                    shared_from_this()}});
}

void PreparedTorchRawTransfer::D2H() {
  PjRtCopyFuture future = D2HAsync();
  absl::Status status = future.Await();
  if (!status.ok()) {
    ThrowStatus("D2H copy failed", status);
  }
}

void PreparedTorchRawTransfer::H2D() {
  PjRtCopyFuture future = H2DAsync();
  absl::Status status = future.Await();
  if (!status.ok()) {
    ThrowStatus("H2D copy failed", status);
  }
}

PreparedTorchRawTransferBatch::PreparedTorchRawTransferBatch(
    const TensorList& tpu_tensors,
    const std::vector<int64_t>& host_buffer_sizes_bytes,
    bool unsafe_skip_buffer_lock)
    : unsafe_skip_buffer_lock_(unsafe_skip_buffer_lock) {
  if (tpu_tensors.empty()) {
    throw std::invalid_argument("tpu_tensors must not be empty");
  }
  if (tpu_tensors.size() != host_buffer_sizes_bytes.size()) {
    throw std::invalid_argument(
        "Lengths of tpu_tensors and host_buffer_sizes_bytes must match");
  }

  prepared_buffers_.reserve(tpu_tensors.size());
  host_buffers_.reserve(tpu_tensors.size());
  for (size_t i = 0; i < tpu_tensors.size(); ++i) {
    if (host_buffer_sizes_bytes[i] <= 0) {
      throw std::invalid_argument(
          "host_buffer_sizes_bytes must contain only positive values");
    }

    const char* stage = "unpack TPU tensor";
    try {
      // Match the proven KVCacheManager lifetime model: retain the owning
      // DeviceBufferRef, but do not retain an additional at::Tensor alias.
      auto unpacked =
          UnpackTorchTensor(tpu_tensors[i], /*unsafe_skip_buffer_lock=*/true);
      stage = "construct host-buffer owner";
      auto host_buffer =
          std::make_shared<RawHostBuffer>(host_buffer_sizes_bytes[i]);
      stage = "allocate and C-API DMA-map host buffer";
      // Duplicate XlaHostMemoryAllocator through the stable PJRT C ABI. This
      // avoids the independently linked PjRtClient C++ vtable while preserving
      // the proven mmap + explicit DmaMap staging semantics.
      host_buffer->EnsureDmaMappedToBuffer(unpacked.buffer.buffer);
      stage = "validate and cache prepared buffer geometry";
      ValidateMajorDimLayout(unpacked.buffer, "Prepared TPU buffer");
      const size_t prepared_physical_size =
          unpacked.buffer.GetOnDeviceSizeInBytes();
      const size_t prepared_slice_byte_size =
          static_cast<size_t>(GetMajorSliceByteSize(unpacked.buffer.shape));
      stage = "retain prepared buffer";
      prepared_buffers_.push_back(PreparedBuffer{
          .buffer = std::move(unpacked.buffer),
          .buffer_ref = std::move(unpacked.ref),
          .physical_size = prepared_physical_size,
          .slice_byte_size = prepared_slice_byte_size,
      });
      stage = "retain host buffer";
      host_buffers_.push_back(std::move(host_buffer));
    } catch (const std::bad_alloc&) {
      throw std::runtime_error(absl::StrCat(
          "Prepared raw DMA batch ran out of memory while attempting to ",
          stage, "; buffer_index=", i,
          " host_bytes=", host_buffer_sizes_bytes[i],
          " batch_buffers=", tpu_tensors.size()));
    }
  }
}

size_t PreparedTorchRawTransferBatch::Size() const {
  return prepared_buffers_.size();
}

std::vector<size_t> PreparedTorchRawTransferBatch::PhysicalSizeBytes() const {
  std::vector<size_t> sizes;
  sizes.reserve(prepared_buffers_.size());
  for (const auto& prepared : prepared_buffers_) {
    sizes.push_back(prepared.physical_size);
  }
  return sizes;
}

std::vector<uintptr_t> PreparedTorchRawTransferBatch::HostDataPtrs() const {
  std::vector<uintptr_t> pointers;
  pointers.reserve(host_buffers_.size());
  for (const auto& host_buffer : host_buffers_) {
    pointers.push_back(host_buffer->DataPtr());
  }
  return pointers;
}

std::vector<size_t> PreparedTorchRawTransferBatch::HostSizeBytes() const {
  std::vector<size_t> sizes;
  sizes.reserve(host_buffers_.size());
  for (const auto& host_buffer : host_buffers_) {
    sizes.push_back(host_buffer->SizeBytes());
  }
  return sizes;
}

RaidenBufferHandle PreparedTorchRawTransferBatch::BufferForCopy(
    const PreparedBuffer& prepared) const {
  if (unsafe_skip_buffer_lock_) {
    return prepared.buffer;
  }
  if (prepared.buffer.buffer == nullptr) {
    throw std::runtime_error(
        "Prepared TPU buffer cannot acquire a per-copy usage hold");
  }
  return ValueOrThrow(
      "Failed to acquire TPU buffer for prepared copy",
      RaidenBufferHandle::Acquire(prepared.buffer.buffer,
                                  /*c_api=*/nullptr,
                                  /*extension=*/nullptr,
                                  /*unsafe_skip_buffer_lock=*/false));
}

PjRtCopyFuture PreparedTorchRawTransferBatch::D2HAsync(
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  tpu_raiden::ValidatePartialSpec(src_offsets_major_dim, dst_offsets_major_dim,
                                  copy_sizes_major_dim);
  std::vector<PjRtCopyFuture> futures;
  futures.reserve(prepared_buffers_.size());
  auto self = shared_from_this();
  for (size_t i = 0; i < prepared_buffers_.size(); ++i) {
    RaidenBufferHandle buffer = BufferForCopy(prepared_buffers_[i]);
    futures.push_back(IssueD2HCopy(
        buffer, static_cast<uint8_t*>(host_buffers_[i]->MutableData()),
        host_buffers_[i]->SizeBytes(), src_offsets_major_dim,
        dst_offsets_major_dim, copy_sizes_major_dim, self,
        static_cast<int64_t>(prepared_buffers_[i].physical_size),
        static_cast<int64_t>(prepared_buffers_[i].slice_byte_size)));
  }
  return JoinPjRtCopyFutures(absl::MakeSpan(futures));
}

PjRtCopyFuture PreparedTorchRawTransferBatch::H2DAsync(
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  tpu_raiden::ValidatePartialSpec(src_offsets_major_dim, dst_offsets_major_dim,
                                  copy_sizes_major_dim);
  std::vector<PjRtCopyFuture> futures;
  futures.reserve(prepared_buffers_.size());
  auto self = shared_from_this();
  for (size_t i = 0; i < prepared_buffers_.size(); ++i) {
    RaidenBufferHandle buffer = BufferForCopy(prepared_buffers_[i]);
    futures.push_back(IssueH2DCopy(
        static_cast<const uint8_t*>(host_buffers_[i]->Data()),
        host_buffers_[i]->SizeBytes(), buffer, src_offsets_major_dim,
        dst_offsets_major_dim, copy_sizes_major_dim, self,
        static_cast<int64_t>(prepared_buffers_[i].physical_size),
        static_cast<int64_t>(prepared_buffers_[i].slice_byte_size)));
  }
  return JoinPjRtCopyFutures(absl::MakeSpan(futures));
}

}  // namespace raiden
