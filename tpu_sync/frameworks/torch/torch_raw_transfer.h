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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_TORCH_RAW_TRANSFER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_TORCH_RAW_TRANSFER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "csrc/api/tensor_buffer.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "xla/pjrt/pjrt_client.h"

namespace raiden {

class RawHostBuffer {
 public:
  explicit RawHostBuffer(int64_t size_bytes);
  ~RawHostBuffer() = default;

  uintptr_t DataPtr() const;
  void* MutableData() const;
  const void* Data() const;
  size_t SizeBytes() const;
  bool IsPjRtBacked() const;
  void EnsureBoundToDevice(xla::PjRtDevice* device);

 private:
  size_t size_bytes_ = 0;
  void* data_ptr_ = nullptr;
  std::unique_ptr<xla::PjRtBuffer> pjrt_buffer_;
  c10::DataPtr data_;
};

class PreparedTorchRawTransfer
    : public std::enable_shared_from_this<PreparedTorchRawTransfer> {
 public:
  PreparedTorchRawTransfer(const at::Tensor& tpu_tensor,
                           std::shared_ptr<RawHostBuffer> host_buffer,
                           bool unsafe_skip_buffer_lock);

  size_t PhysicalSizeBytes() const;
  std::shared_ptr<RawHostBuffer> HostBuffer() const;

  PjRtCopyFuture D2HAsync();
  PjRtCopyFuture H2DAsync();
  void D2H();
  void H2D();

 private:
  std::shared_ptr<RawHostBuffer> host_buffer_;
  // Pins the base storage buffer behind `buffer_` for this object's lifetime.
  std::optional<torch_tpu::TensorBufferHandle> buffer_ref_;
  size_t physical_size_ = 0;
  RaidenBufferHandle buffer_;
};

// Prepared partial-copy engine for a fixed set of TPU and host tensors.
//
// Unlike Transfer{D2H,H2D}BatchAsync, construction resolves and materializes
// every TPU tensor exactly once. Subsequent copies reuse the retained base
// storage references and raw PJRT buffer aliases, which is important for hot
// KV-cache paths that submit many small partial transfers against the same
// long-lived tensors.
//
// Host tensors are retained for this object's lifetime and must be contiguous
// CPU tensors. Pinned tensors avoid libtpu's pageable-memory staging fallback.
// When unsafe_skip_buffer_lock is false, a fresh PJRT usage hold is acquired
// for each submitted copy and retained by its future until completion. When it
// is true, only the owning tensor-buffer reference is retained.
class PreparedTorchRawTransferBatch
    : public std::enable_shared_from_this<PreparedTorchRawTransferBatch> {
 public:
  PreparedTorchRawTransferBatch(const std::vector<at::Tensor>& tpu_tensors,
                                const std::vector<at::Tensor>& host_tensors,
                                bool unsafe_skip_buffer_lock);

  size_t Size() const;
  std::vector<size_t> PhysicalSizeBytes() const;

  PjRtCopyFuture D2HAsync(const std::vector<int64_t>& src_offsets_major_dim,
                          const std::vector<int64_t>& dst_offsets_major_dim,
                          const std::vector<int64_t>& copy_sizes_major_dim);
  PjRtCopyFuture H2DAsync(const std::vector<int64_t>& src_offsets_major_dim,
                          const std::vector<int64_t>& dst_offsets_major_dim,
                          const std::vector<int64_t>& copy_sizes_major_dim);

 private:
  struct PreparedBuffer {
    RaidenBufferHandle buffer;
    std::optional<torch_tpu::TensorBufferHandle> buffer_ref;
    size_t physical_size = 0;
    size_t slice_byte_size = 0;
    int64_t major_dim_size = 0;
  };

  RaidenBufferHandle BufferForCopy(const PreparedBuffer& prepared) const;

  std::vector<PreparedBuffer> prepared_buffers_;
  std::vector<at::Tensor> tpu_tensors_;
  std::vector<at::Tensor> host_tensors_;
  bool unsafe_skip_buffer_lock_ = false;
};

// Raw device<->host DMA over torch tensors, mirroring the JAX-side
// _raw_transfer surface. Each call materializes the TPU tensor's PjRtBuffer
// and awaits its ready future before issuing the copy (the raw PJRT copy API
// does not chain on buffer readiness), so a transfer issued right after the
// producing op reads committed data. Later torch ops on the same tensor
// produce new device buffers; re-issue the transfer (or re-create a
// PreparedTorchRawTransfer) to observe them. Host tensors must be CPU,
// contiguous, and SHOULD be pinned (pin_memory=True or RawHostBuffer):
// pageable memory silently degrades to staged copies — a warning is logged,
// and TPU_RAIDEN_RAW_REQUIRE_PINNED_HOST=1 turns it into an error.
// `unsafe_skip_buffer_lock` skips the dynamic safety locking of the device
// buffer for the duration of the copy, matching the JAX bindings' parameter.
PjRtCopyFuture TransferD2HBatchAsync(
    const std::vector<at::Tensor>& src_arrs,
    const std::vector<at::Tensor>& dst_arrs,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim,
    bool unsafe_skip_buffer_lock = false);

PjRtCopyFuture TransferH2DBatchAsync(
    const std::vector<at::Tensor>& src_arrs,
    const std::vector<at::Tensor>& dst_arrs,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim,
    bool unsafe_skip_buffer_lock = false);

PjRtCopyFuture TransferD2HAsync(
    const at::Tensor& src_arr, const at::Tensor& dst_arr,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim,
    bool unsafe_skip_buffer_lock = false);

PjRtCopyFuture TransferH2DAsync(
    const at::Tensor& src_arr, const at::Tensor& dst_arr,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim,
    bool unsafe_skip_buffer_lock = false);

}  // namespace raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_TORCH_RAW_TRANSFER_H_
