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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_STORAGE_TDS_BACKEND_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_STORAGE_TDS_BACKEND_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/core/numa_thread_pool.h"
#include "tpu_sync/kv_cache/backends/backend.h"
#include "tpu_sync/kv_cache/backends/storage/posix_backend.h"

namespace tpu_raiden {
namespace kv_cache {
namespace backends {
namespace storage {

inline constexpr absl::string_view kTdsBackendName = "tds";

// Configuration options for TdsKVBackend.
//
// Controls how KV-cache blocks transfer between:
//   - Storage Tier: [local-storage: NVMe SSD] or [shared-storage: Lustre/NFS]
//   - Host Tier:    [dram: User host_buf] (bypassing [dram: OS Page Cache] and
//                   [cpu-cache] when `use_direct_io == true`).
struct TdsBackendOptions {
  std::string root_dir = "/tmp/raiden_storage";
  std::string model_name = "unknown";
  int tp_size = 1;
  int tp_rank = 0;
  size_t capacity_bytes = 0;
  size_t lookup_batch_size = kDefaultLookupBatchSize;
  int storage_io_thread_pool_size = 16;
  size_t alignment = 4096;
  bool use_direct_io = true;
  // When false (default for Phase 1a O_DIRECT pwritev/preadv), any slice in
  // [dram: User host_buf] whose address, size, and file_offset are multiples of
  // `alignment` (4096B) takes the zero-copy O_DIRECT fast path without
  // requiring prior RegisterBuffer() calls. When true (used for strict pool
  // validation or Phase 1b io_uring fixed-buffer registration), slices must
  // also reside inside a region registered via RegisterBuffer().
  bool require_registration = false;
  size_t lustre_stripe_size = 0;
  int lustre_stripe_count = 0;

  static absl::StatusOr<TdsBackendOptions> FromProperties(
      const absl::flat_hash_map<std::string, std::string>& properties);
};

// Telemetry and I/O statistics for TdsKVBackend.
struct TdsStats {
  int64_t ops = 0;
  int64_t bytes = 0;
  // Operations transferred directly between [local-storage: NVMe SSD] /
  // [shared-storage: Lustre/NFS] and [dram: User host_buf] via O_DIRECT
  // vectored DMA (0 bytes in [dram: OS Page Cache], 0 CPU memcpy via
  // [cpu-cache]).
  int64_t direct_ops = 0;
  // Operations that required a 4KB-aligned bounce buffer in [dram: User
  // host_buf] because the caller slice or file offset was not 4KB-aligned (or
  // was unregistered when `require_registration == true`).
  int64_t bounced_ops = 0;
  int64_t failed_ops = 0;
};

// TdsKVBackend implements KVBackend for high-throughput transfers between
// [local-storage: NVMe SSD] / [shared-storage: Lustre/NFS] and [dram: User
// host_buf] (which Raiden's H2D/D2H engine then DMAs to/from [hbm]):
//
//   1. Zero-Copy O_DIRECT Vectored Fast Path (`direct_ops`):
//      [local-storage: NVMe SSD] / [shared-storage: Lustre/NFS]
//        <--(::pwritev / ::preadv with O_DIRECT)--> [dram: User host_buf]
//      Bypasses [dram: OS Page Cache] and [cpu-cache] completely.
//
//   2. 4KB Bounce-Buffer Fallback (`bounced_ops`):
//      Allocated via `posix_memalign(4096)` in [dram: User host_buf] when
//      caller slices or file offsets are unaligned, satisfying the upper-layer
//      bounce-buffering responsibility defined in `go/tdsul-api` (§1 & §3.3.1).
//
//   3. Atomic Whole-Block Publishing & Lustre Striping:
//      Writes at `offset == 0` stage into `<path>.tmp_<pid>_<seq>` (applying
//      `LL_IOC_LOV_SETSTRIPE` on [shared-storage: Lustre/NFS]) and atomically
//      `::rename()` into place so concurrent `BatchExistsAsync` lookups across
//      nodes never observe partial files.
class TdsKVBackend : public KVBackend {
 public:
  explicit TdsKVBackend(
      std::string name,
      absl::flat_hash_map<std::string, std::string> properties = {});

  TdsKVBackend(std::string name, const TdsBackendOptions& options,
               absl::flat_hash_map<std::string, std::string> properties = {});

  ~TdsKVBackend() override;

  std::string name() const override { return name_; }

  void WriteAsync(const BlockKey& key,
                  absl::Span<const HostBufferDescriptor> slices,
                  size_t total_bytes,
                  std::function<void(absl::Status)> callback) override;

  void ReadAsync(const BlockKey& key,
                 absl::Span<const HostBufferDescriptor> slices,
                 size_t total_bytes,
                 std::function<void(absl::Status)> callback) override;

  void BatchExistsAsync(
      absl::Span<const BlockKey> keys,
      std::function<void(std::vector<absl::StatusOr<bool>>)> callback) override;

  // Registers a [dram: User host_buf] region for zero-copy O_DIRECT DMA and
  // future io_uring fixed-buffer pinning (IORING_REGISTER_BUFFERS).
  absl::Status RegisterBuffer(void* ptr, size_t size);

  // Unregisters a previously registered [dram: User host_buf] region.
  absl::Status UnregisterBuffer(void* ptr);

  // Returns true if the buffer range [ptr, ptr + size) in [dram: User host_buf]
  // and `file_offset` satisfy hardware `alignment` (4096B) and reside within a
  // memory pool registered via RegisterBuffer().
  bool IsDmaAlignedAndRegistered(const void* ptr, size_t size,
                                 int64_t file_offset) const;

  TdsStats stats() const;

  void ResetStats();

  size_t alignment() const { return options_.alignment; }

  const TdsBackendOptions& options() const { return options_; }

 private:
  bool IsRegisteredLocked(const void* ptr, size_t size) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  bool CanUseDirectZeroCopy(
      int64_t file_offset, absl::Span<const HostBufferDescriptor> slices) const;

  absl::StatusOr<std::string> ResolvePath(const BlockKey& key) const;

  absl::StatusOr<bool> Exists(const BlockKey& key) const;

  void RecordSuccess(bool direct, size_t bytes);

  absl::StatusOr<std::string> ValidateAndResolve(
      const BlockKey& key, absl::Span<const HostBufferDescriptor> slices,
      size_t total_bytes) const;

  absl::Status ExecuteWrite(const BlockKey& key,
                            absl::Span<const HostBufferDescriptor> slices,
                            size_t total_bytes);

  absl::Status ExecuteRead(const BlockKey& key,
                           absl::Span<const HostBufferDescriptor> slices,
                           size_t total_bytes);

  std::string name_;
  TdsBackendOptions options_;
  mutable absl::Mutex mu_;
  std::map<uintptr_t, size_t> registrations_ ABSL_GUARDED_BY(mu_);
  TdsStats stats_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<NumaThreadPool> thread_pool_;
};

}  // namespace storage
}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden

namespace tkv {
namespace backends = ::tpu_raiden::kv_cache::backends;
}  // namespace tkv

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_STORAGE_TDS_BACKEND_H_
