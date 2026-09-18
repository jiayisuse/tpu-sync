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

#include "tpu_sync/kv_cache/backends/storage/tds_backend.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/logging.h"
#include "tpu_sync/core/controller/raiden_controller.h"
#include "tpu_sync/core/numa_thread_pool.h"
#include "tpu_sync/kv_cache/backends/backend.h"
#include "tpu_sync/kv_cache/backends/storage/posix_backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"

namespace tpu_raiden {
namespace kv_cache {
namespace backends {
namespace storage {

namespace {

// Lustre user-space stripe descriptor for `ioctl(fd, LL_IOC_LOV_SETSTRIPE)`
// on [shared-storage: Lustre/NFS].
struct LustreLovUserMdV1 {
  uint32_t lmm_magic;
  uint32_t lmm_pattern;
  uint64_t lmm_object_id;
  uint64_t lmm_object_seq;
  uint32_t lmm_stripe_size;
  uint16_t lmm_stripe_count;
  uint16_t lmm_stripe_offset;
};

constexpr uint32_t kLovUserMagicV1 = 0x0BD10BD0;
constexpr uint32_t kLovPatternRaid0 = 0x00000001;

#ifndef LL_IOC_LOV_SETSTRIPE
#define LL_IOC_LOV_SETSTRIPE _IOW('f', 154, long)
#endif

// Lustre flag (`O_LOV_DELAY_CREATE`) instructing the Lustre Metadata Server
// (MDS) to defer OST object allocation on `open(O_CREAT)` until the subsequent
// `ioctl(fd, LL_IOC_LOV_SETSTRIPE)` call specifies the RAID-0 stripe geometry.
#ifndef O_LOV_DELAY_CREATE
#define O_LOV_DELAY_CREATE 0100000000
#endif

#ifndef UIO_MAXIOV
#define UIO_MAXIOV 1024
#endif

constexpr size_t kMaxIovCount = UIO_MAXIOV;

// Generates a unique sibling temporary path in the same directory as `path`
// so `::rename(tmp_path, path)` is guaranteed to be an atomic metadata operation
// on both [local-storage: NVMe SSD] and [shared-storage: Lustre/NFS].
std::string MakeUniqueTempPath(absl::string_view path) {
  static std::atomic<uint64_t> tmp_seq{0};
  return absl::StrCat(path, ".tmp_", ::getpid(), "_",
                      tmp_seq.fetch_add(1, std::memory_order_relaxed));
}

// Opens a file on [local-storage: NVMe SSD] or [shared-storage: Lustre/NFS].
//
// Performance optimizations:
// 1. Optimistic `::open()` first: avoids issuing redundant `stat`/`mkdir`
//    metadata RPCs to the [shared-storage: Lustre/NFS] Metadata Server (MDS)
//    when the target shard directory (`abc/d0/`) already exists; only creates
//    parent directories on `ENOENT`.
// 2. `O_DIRECT` page-cache bypass: when `use_direct_io == true`, sets `O_DIRECT`
//    so the kernel VFS bypasses [dram: OS Page Cache] and DMAs directly to/from
//    [dram: User host_buf], falling back cleanly if the test filesystem
//    (`tmpfs`) returns `EINVAL`.
// 3. `O_LOV_DELAY_CREATE` + `LL_IOC_LOV_SETSTRIPE`: defers OST allocation on
//    [shared-storage: Lustre/NFS] until the custom stripe count/size is set.
absl::StatusOr<int> OpenFile(const std::string& path, int flags, mode_t mode,
                             bool use_direct_io,
                             const TdsBackendOptions& options = {}) {
  const bool has_lustre_striping =
      (flags & O_CREAT) != 0 &&
      (options.lustre_stripe_count > 0 || options.lustre_stripe_size > 0);

  auto try_open_once = [&](int base_flags) -> int {
    const int open_flags = has_lustre_striping
                               ? (base_flags | O_LOV_DELAY_CREATE)
                               : base_flags;
    int fd = -1;
    if (use_direct_io) {
      fd = ::open(path.c_str(), open_flags | O_DIRECT, mode);
      if (fd < 0 && errno == EINVAL) {
        fd = ::open(path.c_str(), base_flags & ~O_DIRECT, mode);
      }
    } else {
      fd = ::open(path.c_str(), open_flags & ~O_DIRECT, mode);
      if (fd < 0 && errno == EINVAL && has_lustre_striping) {
        fd = ::open(path.c_str(), base_flags & ~O_DIRECT, mode);
      }
    }
    return fd;
  };

  int fd = try_open_once(flags);
  if (fd < 0 && errno == ENOENT && (flags & O_CREAT) != 0) {
    std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
      std::error_code ec;
      std::filesystem::create_directories(parent, ec);
      if (ec && !std::filesystem::exists(parent, ec)) {
        return absl::InternalError(
            absl::StrCat("Failed to create parent directories for ", path, ": ",
                         ec.message()));
      }
      fd = try_open_once(flags);
    }
  }

  if (fd < 0) {
    return absl::ErrnoToStatus(errno,
                               absl::StrCat("Failed to open file: ", path));
  }

  if (has_lustre_striping) {
    LustreLovUserMdV1 lum = {};
    lum.lmm_magic = kLovUserMagicV1;
    lum.lmm_pattern = kLovPatternRaid0;
    lum.lmm_stripe_size = static_cast<uint32_t>(options.lustre_stripe_size);
    lum.lmm_stripe_count = static_cast<uint16_t>(options.lustre_stripe_count);
    lum.lmm_stripe_offset = static_cast<uint16_t>(-1);
    if (::ioctl(fd, LL_IOC_LOV_SETSTRIPE, &lum) < 0) {
      VLOG(1) << "Optional LL_IOC_LOV_SETSTRIPE ioctl on " << path
              << " returned: " << std::strerror(errno);
    }
  }

  return fd;
}

// Performs a vectored scatter-gather transfer (`::pwritev` when `is_write` is
// true, `::preadv` when false) directly between 4KB-aligned slices in
// [dram: User host_buf] and `fd` on [local-storage: NVMe SSD] or
// [shared-storage: Lustre/NFS].
absl::Status ExecuteVectoredIo(int fd,
                               absl::Span<const HostBufferDescriptor> slices,
                               int64_t file_offset, bool is_write) {
  std::vector<struct iovec> iov;
  iov.reserve(slices.size());
  for (const auto& slice : slices) {
    if (slice.size > 0) {
      iov.push_back(iovec{.iov_base = slice.ptr, .iov_len = slice.size});
    }
  }

  size_t idx = 0;
  int64_t current_offset = file_offset;
  while (idx < iov.size()) {
    const int iov_cnt =
        static_cast<int>(std::min<size_t>(iov.size() - idx, kMaxIovCount));
    ssize_t n = is_write ? ::pwritev(fd, &iov[idx], iov_cnt, current_offset)
                         : ::preadv(fd, &iov[idx], iov_cnt, current_offset);
    if (n < 0) {
      if (errno == EINTR) continue;
      if (errno == EINVAL) {
        int fl = ::fcntl(fd, F_GETFL);
        if (fl >= 0 && (fl & O_DIRECT)) {
          ::fcntl(fd, F_SETFL, fl & ~O_DIRECT);
          continue;
        }
      }
      return absl::ErrnoToStatus(
          errno, is_write ? "Direct zero-copy pwritev failed"
                          : "Direct zero-copy preadv failed");
    }
    if (!is_write && n == 0) {
      return absl::OutOfRangeError(absl::StrCat(
          "Unexpected EOF during direct zero-copy preadv at offset ",
          current_offset));
    }

    size_t remaining = static_cast<size_t>(n);
    current_offset += n;
    while (remaining > 0 && idx < iov.size()) {
      if (remaining >= iov[idx].iov_len) {
        remaining -= iov[idx].iov_len;
        ++idx;
      } else {
        iov[idx].iov_base =
            static_cast<uint8_t*>(iov[idx].iov_base) + remaining;
        iov[idx].iov_len -= remaining;
        remaining = 0;
      }
    }
  }
  return absl::OkStatus();
}

// Encapsulates a 4KB-aligned bounce buffer window in [dram: User host_buf]
// (`posix_memalign`) for unaligned O_DIRECT transfers (`go/tdsul-api` §3.3.1).
struct BounceWindow {
  int64_t aligned_start = 0;
  int64_t end_offset = 0;
  int64_t aligned_end = 0;
  size_t aligned_size = 0;
  size_t head_pad = 0;
  std::unique_ptr<uint8_t, decltype(&std::free)> buffer{nullptr, &std::free};
};

absl::StatusOr<BounceWindow> AllocateBounceWindow(int64_t file_offset,
                                                  size_t total_bytes,
                                                  size_t alignment) {
  BounceWindow win;
  const int64_t align_i64 = static_cast<int64_t>(alignment);
  win.aligned_start = (file_offset / align_i64) * align_i64;
  win.end_offset = file_offset + static_cast<int64_t>(total_bytes);
  win.aligned_end = ((win.end_offset + align_i64 - 1) / align_i64) * align_i64;
  win.aligned_size = static_cast<size_t>(win.aligned_end - win.aligned_start);
  win.head_pad = static_cast<size_t>(file_offset - win.aligned_start);

  void* raw = nullptr;
  if (::posix_memalign(&raw, alignment, win.aligned_size) != 0 ||
      raw == nullptr) {
    return absl::ResourceExhaustedError(
        "Failed to allocate aligned bounce buffer in [dram: User host_buf]");
  }
  std::memset(raw, 0, win.aligned_size);
  win.buffer.reset(static_cast<uint8_t*>(raw));
  return win;
}

// Reads up to `max_bytes` from `fd` at `file_offset` into `buf` in [dram: User
// host_buf], retrying on `EINTR` and stripping `O_DIRECT` if the filesystem
// rejects `O_DIRECT` with `EINVAL`. Returns an error if fewer than
// `min_required_bytes` are read before EOF.
absl::StatusOr<size_t> PreadUpTo(int fd, void* buf, size_t max_bytes,
                                 size_t min_required_bytes,
                                 int64_t file_offset) {
  size_t total_read = 0;
  uint8_t* dst = static_cast<uint8_t*>(buf);
  while (total_read < max_bytes) {
    ssize_t n = ::pread(fd, dst + total_read, max_bytes - total_read,
                        file_offset + static_cast<int64_t>(total_read));
    if (n < 0) {
      if (errno == EINTR) continue;
      if (errno == EINVAL) {
        int fl = ::fcntl(fd, F_GETFL);
        if (fl >= 0 && (fl & O_DIRECT)) {
          ::fcntl(fd, F_SETFL, fl & ~O_DIRECT);
          continue;
        }
      }
      return absl::ErrnoToStatus(errno, "Bounce buffer pread failed");
    }
    if (n == 0) break;
    total_read += static_cast<size_t>(n);
  }
  if (total_read < min_required_bytes) {
    return absl::OutOfRangeError(
        absl::StrCat("Short read in bounce buffer: required ",
                     min_required_bytes, " bytes, got ", total_read));
  }
  return total_read;
}

}  // namespace

// --- TdsBackendOptions Implementation ---

absl::StatusOr<TdsBackendOptions> TdsBackendOptions::FromProperties(
    const absl::flat_hash_map<std::string, std::string>& properties) {
  TdsBackendOptions options;

  auto parse_int = [&](absl::string_view key, auto* out) -> absl::Status {
    auto it = properties.find(key);
    if (it == properties.end()) return absl::OkStatus();
    if (!absl::SimpleAtoi(it->second, out)) {
      return absl::InvalidArgumentError(
          absl::StrCat(key, " is not a valid integer: ", it->second));
    }
    return absl::OkStatus();
  };

  auto parse_bool = [&](absl::string_view key, bool* out) -> absl::Status {
    auto it = properties.find(key);
    if (it == properties.end()) return absl::OkStatus();
    if (it->second == "true" || it->second == "1") {
      *out = true;
    } else if (it->second == "false" || it->second == "0") {
      *out = false;
    } else if (!absl::SimpleAtob(it->second, out)) {
      return absl::InvalidArgumentError(
          absl::StrCat(key, " must be a valid boolean"));
    }
    return absl::OkStatus();
  };

  if (auto it = properties.find("root_dir"); it != properties.end()) {
    if (it->second.empty()) {
      return absl::InvalidArgumentError("root_dir cannot be empty");
    }
    options.root_dir = it->second;
  }
  if (auto it = properties.find("model_name");
      it != properties.end() && !it->second.empty()) {
    options.model_name = it->second;
  }

  ABSL_RETURN_IF_ERROR(parse_int("tp_size", &options.tp_size));
  if (options.tp_size <= 0) {
    return absl::InvalidArgumentError("tp_size must be a positive integer");
  }
  ABSL_RETURN_IF_ERROR(parse_int("tp_rank", &options.tp_rank));
  if (options.tp_rank < 0 || options.tp_rank >= options.tp_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("tp_rank must be in [0, ", options.tp_size, ")"));
  }
  ABSL_RETURN_IF_ERROR(parse_int("capacity_bytes", &options.capacity_bytes));
  ABSL_RETURN_IF_ERROR(
      parse_int("lookup_batch_size", &options.lookup_batch_size));
  if (options.lookup_batch_size == 0) {
    return absl::InvalidArgumentError(
        "lookup_batch_size must be a positive integer");
  }
  ABSL_RETURN_IF_ERROR(parse_int("storage_io_thread_pool_size",
                                 &options.storage_io_thread_pool_size));
  if (options.storage_io_thread_pool_size <= 0) {
    return absl::InvalidArgumentError(
        "storage_io_thread_pool_size must be a positive integer");
  }
  ABSL_RETURN_IF_ERROR(parse_int("alignment", &options.alignment));
  if (options.alignment == 0 ||
      (options.alignment & (options.alignment - 1)) != 0) {
    return absl::InvalidArgumentError(
        "alignment must be a positive power of 2");
  }
  ABSL_RETURN_IF_ERROR(parse_bool("use_direct_io", &options.use_direct_io));
  ABSL_RETURN_IF_ERROR(
      parse_bool("require_registration", &options.require_registration));
  ABSL_RETURN_IF_ERROR(
      parse_int("lustre_stripe_size", &options.lustre_stripe_size));
  ABSL_RETURN_IF_ERROR(
      parse_int("lustre_stripe_count", &options.lustre_stripe_count));
  if (options.lustre_stripe_count < 0) {
    return absl::InvalidArgumentError(
        "lustre_stripe_count must be a non-negative integer");
  }

  return options;
}

// --- TdsKVBackend Implementation ---

TdsKVBackend::TdsKVBackend(
    std::string name, absl::flat_hash_map<std::string, std::string> properties)
    : KVBackend(properties), name_(std::move(name)) {
  absl::StatusOr<TdsBackendOptions> parsed =
      TdsBackendOptions::FromProperties(properties);
  if (!parsed.ok()) {
    LOG(WARNING) << "[TdsKVBackend] Invalid configuration for " << name_ << ": "
                 << parsed.status() << "; falling back to default options.";
    options_ = TdsBackendOptions{};
  } else {
    options_ = *std::move(parsed);
  }
  thread_pool_ = std::make_unique<NumaThreadPool>(
      std::max(1, options_.storage_io_thread_pool_size));
  mapper_ = std::make_shared<PosixPathMapper>(
      options_.root_dir, options_.model_name, options_.tp_size,
      options_.tp_rank);
}

TdsKVBackend::TdsKVBackend(
    std::string name, const TdsBackendOptions& options,
    absl::flat_hash_map<std::string, std::string> properties)
    : KVBackend(std::move(properties)),
      name_(std::move(name)),
      options_(options),
      thread_pool_(std::make_unique<NumaThreadPool>(
          std::max(1, options_.storage_io_thread_pool_size))) {
  mapper_ = std::make_shared<PosixPathMapper>(
      options_.root_dir, options_.model_name, options_.tp_size,
      options_.tp_rank);
}

TdsKVBackend::~TdsKVBackend() { thread_pool_.reset(); }

absl::Status TdsKVBackend::RegisterBuffer(void* ptr, size_t size) {
  if (ptr == nullptr || size == 0) {
    return absl::InvalidArgumentError(
        "Buffer pointer must be non-null and size must be positive");
  }
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  absl::MutexLock lock(mu_);
  // Check for overlapping registrations in [dram: User host_buf].
  auto it = registrations_.upper_bound(addr);
  if (it != registrations_.end() && addr + size > it->first) {
    return absl::AlreadyExistsError(
        "Buffer overlaps with existing registration");
  }
  if (it != registrations_.begin()) {
    auto prev = std::prev(it);
    if (prev->first + prev->second > addr) {
      return absl::AlreadyExistsError(
          "Buffer overlaps with existing registration");
    }
  }
  registrations_[addr] = size;
  return absl::OkStatus();
}

absl::Status TdsKVBackend::UnregisterBuffer(void* ptr) {
  if (ptr == nullptr) {
    return absl::InvalidArgumentError("Buffer pointer must be non-null");
  }
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  absl::MutexLock lock(mu_);
  auto it = registrations_.find(addr);
  if (it == registrations_.end()) {
    return absl::NotFoundError("Buffer not found in registration table");
  }
  registrations_.erase(it);
  return absl::OkStatus();
}

bool TdsKVBackend::IsRegisteredLocked(const void* ptr, size_t size) const {
  if (ptr == nullptr) return false;
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  auto it = registrations_.upper_bound(addr);
  if (it == registrations_.begin()) {
    return false;
  }
  --it;
  return addr >= it->first && (addr + size) <= (it->first + it->second);
}

bool TdsKVBackend::IsDmaAlignedAndRegistered(const void* ptr, size_t size,
                                             int64_t file_offset) const {
  if (ptr == nullptr || file_offset < 0) return false;
  const size_t align = options_.alignment;
  if (align == 0) return false;
  if (static_cast<size_t>(file_offset) % align != 0) return false;
  if (reinterpret_cast<uintptr_t>(ptr) % align != 0) return false;
  if (size % align != 0) return false;

  absl::MutexLock lock(mu_);
  return IsRegisteredLocked(ptr, size);
}

bool TdsKVBackend::CanUseDirectZeroCopy(
    int64_t file_offset, absl::Span<const HostBufferDescriptor> slices) const {
  if (!options_.use_direct_io) {
    return false;
  }
  const size_t align = options_.alignment;
  if (align == 0 || file_offset < 0 ||
      static_cast<size_t>(file_offset) % align != 0) {
    return false;
  }
  absl::MutexLock lock(mu_);
  for (const auto& slice : slices) {
    if (slice.ptr == nullptr && slice.size > 0) {
      return false;
    }
    if (reinterpret_cast<uintptr_t>(slice.ptr) % align != 0) {
      return false;
    }
    if (slice.size % align != 0) {
      return false;
    }
    if (options_.require_registration &&
        !IsRegisteredLocked(slice.ptr, slice.size)) {
      return false;
    }
  }
  return true;
}

TdsStats TdsKVBackend::stats() const {
  absl::MutexLock lock(mu_);
  return stats_;
}

void TdsKVBackend::ResetStats() {
  absl::MutexLock lock(mu_);
  stats_ = TdsStats{};
}

absl::StatusOr<std::string> TdsKVBackend::ResolvePath(
    const BlockKey& key) const {
  std::string target = key.resolved_key;
  if (target.empty() && mapper_ != nullptr) {
    ABSL_ASSIGN_OR_RETURN(BlockKey mapped, mapper_->MapKey(key.block_hash));
    target = mapped.resolved_key;
  }
  if (target.empty()) {
    target = key.block_hash;
  }
  if (target.empty()) {
    return absl::InvalidArgumentError("BlockKey has empty path and hash");
  }

  // Strip URI scheme if present (e.g., lustre://, nfs://, file://).
  size_t scheme_pos = target.find("://");
  if (scheme_pos != std::string::npos) {
    target = target.substr(scheme_pos + 3);
  }

  if (!target.empty() && target[0] == '/') {
    return target;
  }
  return absl::StrCat(options_.root_dir, "/", target);
}

absl::StatusOr<std::string> TdsKVBackend::ValidateAndResolve(
    const BlockKey& key, absl::Span<const HostBufferDescriptor> slices,
    size_t total_bytes) const {
  if (key.offset < 0) {
    return absl::InvalidArgumentError("Negative key offset");
  }
  size_t sum_bytes = 0;
  for (const auto& slice : slices) {
    if (slice.size > 0 && slice.ptr == nullptr) {
      return absl::InvalidArgumentError(
          "Null slice pointer with non-zero size");
    }
    sum_bytes += slice.size;
  }
  if (sum_bytes != total_bytes) {
    return absl::InvalidArgumentError(
        absl::StrCat("total_bytes (", total_bytes,
                     ") does not match sum of slice sizes (", sum_bytes, ")"));
  }
  return ResolvePath(key);
}

void TdsKVBackend::RecordSuccess(bool direct, size_t bytes) {
  absl::MutexLock lock(mu_);
  stats_.ops++;
  if (direct) {
    stats_.direct_ops++;
  } else {
    stats_.bounced_ops++;
  }
  stats_.bytes += static_cast<int64_t>(bytes);
}

absl::Status TdsKVBackend::ExecuteWrite(
    const BlockKey& key, absl::Span<const HostBufferDescriptor> slices,
    size_t total_bytes) {
  ABSL_ASSIGN_OR_RETURN(std::string path,
                        ValidateAndResolve(key, slices, total_bytes));

  // When `key.offset == 0` and we are writing the complete file (the standard
  // whole-block write case), write to a sibling `.tmp` file and atomically
  // `::rename()` on completion so concurrent `BatchExistsAsync` lookups on
  // [shared-storage: Lustre/NFS] never observe a partial file.
  struct stat existing_st = {};
  const bool file_exists = (::stat(path.c_str(), &existing_st) == 0);
  const int64_t initial_file_size = file_exists ? existing_st.st_size : 0;
  const bool use_atomic_tmp_publish =
      (key.offset == 0) &&
      (static_cast<int64_t>(total_bytes) >= initial_file_size);

  const std::string write_path =
      use_atomic_tmp_publish ? MakeUniqueTempPath(path) : path;
  bool publish_succeeded = false;
  absl::Cleanup cleanup_tmp = [&]() {
    if (use_atomic_tmp_publish && !publish_succeeded) {
      ::unlink(write_path.c_str());
    }
  };

  auto finalize_publish = [&](int* fd_ptr) -> absl::Status {
    const int fd_to_close = *fd_ptr;
    *fd_ptr = -1;
    // On [shared-storage: Lustre/NFS], ::close(fd) flushes client state and can
    // surface deferred write/quota errors (EIO, ENOSPC, EDQUOT). Verify ::close
    // succeeds before atomically renaming `write_path` into `path`.
    if (fd_to_close >= 0 && ::close(fd_to_close) < 0) {
      return absl::ErrnoToStatus(
          errno, absl::StrCat("close failed before publish for ", write_path));
    }
    if (use_atomic_tmp_publish) {
      if (::rename(write_path.c_str(), path.c_str()) < 0) {
        return absl::ErrnoToStatus(
            errno, absl::StrCat("Atomic rename failed from ", write_path,
                                " to ", path));
      }
      publish_succeeded = true;
    }
    return absl::OkStatus();
  };

  if (CanUseDirectZeroCopy(key.offset, slices)) {
    // Fast Path (Path 2 — 1-Hop O_DIRECT Vectored DMA):
    // [dram: User host_buf] --(::pwritev with O_DIRECT)--> [local-storage: NVMe
    // SSD] / [shared-storage: Lustre/NFS]. Zero bytes in [dram: OS Page Cache],
    // zero CPU memcpy through [cpu-cache].
    const int open_flags = use_atomic_tmp_publish
                               ? (O_WRONLY | O_CREAT | O_TRUNC)
                               : (O_WRONLY | O_CREAT);
    ABSL_ASSIGN_OR_RETURN(int fd, OpenFile(write_path, open_flags, 0644,
                                           options_.use_direct_io, options_));
    absl::Cleanup close_fd = [&fd]() {
      if (fd >= 0) ::close(fd);
    };

    ABSL_RETURN_IF_ERROR(
        ExecuteVectoredIo(fd, slices, key.offset, /*is_write=*/true));
    ABSL_RETURN_IF_ERROR(finalize_publish(&fd));
    RecordSuccess(/*direct=*/true, total_bytes);
    return absl::OkStatus();
  }

  // Fallback Path: 4KB Bounce-Buffer in [dram: User host_buf] (`posix_memalign`).
  // Handles unaligned caller slices or unaligned file offsets as required by
  // `go/tdsul-api` (§1 & §3.3.1).
  const int open_flags = use_atomic_tmp_publish
                             ? (O_RDWR | O_CREAT | O_TRUNC)
                             : (O_RDWR | O_CREAT);
  ABSL_ASSIGN_OR_RETURN(int fd, OpenFile(write_path, open_flags, 0644,
                                         options_.use_direct_io, options_));
  absl::Cleanup close_fd = [&fd]() {
    if (fd >= 0) ::close(fd);
  };

  if (total_bytes == 0) {
    ABSL_RETURN_IF_ERROR(finalize_publish(&fd));
    RecordSuccess(/*direct=*/false, 0);
    return absl::OkStatus();
  }

  ABSL_ASSIGN_OR_RETURN(
      BounceWindow win,
      AllocateBounceWindow(key.offset, total_bytes, options_.alignment));

  // If updating an existing file in-place and the bounce window overlaps
  // existing head or tail bytes, pre-read the existing window to preserve them.
  if (!use_atomic_tmp_publish && initial_file_size > win.aligned_start &&
      (win.head_pad > 0 || win.end_offset < win.aligned_end)) {
    const size_t existing_in_window = static_cast<size_t>(
        std::min<int64_t>(win.aligned_end, initial_file_size) -
        win.aligned_start);
    (void)PreadUpTo(fd, win.buffer.get(), win.aligned_size, existing_in_window,
                    win.aligned_start);
  }

  size_t offset_in_slices = 0;
  for (const auto& slice : slices) {
    if (slice.size > 0) {
      std::memcpy(win.buffer.get() + win.head_pad + offset_in_slices, slice.ptr,
                  slice.size);
      offset_in_slices += slice.size;
    }
  }

  const HostBufferDescriptor bounce_desc{.ptr = win.buffer.get(),
                                         .size = win.aligned_size};
  ABSL_RETURN_IF_ERROR(ExecuteVectoredIo(fd, {bounce_desc}, win.aligned_start,
                                         /*is_write=*/true));

  const int64_t target_file_size =
      use_atomic_tmp_publish
          ? win.end_offset
          : std::max<int64_t>(initial_file_size, win.end_offset);
  if (target_file_size < win.aligned_end) {
    if (::ftruncate(fd, target_file_size) < 0) {
      return absl::ErrnoToStatus(errno, "ftruncate failed after bounce write");
    }
  }

  ABSL_RETURN_IF_ERROR(finalize_publish(&fd));
  RecordSuccess(/*direct=*/false, total_bytes);
  return absl::OkStatus();
}

absl::Status TdsKVBackend::ExecuteRead(
    const BlockKey& key, absl::Span<const HostBufferDescriptor> slices,
    size_t total_bytes) {
  ABSL_ASSIGN_OR_RETURN(std::string path,
                        ValidateAndResolve(key, slices, total_bytes));

  ABSL_ASSIGN_OR_RETURN(
      int fd, OpenFile(path, O_RDONLY, 0, options_.use_direct_io, options_));
  absl::Cleanup close_fd = [&fd]() {
    if (fd >= 0) ::close(fd);
  };

  auto close_and_verify = [&]() -> absl::Status {
    const int fd_to_close = fd;
    fd = -1;
    if (fd_to_close >= 0 && ::close(fd_to_close) < 0) {
      return absl::ErrnoToStatus(
          errno, absl::StrCat("Failed to close file after read: ", path));
    }
    return absl::OkStatus();
  };

  if (CanUseDirectZeroCopy(key.offset, slices)) {
    // Fast Path (Path 2 — 1-Hop O_DIRECT Vectored DMA):
    // [local-storage: NVMe SSD] / [shared-storage: Lustre/NFS]
    //   --(::preadv with O_DIRECT)--> [dram: User host_buf].
    ABSL_RETURN_IF_ERROR(
        ExecuteVectoredIo(fd, slices, key.offset, /*is_write=*/false));
    ABSL_RETURN_IF_ERROR(close_and_verify());
    RecordSuccess(/*direct=*/true, total_bytes);
    return absl::OkStatus();
  }

  // Fallback Path: 4KB Bounce-Buffer in [dram: User host_buf] (`posix_memalign`).
  if (total_bytes == 0) {
    ABSL_RETURN_IF_ERROR(close_and_verify());
    RecordSuccess(/*direct=*/false, 0);
    return absl::OkStatus();
  }

  ABSL_ASSIGN_OR_RETURN(
      BounceWindow win,
      AllocateBounceWindow(key.offset, total_bytes, options_.alignment));
  const size_t required_bytes = win.head_pad + total_bytes;

  ABSL_RETURN_IF_ERROR(PreadUpTo(fd, win.buffer.get(), win.aligned_size,
                                 required_bytes, win.aligned_start)
                           .status());
  ABSL_RETURN_IF_ERROR(close_and_verify());

  size_t offset_in_slices = 0;
  for (const auto& slice : slices) {
    if (slice.size > 0) {
      std::memcpy(slice.ptr, win.buffer.get() + win.head_pad + offset_in_slices,
                  slice.size);
      offset_in_slices += slice.size;
    }
  }

  RecordSuccess(/*direct=*/false, total_bytes);
  return absl::OkStatus();
}

void TdsKVBackend::WriteAsync(const BlockKey& key,
                              absl::Span<const HostBufferDescriptor> slices,
                              size_t total_bytes,
                              std::function<void(absl::Status)> callback) {
  std::vector<HostBufferDescriptor> owned_slices(slices.begin(), slices.end());
  thread_pool_->Schedule(
      std::nullopt, [this, key, owned_slices = std::move(owned_slices),
                     total_bytes, callback = std::move(callback)]() {
        absl::Status status = ExecuteWrite(key, owned_slices, total_bytes);
        if (!status.ok()) {
          absl::MutexLock lock(mu_);
          stats_.ops++;
          stats_.failed_ops++;
        }
        if (callback) {
          callback(std::move(status));
        }
      });
}

void TdsKVBackend::ReadAsync(const BlockKey& key,
                             absl::Span<const HostBufferDescriptor> slices,
                             size_t total_bytes,
                             std::function<void(absl::Status)> callback) {
  std::vector<HostBufferDescriptor> owned_slices(slices.begin(), slices.end());
  thread_pool_->Schedule(
      std::nullopt, [this, key, owned_slices = std::move(owned_slices),
                     total_bytes, callback = std::move(callback)]() {
        absl::Status status = ExecuteRead(key, owned_slices, total_bytes);
        if (!status.ok()) {
          absl::MutexLock lock(mu_);
          stats_.ops++;
          stats_.failed_ops++;
        }
        if (callback) {
          callback(std::move(status));
        }
      });
}

absl::StatusOr<bool> TdsKVBackend::Exists(const BlockKey& key) const {
  if (key.offset < 0 || key.size < 0) {
    return absl::InvalidArgumentError("Negative offset or size in BlockKey");
  }
  ABSL_ASSIGN_OR_RETURN(std::string path, ResolvePath(key));
  struct stat st = {};
  if (::stat(path.c_str(), &st) == 0) {
    return S_ISREG(st.st_mode) && (st.st_size >= key.offset + key.size);
  }
  if (errno == ENOENT || errno == ENOTDIR) {
    return false;
  }
  return absl::ErrnoToStatus(errno, absl::StrCat("stat failed for ", path));
}

void TdsKVBackend::BatchExistsAsync(
    absl::Span<const BlockKey> keys,
    std::function<void(std::vector<absl::StatusOr<bool>>)> callback) {
  if (keys.empty()) {
    thread_pool_->Schedule(std::nullopt, [callback = std::move(callback)]() {
      if (callback) {
        callback({});
      }
    });
    return;
  }

  struct BatchState {
    explicit BatchState(
        size_t count, std::function<void(std::vector<absl::StatusOr<bool>>)> cb)
        : remaining(count), results(count), callback(std::move(cb)) {}
    std::atomic<size_t> remaining;
    std::vector<absl::StatusOr<bool>> results;
    std::function<void(std::vector<absl::StatusOr<bool>>)> callback;
  };

  auto state = std::make_shared<BatchState>(keys.size(), std::move(callback));
  for (size_t i = 0; i < keys.size(); ++i) {
    thread_pool_->Schedule(std::nullopt, [this, i, key = keys[i], state]() {
      state->results[i] = Exists(key);
      if (state->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1 &&
          state->callback) {
        state->callback(std::move(state->results));
      }
    });
  }
}

}  // namespace storage
}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden

using ::tpu_raiden::kv_cache::backends::storage::PosixKVCacheStoreBackend;
using ::tpu_raiden::kv_cache::backends::storage::TdsBackendOptions;
using ::tpu_raiden::kv_cache::backends::storage::TdsKVBackend;

REGISTER_KV_CACHE_STORE_BACKEND(
    ::tpu_raiden::kv_cache::backends::storage::kTdsBackendName,
    [](const ::tpu_raiden::kv_cache::BackendConfig& config,
       ::tpu_raiden::controller::RaidenController* controller)
        -> absl::StatusOr<
            std::shared_ptr<::tpu_raiden::kv_cache::KVCacheStoreBackend>> {
      int tp_size = static_cast<int>(config.GetIntProperty("tp_size", 0));
      if (tp_size <= 0 && controller != nullptr) {
        size_t registered_workers =
            controller->worker_registry()->GetRegisteredWorkers().size();
        if (registered_workers > 0) {
          tp_size = static_cast<int>(registered_workers);
        } else if (controller->num_shards() > 0) {
          tp_size = controller->num_shards();
        }
      }
      if (tp_size <= 0) {
        tp_size = 1;
      }

      // The coordinator resolves block existence through the rank-0 witness
      // via PosixKVCacheStoreBackend::Lookup, so its mapper is pinned to
      // rank 0. Per-worker tp_rank lives on each worker's own config.
      auto properties = config.properties;
      properties["tp_size"] = absl::StrCat(tp_size);
      properties["tp_rank"] = "0";

      const std::string backend_name = std::string(
          ::tpu_raiden::kv_cache::backends::storage::kTdsBackendName);
      ABSL_ASSIGN_OR_RETURN(TdsBackendOptions options,
                            TdsBackendOptions::FromProperties(properties));
      auto backend =
          std::make_shared<TdsKVBackend>(backend_name, options, properties);
      return std::make_shared<PosixKVCacheStoreBackend>(
          std::move(backend), backend_name, options.capacity_bytes,
          options.lookup_batch_size);
    });
