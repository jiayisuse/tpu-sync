#ifndef SGLANG_TPU_OLD_ABI_MATERIALIZE_H_
#define SGLANG_TPU_OLD_ABI_MATERIALIZE_H_

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "torch_tpu/csrc/eager/device_buffer.h"

namespace torch_tpu {

// Numeric values are pinned to the August 2026 wheel ABI and verified from
// the released TPU Sync extension's call site.
enum class MaterializationReason : int {
  kUnknown = 0,
  kExecute = 1,
  kToHost = 2,
  kExplicitSync = 3,
};

enum class MaterializationMode : int {
  kAsync = 0,
  kWait = 1,
};

absl::Status Materialize(absl::Span<const DeviceBufferRef> buffers,
                         MaterializationReason reason,
                         MaterializationMode mode);

inline absl::Status Materialize(const DeviceBufferRef& buffer,
                                MaterializationReason reason) {
  return Materialize(absl::Span<const DeviceBufferRef>(&buffer, 1), reason,
                     MaterializationMode::kWait);
}

}  // namespace torch_tpu

#endif  // SGLANG_TPU_OLD_ABI_MATERIALIZE_H_
