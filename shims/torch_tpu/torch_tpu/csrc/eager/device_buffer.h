#ifndef SGLANG_TPU_OLD_ABI_DEVICE_BUFFER_H_
#define SGLANG_TPU_OLD_ABI_DEVICE_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"

namespace torch_tpu {

class DeviceBufferList;

// ABI declaration for torch-tpu 0.1.1.dev20260817010134. The implementation
// is supplied by that wheel's libpywrap_<torch-version>_common.so. Keep this
// layout pinned: shared_ptr<DeviceBufferList> followed by the buffer index.
class DeviceBufferRef {
 public:
  DeviceBufferRef() = default;
  DeviceBufferRef(const DeviceBufferRef&) = default;
  DeviceBufferRef(DeviceBufferRef&&) noexcept = default;
  DeviceBufferRef& operator=(const DeviceBufferRef&) = default;
  DeviceBufferRef& operator=(DeviceBufferRef&&) noexcept = default;
  ~DeviceBufferRef();

  absl::Span<const int64_t> dimensions() const;
  size_t size_bytes() const;
  absl::StatusOr<xla::PjRtBuffer*> AwaitBuffer() const;

 private:
  std::shared_ptr<DeviceBufferList> buffers_;
  int64_t index_ = -1;
};

static_assert(sizeof(DeviceBufferRef) == 24);

}  // namespace torch_tpu

#endif  // SGLANG_TPU_OLD_ABI_DEVICE_BUFFER_H_
