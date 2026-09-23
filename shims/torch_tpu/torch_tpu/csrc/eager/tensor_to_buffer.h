#ifndef SGLANG_TPU_OLD_ABI_TENSOR_TO_BUFFER_H_
#define SGLANG_TPU_OLD_ABI_TENSOR_TO_BUFFER_H_

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "torch_tpu/csrc/eager/device_buffer.h"

namespace torch_tpu {

absl::StatusOr<DeviceBufferRef> GetBaseBuffer(const at::Tensor& tensor);

}  // namespace torch_tpu

#endif  // SGLANG_TPU_OLD_ABI_TENSOR_TO_BUFFER_H_
