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

#include <memory>
#include <vector>

#include "torch/torch.h"
#include "tpu_sync/frameworks/torch/torch_tpu_utils_mock.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace raiden {
namespace {

using ::tpu_raiden::torch::RegisterMockTensor;

class TorchRawTransferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(client_,
                            xla::GetXlaPjrtCpuClient(xla::CpuClientOptions()));
    device_ = client_->addressable_devices()[0];
  }

  std::unique_ptr<xla::PjRtClient> client_;
  xla::PjRtDevice* device_;
};

TEST_F(TorchRawTransferTest, RawHostBufferAllocation) {
  RawHostBuffer buffer(1024);
  EXPECT_EQ(buffer.SizeBytes(), 1024);
  EXPECT_FALSE(buffer.IsPjRtBacked());

  buffer.EnsureBoundToDevice(device_);
  EXPECT_TRUE(buffer.IsPjRtBacked() || buffer.DataPtr() != 0);
  EXPECT_NE(buffer.DataPtr(), 0);
}

TEST_F(TorchRawTransferTest, PreparedTransferD2HAndH2D) {
  // Create a real CPU PJRT buffer to mock TPU buffer
  TF_ASSERT_OK_AND_ASSIGN(xla::PjRtMemorySpace * memory_space,
                          device_->default_memory_space());
  std::vector<float> initial_data(256, 42.0f);
  TF_ASSERT_OK_AND_ASSIGN(
      auto pjrt_buffer, client_->BufferFromHostBuffer(
                            initial_data.data(), xla::F32, {256},
                            /*byte_strides=*/std::nullopt,
                            xla::PjRtClient::HostBufferSemantics::
                                kImmutableUntilTransferCompletes,
                            /*on_done_with_host_buffer=*/nullptr, memory_space,
                            /*device_layout=*/nullptr));

  // Create a CPU PyTorch tensor to represent the TPU tensor
  at::Tensor tpu_tensor = ::torch::zeros({256}, ::torch::kFloat32);
  RegisterMockTensor(tpu_tensor, pjrt_buffer.get());

  // Create Host buffer
  auto host_buffer = std::make_shared<RawHostBuffer>(256 * sizeof(float));
  host_buffer->EnsureBoundToDevice(device_);

  // Prepare transfer
  auto transfer = std::make_shared<PreparedTorchRawTransfer>(
      tpu_tensor, host_buffer, /*unsafe_skip_buffer_lock=*/true);

  // Test D2H (Device to Host)
  transfer->D2H();

  // Verify host buffer has the data
  float* host_data = reinterpret_cast<float*>(host_buffer->MutableData());
  for (int i = 0; i < 256; ++i) {
    EXPECT_EQ(host_data[i], 42.0f);
  }

  // Modify host buffer data
  for (int i = 0; i < 256; ++i) {
    host_data[i] = 24.0f;
  }

  // Test H2D (Host to Device)
  transfer->H2D();

  // To verify H2D, we would need to read back from pjrt_buffer.
  // We can do this by creating another tensor/buffer and doing D2H again,
  // or just use PJRT API to read from pjrt_buffer.
  // Let's use PJRT API to check.
  std::vector<float> readback_data(256);
  ASSERT_OK(
      pjrt_buffer->CopyRawToHost(readback_data.data(), 0, 256 * sizeof(float))
          .Await());

  for (int i = 0; i < 256; ++i) {
    EXPECT_EQ(readback_data[i], 24.0f);
  }
}

TEST_F(TorchRawTransferTest, BatchTransferD2HAndH2D) {
  TF_ASSERT_OK_AND_ASSIGN(xla::PjRtMemorySpace * memory_space,
                          device_->default_memory_space());
  std::vector<float> initial_data(256, 42.0f);
  TF_ASSERT_OK_AND_ASSIGN(
      auto pjrt_buffer, client_->BufferFromHostBuffer(
                            initial_data.data(), xla::F32, {256},
                            /*byte_strides=*/std::nullopt,
                            xla::PjRtClient::HostBufferSemantics::
                                kImmutableUntilTransferCompletes,
                            /*on_done_with_host_buffer=*/nullptr, memory_space,
                            /*device_layout=*/nullptr));

  at::Tensor tpu_tensor = ::torch::zeros({256}, ::torch::kFloat32);
  RegisterMockTensor(tpu_tensor, pjrt_buffer.get());

  at::Tensor host_tensor = ::torch::zeros({256}, ::torch::kFloat32);

  // D2H Batch Async
  auto future = TransferD2HBatchAsync({tpu_tensor}, {host_tensor}, {}, {}, {});
  ASSERT_OK(future.Await());

  float* host_data = reinterpret_cast<float*>(host_tensor.data_ptr());
  for (int i = 0; i < 256; ++i) {
    EXPECT_EQ(host_data[i], 42.0f);
  }

  // Modify host data
  for (int i = 0; i < 256; ++i) {
    host_data[i] = 24.0f;
  }

  // H2D Batch Async
  auto future2 = TransferH2DBatchAsync({host_tensor}, {tpu_tensor}, {}, {}, {});
  ASSERT_OK(future2.Await());

  // Verify
  std::vector<float> readback_data(256);
  ASSERT_OK(
      pjrt_buffer->CopyRawToHost(readback_data.data(), 0, 256 * sizeof(float))
          .Await());

  for (int i = 0; i < 256; ++i) {
    EXPECT_EQ(readback_data[i], 24.0f);
  }
}

TEST_F(TorchRawTransferTest, PreparedBatchPartialD2HAndH2D) {
  TF_ASSERT_OK_AND_ASSIGN(xla::PjRtMemorySpace * memory_space,
                          device_->default_memory_space());
  constexpr int64_t kMajor = 4;
  constexpr int64_t kRows = 32;
  constexpr int64_t kCols = 32;
  constexpr int64_t kSliceElements = kRows * kCols;
  constexpr int64_t kElements = kMajor * kSliceElements;

  std::vector<float> first(kElements);
  std::vector<float> second(kElements);
  for (int64_t i = 0; i < kElements; ++i) {
    first[i] = static_cast<float>(i);
    second[i] = static_cast<float>(10000 + i);
  }
  TF_ASSERT_OK_AND_ASSIGN(
      auto first_buffer, client_->BufferFromHostBuffer(
                             first.data(), xla::F32, {kMajor, kRows, kCols},
                             /*byte_strides=*/std::nullopt,
                             xla::PjRtClient::HostBufferSemantics::
                                 kImmutableUntilTransferCompletes,
                             /*on_done_with_host_buffer=*/nullptr, memory_space,
                             /*device_layout=*/nullptr));
  TF_ASSERT_OK_AND_ASSIGN(
      auto second_buffer,
      client_->BufferFromHostBuffer(
          second.data(), xla::F32, {kMajor, kRows, kCols},
          /*byte_strides=*/std::nullopt,
          xla::PjRtClient::HostBufferSemantics::
              kImmutableUntilTransferCompletes,
          /*on_done_with_host_buffer=*/nullptr, memory_space,
          /*device_layout=*/nullptr));

  at::Tensor first_tpu =
      ::torch::zeros({kMajor, kRows, kCols}, ::torch::kFloat32);
  at::Tensor second_tpu =
      ::torch::zeros({kMajor, kRows, kCols}, ::torch::kFloat32);
  RegisterMockTensor(first_tpu, first_buffer.get());
  RegisterMockTensor(second_tpu, second_buffer.get());
  at::Tensor first_host =
      ::torch::zeros({kMajor, kRows, kCols}, ::torch::kFloat32);
  at::Tensor second_host =
      ::torch::zeros({kMajor, kRows, kCols}, ::torch::kFloat32);

  auto transfer = std::make_shared<PreparedTorchRawTransferBatch>(
      std::vector<at::Tensor>{first_tpu, second_tpu},
      std::vector<at::Tensor>{first_host, second_host},
      /*unsafe_skip_buffer_lock=*/true);
  EXPECT_EQ(transfer->Size(), 2);

  // Copy TPU page 1 into host page 2 for both tensors.
  ASSERT_OK(transfer
                ->D2HAsync(/*src_offsets_major_dim=*/{1},
                           /*dst_offsets_major_dim=*/{2},
                           /*copy_sizes_major_dim=*/{1})
                .Await());
  const float* first_host_data = first_host.data_ptr<float>();
  const float* second_host_data = second_host.data_ptr<float>();
  for (int64_t i = 0; i < kSliceElements; ++i) {
    EXPECT_EQ(first_host_data[2 * kSliceElements + i],
              first[kSliceElements + i]);
    EXPECT_EQ(second_host_data[2 * kSliceElements + i],
              second[kSliceElements + i]);
  }

  // Copy the staged host page into TPU page 3, then verify with PJRT.
  first_host.slice(0, 2, 3).fill_(7.0f);
  second_host.slice(0, 2, 3).fill_(9.0f);
  ASSERT_OK(transfer
                ->H2DAsync(/*src_offsets_major_dim=*/{2},
                           /*dst_offsets_major_dim=*/{3},
                           /*copy_sizes_major_dim=*/{1})
                .Await());
  std::vector<float> first_readback(kElements);
  std::vector<float> second_readback(kElements);
  ASSERT_OK(
      first_buffer
          ->CopyRawToHost(first_readback.data(), 0, kElements * sizeof(float))
          .Await());
  ASSERT_OK(
      second_buffer
          ->CopyRawToHost(second_readback.data(), 0, kElements * sizeof(float))
          .Await());
  for (int64_t i = 0; i < kSliceElements; ++i) {
    EXPECT_EQ(first_readback[3 * kSliceElements + i], 7.0f);
    EXPECT_EQ(second_readback[3 * kSliceElements + i], 9.0f);
  }
}

}  // namespace
}  // namespace raiden
