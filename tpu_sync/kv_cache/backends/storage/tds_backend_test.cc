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

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "tpu_sync/kv_cache/backends/backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"

namespace tpu_raiden {
namespace kv_cache {
namespace backends {
namespace storage {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

// Verify class inheritance hierarchy and namespace alias.
static_assert(std::is_base_of_v<tkv::backends::KVBackend,
                                tkv::backends::storage::TdsKVBackend>);

constexpr size_t kAlignment = 4096;

class AlignedBuffer {
 public:
  explicit AlignedBuffer(size_t size, size_t alignment = kAlignment)
      : size_(size) {
    void* raw = nullptr;
    if (posix_memalign(&raw, alignment, size) != 0) {
      raw = nullptr;
    }
    ptr_ = static_cast<uint8_t*>(raw);
  }

  ~AlignedBuffer() { std::free(ptr_); }

  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  uint8_t* data() const { return ptr_; }
  size_t size() const { return size_; }

 private:
  uint8_t* ptr_ = nullptr;
  size_t size_ = 0;
};

std::vector<uint8_t> MakeDeterministicPayload(size_t size, uint8_t seed) {
  std::vector<uint8_t> payload(size);
  for (size_t i = 0; i < size; ++i) {
    payload[i] = static_cast<uint8_t>((i * 37 + seed * 13 + 7) & 0xFF);
  }
  return payload;
}

class TdsKVBackendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const ::testing::TestInfo* info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    const std::string test_name = info != nullptr ? info->name() : "test";
    test_dir_ = absl::StrCat(::testing::TempDir(), "/tds_kv_", test_name);
    std::filesystem::remove_all(test_dir_);
    std::filesystem::create_directories(test_dir_);
  }

  void TearDown() override { std::filesystem::remove_all(test_dir_); }

  std::string test_dir_;
};

TEST_F(TdsKVBackendTest, DirectZeroCopyMultiSliceReadWrite) {
  TdsBackendOptions options;
  options.root_dir = test_dir_;
  options.storage_io_thread_pool_size = 4;
  options.alignment = kAlignment;
  options.use_direct_io = true;

  TdsKVBackend backend("tds_direct", options);

  constexpr size_t kSlice0Size = 4096;
  constexpr size_t kSlice1Size = 8192;
  constexpr size_t kSlice2Size = 4096;
  constexpr size_t kTotalBytes = kSlice0Size + kSlice1Size + kSlice2Size;

  AlignedBuffer src0(kSlice0Size);
  AlignedBuffer src1(kSlice1Size);
  AlignedBuffer src2(kSlice2Size);
  ASSERT_NE(src0.data(), nullptr);
  ASSERT_NE(src1.data(), nullptr);
  ASSERT_NE(src2.data(), nullptr);

  std::vector<uint8_t> expected0 = MakeDeterministicPayload(kSlice0Size, 1);
  std::vector<uint8_t> expected1 = MakeDeterministicPayload(kSlice1Size, 2);
  std::vector<uint8_t> expected2 = MakeDeterministicPayload(kSlice2Size, 3);
  std::memcpy(src0.data(), expected0.data(), kSlice0Size);
  std::memcpy(src1.data(), expected1.data(), kSlice1Size);
  std::memcpy(src2.data(), expected2.data(), kSlice2Size);

  ABSL_ASSERT_OK(backend.RegisterBuffer(src0.data(), kSlice0Size));
  ABSL_ASSERT_OK(backend.RegisterBuffer(src1.data(), kSlice1Size));
  ABSL_ASSERT_OK(backend.RegisterBuffer(src2.data(), kSlice2Size));

  EXPECT_TRUE(backend.IsDmaAlignedAndRegistered(src0.data(), kSlice0Size, 0));
  EXPECT_TRUE(
      backend.IsDmaAlignedAndRegistered(src1.data(), kSlice1Size, kSlice0Size));
  EXPECT_TRUE(backend.IsDmaAlignedAndRegistered(src2.data(), kSlice2Size,
                                                kSlice0Size + kSlice1Size));

  std::vector<HostBufferDescriptor> write_slices = {
      HostBufferDescriptor{.ptr = src0.data(), .size = kSlice0Size},
      HostBufferDescriptor{.ptr = src1.data(), .size = kSlice1Size},
      HostBufferDescriptor{.ptr = src2.data(), .size = kSlice2Size},
  };

  BlockKey key;
  key.block_hash = "direct_block_01";
  key.resolved_key = absl::StrCat(test_dir_, "/direct_block_01.bin");
  key.offset = 0;
  key.size = kTotalBytes;

  absl::Notification write_done;
  absl::Status write_status;
  backend.WriteAsync(key, write_slices, kTotalBytes, [&](absl::Status status) {
    write_status = std::move(status);
    write_done.Notify();
  });
  write_done.WaitForNotification();
  ABSL_ASSERT_OK(write_status);

  AlignedBuffer dst0(kSlice0Size);
  AlignedBuffer dst1(kSlice1Size);
  AlignedBuffer dst2(kSlice2Size);
  std::memset(dst0.data(), 0, kSlice0Size);
  std::memset(dst1.data(), 0, kSlice1Size);
  std::memset(dst2.data(), 0, kSlice2Size);

  ABSL_ASSERT_OK(backend.RegisterBuffer(dst0.data(), kSlice0Size));
  ABSL_ASSERT_OK(backend.RegisterBuffer(dst1.data(), kSlice1Size));
  ABSL_ASSERT_OK(backend.RegisterBuffer(dst2.data(), kSlice2Size));

  std::vector<HostBufferDescriptor> read_slices = {
      HostBufferDescriptor{.ptr = dst0.data(), .size = kSlice0Size},
      HostBufferDescriptor{.ptr = dst1.data(), .size = kSlice1Size},
      HostBufferDescriptor{.ptr = dst2.data(), .size = kSlice2Size},
  };

  absl::Notification read_done;
  absl::Status read_status;
  backend.ReadAsync(key, read_slices, kTotalBytes, [&](absl::Status status) {
    read_status = std::move(status);
    read_done.Notify();
  });
  read_done.WaitForNotification();
  ABSL_ASSERT_OK(read_status);

  EXPECT_EQ(std::memcmp(dst0.data(), expected0.data(), kSlice0Size), 0);
  EXPECT_EQ(std::memcmp(dst1.data(), expected1.data(), kSlice1Size), 0);
  EXPECT_EQ(std::memcmp(dst2.data(), expected2.data(), kSlice2Size), 0);

  TdsStats stats = backend.stats();
  EXPECT_EQ(stats.ops, 2);
  EXPECT_EQ(stats.direct_ops, 2);
  EXPECT_EQ(stats.bounced_ops, 0);
  EXPECT_EQ(stats.failed_ops, 0);
  EXPECT_EQ(stats.bytes, 2 * kTotalBytes);

  ABSL_EXPECT_OK(backend.UnregisterBuffer(src0.data()));
  ABSL_EXPECT_OK(backend.UnregisterBuffer(src1.data()));
  ABSL_EXPECT_OK(backend.UnregisterBuffer(src2.data()));
  ABSL_EXPECT_OK(backend.UnregisterBuffer(dst0.data()));
  ABSL_EXPECT_OK(backend.UnregisterBuffer(dst1.data()));
  ABSL_EXPECT_OK(backend.UnregisterBuffer(dst2.data()));
}

TEST_F(TdsKVBackendTest, UnalignedBounceBufferMultiSliceReadWrite) {
  TdsBackendOptions options;
  options.root_dir = test_dir_;
  options.storage_io_thread_pool_size = 4;
  options.alignment = kAlignment;
  options.use_direct_io = true;

  TdsKVBackend backend("tds_bounce", options);

  constexpr size_t kSlice0Size = 1000;
  constexpr size_t kSlice1Size = 513;
  constexpr size_t kSlice2Size = 2000;
  constexpr size_t kTotalBytes = kSlice0Size + kSlice1Size + kSlice2Size;
  static_assert(kTotalBytes == 3513);

  std::vector<uint8_t> src0 = MakeDeterministicPayload(kSlice0Size, 11);
  std::vector<uint8_t> src1 = MakeDeterministicPayload(kSlice1Size, 22);
  std::vector<uint8_t> src2 = MakeDeterministicPayload(kSlice2Size, 33);

  std::vector<HostBufferDescriptor> write_slices = {
      HostBufferDescriptor{.ptr = src0.data(), .size = kSlice0Size},
      HostBufferDescriptor{.ptr = src1.data(), .size = kSlice1Size},
      HostBufferDescriptor{.ptr = src2.data(), .size = kSlice2Size},
  };

  BlockKey key;
  key.block_hash = "bounced_block_01";
  key.resolved_key = absl::StrCat(test_dir_, "/bounced_block_01.bin");
  key.offset = 0;
  key.size = kTotalBytes;

  absl::Notification write_done;
  absl::Status write_status;
  backend.WriteAsync(key, write_slices, kTotalBytes, [&](absl::Status status) {
    write_status = std::move(status);
    write_done.Notify();
  });
  write_done.WaitForNotification();
  ABSL_ASSERT_OK(write_status);

  // Verify logical file size matches exact written bytes (not padded 4096).
  ASSERT_TRUE(std::filesystem::exists(key.resolved_key));
  EXPECT_EQ(std::filesystem::file_size(key.resolved_key), kTotalBytes);

  std::vector<uint8_t> dst0(kSlice0Size, 0);
  std::vector<uint8_t> dst1(kSlice1Size, 0);
  std::vector<uint8_t> dst2(kSlice2Size, 0);

  std::vector<HostBufferDescriptor> read_slices = {
      HostBufferDescriptor{.ptr = dst0.data(), .size = kSlice0Size},
      HostBufferDescriptor{.ptr = dst1.data(), .size = kSlice1Size},
      HostBufferDescriptor{.ptr = dst2.data(), .size = kSlice2Size},
  };

  absl::Notification read_done;
  absl::Status read_status;
  backend.ReadAsync(key, read_slices, kTotalBytes, [&](absl::Status status) {
    read_status = std::move(status);
    read_done.Notify();
  });
  read_done.WaitForNotification();
  ABSL_ASSERT_OK(read_status);

  EXPECT_EQ(dst0, src0);
  EXPECT_EQ(dst1, src1);
  EXPECT_EQ(dst2, src2);

  TdsStats stats = backend.stats();
  EXPECT_EQ(stats.ops, 2);
  EXPECT_EQ(stats.bounced_ops, 2);
  EXPECT_EQ(stats.direct_ops, 0);
  EXPECT_EQ(stats.failed_ops, 0);
  EXPECT_EQ(stats.bytes, 2 * kTotalBytes);
}

TEST_F(TdsKVBackendTest, LustreAndNfsSchemeConfiguration) {
  absl::flat_hash_map<std::string, std::string> lustre_props = {
      {"root_dir", test_dir_},
      {"storage_io_thread_pool_size", "8"},
      {"alignment", "4096"},
      {"use_direct_io", "true"},
      {"lustre_stripe_size", "1048576"},
      {"lustre_stripe_count", "4"},
  };

  absl::StatusOr<TdsBackendOptions> parsed_lustre_opts =
      TdsBackendOptions::FromProperties(lustre_props);
  ABSL_ASSERT_OK(parsed_lustre_opts);
  EXPECT_EQ(parsed_lustre_opts->root_dir, test_dir_);
  EXPECT_EQ(parsed_lustre_opts->storage_io_thread_pool_size, 8);
  EXPECT_EQ(parsed_lustre_opts->alignment, 4096);
  EXPECT_TRUE(parsed_lustre_opts->use_direct_io);
  EXPECT_EQ(parsed_lustre_opts->lustre_stripe_size, 1048576);
  EXPECT_EQ(parsed_lustre_opts->lustre_stripe_count, 4);

  TdsKVBackend lustre_backend("lustre", lustre_props);
  EXPECT_EQ(lustre_backend.name(), "lustre");
  EXPECT_EQ(lustre_backend.alignment(), 4096);
  EXPECT_EQ(lustre_backend.options().lustre_stripe_count, 4);

  absl::flat_hash_map<std::string, std::string> nfs_props = {
      {"root_dir", test_dir_},
      {"storage_io_thread_pool_size", "4"},
      {"alignment", "4096"},
      {"use_direct_io", "false"},
  };
  absl::StatusOr<TdsBackendOptions> parsed_nfs_opts =
      TdsBackendOptions::FromProperties(nfs_props);
  ABSL_ASSERT_OK(parsed_nfs_opts);
  EXPECT_FALSE(parsed_nfs_opts->use_direct_io);

  TdsKVBackend nfs_backend("nfs", *parsed_nfs_opts, nfs_props);
  EXPECT_EQ(nfs_backend.name(), "nfs");

  // Verify URI scheme resolution (e.g. lustre:// and nfs://).
  std::vector<uint8_t> data = MakeDeterministicPayload(1024, 99);
  HostBufferDescriptor desc{.ptr = data.data(), .size = data.size()};

  BlockKey lustre_key;
  lustre_key.block_hash = "lustre_hash";
  lustre_key.resolved_key =
      absl::StrCat("lustre://", test_dir_, "/lustre_obj.bin");
  lustre_key.size = data.size();

  absl::Notification done1;
  absl::Status status1;
  lustre_backend.WriteAsync(lustre_key, {desc}, data.size(),
                            [&](absl::Status s) {
                              status1 = std::move(s);
                              done1.Notify();
                            });
  done1.WaitForNotification();
  ABSL_ASSERT_OK(status1);

  std::vector<uint8_t> read_back(1024, 0);
  HostBufferDescriptor read_desc{.ptr = read_back.data(),
                                 .size = read_back.size()};
  absl::Notification done2;
  absl::Status status2;
  nfs_backend.ReadAsync(BlockKey{.block_hash = "lustre_hash",
                                 .resolved_key = absl::StrCat(
                                     "nfs://", test_dir_, "/lustre_obj.bin"),
                                 .size = static_cast<int64_t>(data.size())},
                        {read_desc}, read_back.size(), [&](absl::Status s) {
                          status2 = std::move(s);
                          done2.Notify();
                        });
  done2.WaitForNotification();
  ABSL_ASSERT_OK(status2);
  EXPECT_EQ(read_back, data);

  // Verify invalid property error handling.
  EXPECT_THAT(TdsBackendOptions::FromProperties({{"alignment", "3000"}}),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TdsBackendOptions::FromProperties(
                  {{"storage_io_thread_pool_size", "-2"}}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(TdsKVBackendTest, BatchExistsAsync) {
  TdsBackendOptions options;
  options.root_dir = test_dir_;
  options.storage_io_thread_pool_size = 8;
  TdsKVBackend backend("tds_exists", options);

  constexpr int kNumKeys = 16;
  std::vector<uint8_t> payload = MakeDeterministicPayload(2048, 7);
  HostBufferDescriptor slice{.ptr = payload.data(), .size = payload.size()};

  std::vector<BlockKey> keys;
  for (int i = 0; i < kNumKeys; ++i) {
    BlockKey key;
    key.block_hash = absl::StrCat("block_", i);
    key.resolved_key = absl::StrCat(test_dir_, "/exists_", i, ".bin");
    key.offset = 0;
    key.size = static_cast<int64_t>(payload.size());
    keys.push_back(key);

    if (i % 2 == 0) {
      absl::Notification write_done;
      absl::Status write_status;
      backend.WriteAsync(key, {slice}, payload.size(), [&](absl::Status s) {
        write_status = std::move(s);
        write_done.Notify();
      });
      write_done.WaitForNotification();
      ABSL_ASSERT_OK(write_status);
    }
  }

  absl::Notification batch_done;
  std::vector<absl::StatusOr<bool>> results;
  backend.BatchExistsAsync(keys, [&](std::vector<absl::StatusOr<bool>> res) {
    results = std::move(res);
    batch_done.Notify();
  });
  batch_done.WaitForNotification();

  ASSERT_EQ(results.size(), kNumKeys);
  for (int i = 0; i < kNumKeys; ++i) {
    EXPECT_THAT(results[i], IsOkAndHolds(i % 2 == 0)) << "Failed at key " << i;
  }
}

TEST_F(TdsKVBackendTest, UnalignedNonZeroOffsetPreservesAdjacentBytes) {
  TdsBackendOptions options;
  options.root_dir = test_dir_;
  options.alignment = kAlignment;
  TdsKVBackend backend("tds_offset", options);

  // First write a 6000-byte baseline block at offset 0.
  std::vector<uint8_t> base_payload = MakeDeterministicPayload(6000, 42);
  HostBufferDescriptor base_desc{.ptr = base_payload.data(),
                                 .size = base_payload.size()};
  BlockKey base_key;
  base_key.block_hash = "offset_block";
  base_key.resolved_key = absl::StrCat(test_dir_, "/offset_block.bin");
  base_key.offset = 0;
  base_key.size = 6000;

  absl::Notification done1;
  absl::Status status1;
  backend.WriteAsync(base_key, {base_desc}, base_payload.size(),
                     [&](absl::Status s) {
                       status1 = std::move(s);
                       done1.Notify();
                     });
  done1.WaitForNotification();
  ABSL_ASSERT_OK(status1);

  // Overwrite 777 bytes at unaligned offset 500 inside the existing file.
  std::vector<uint8_t> patch_payload = MakeDeterministicPayload(777, 84);
  HostBufferDescriptor patch_desc{.ptr = patch_payload.data(),
                                  .size = patch_payload.size()};
  BlockKey patch_key = base_key;
  patch_key.offset = 500;
  patch_key.size = 777;

  absl::Notification done2;
  absl::Status status2;
  backend.WriteAsync(patch_key, {patch_desc}, patch_payload.size(),
                     [&](absl::Status s) {
                       status2 = std::move(s);
                       done2.Notify();
                     });
  done2.WaitForNotification();
  ABSL_ASSERT_OK(status2);

  // File size must remain 6000.
  EXPECT_EQ(std::filesystem::file_size(base_key.resolved_key), 6000);

  // Read back the entire 6000 bytes and verify head, patch, and tail bytes.
  std::vector<uint8_t> read_full(6000, 0);
  HostBufferDescriptor read_desc{.ptr = read_full.data(),
                                 .size = read_full.size()};
  absl::Notification done3;
  absl::Status status3;
  backend.ReadAsync(base_key, {read_desc}, read_full.size(),
                    [&](absl::Status s) {
                      status3 = std::move(s);
                      done3.Notify();
                    });
  done3.WaitForNotification();
  ABSL_ASSERT_OK(status3);

  std::vector<uint8_t> expected = base_payload;
  std::memcpy(expected.data() + 500, patch_payload.data(), 777);
  EXPECT_EQ(read_full, expected);
}

TEST_F(TdsKVBackendTest, SubSliceOfRegisteredPoolUsesDirectZeroCopy) {
  TdsBackendOptions options;
  options.root_dir = test_dir_;
  options.alignment = kAlignment;
  TdsKVBackend backend("tds_pool", options);

  constexpr size_t kPoolSize = 64 * 1024;
  AlignedBuffer pool(kPoolSize);
  ASSERT_NE(pool.data(), nullptr);
  std::vector<uint8_t> fill = MakeDeterministicPayload(kPoolSize, 55);
  std::memcpy(pool.data(), fill.data(), kPoolSize);

  ABSL_ASSERT_OK(backend.RegisterBuffer(pool.data(), kPoolSize));

  // Duplicate/overlapping registration should fail with AlreadyExists.
  EXPECT_THAT(backend.RegisterBuffer(pool.data() + 4096, 4096),
              StatusIs(absl::StatusCode::kAlreadyExists));

  // Write two slices that are sub-regions of the single registered pool.
  std::vector<HostBufferDescriptor> slices = {
      HostBufferDescriptor{.ptr = pool.data() + 4096, .size = 8192},
      HostBufferDescriptor{.ptr = pool.data() + 32768, .size = 4096},
  };
  BlockKey key;
  key.block_hash = "pool_block";
  key.resolved_key = absl::StrCat(test_dir_, "/pool_block.bin");
  key.offset = 0;
  key.size = 12288;

  absl::Notification write_done;
  absl::Status write_status;
  backend.WriteAsync(key, slices, 12288, [&](absl::Status s) {
    write_status = std::move(s);
    write_done.Notify();
  });
  write_done.WaitForNotification();
  ABSL_ASSERT_OK(write_status);

  TdsStats stats = backend.stats();
  EXPECT_EQ(stats.direct_ops, 1);
  EXPECT_EQ(stats.bounced_ops, 0);

  backend.ResetStats();
  EXPECT_EQ(backend.stats().ops, 0);
  EXPECT_EQ(backend.stats().direct_ops, 0);

  ABSL_EXPECT_OK(backend.UnregisterBuffer(pool.data()));
}

TEST_F(
    TdsKVBackendTest,
    UnregisteredAlignedSlicesUseZeroCopyByDefaultAndBounceWhenRegistrationRequired) {
  constexpr size_t kSliceSize = 8192;
  AlignedBuffer src(kSliceSize);
  AlignedBuffer dst(kSliceSize);
  ASSERT_NE(src.data(), nullptr);
  ASSERT_NE(dst.data(), nullptr);
  std::vector<uint8_t> expected = MakeDeterministicPayload(kSliceSize, 77);
  std::memcpy(src.data(), expected.data(), kSliceSize);

  // Case 1: Default `require_registration = false` uses O_DIRECT zero-copy
  // (`direct_ops`) for 4KB-aligned [dram: User host_buf] slices without needing
  // RegisterBuffer().
  {
    TdsBackendOptions default_opts;
    default_opts.root_dir = test_dir_;
    default_opts.alignment = kAlignment;
    default_opts.use_direct_io = true;
    EXPECT_FALSE(default_opts.require_registration);

    TdsKVBackend default_backend("tds_default_unregistered", default_opts);
    BlockKey key{
        .block_hash = "unreg_zero_copy",
        .resolved_key = absl::StrCat(test_dir_, "/unreg_zero_copy.bin"),
        .offset = 0,
        .size = static_cast<int64_t>(kSliceSize),
    };
    HostBufferDescriptor write_slice{.ptr = src.data(), .size = kSliceSize};
    absl::Notification write_done;
    absl::Status write_status;
    default_backend.WriteAsync(key, {write_slice}, kSliceSize,
                               [&](absl::Status s) {
                                 write_status = std::move(s);
                                 write_done.Notify();
                               });
    write_done.WaitForNotification();
    ABSL_ASSERT_OK(write_status);
    EXPECT_EQ(default_backend.stats().direct_ops, 1);
    EXPECT_EQ(default_backend.stats().bounced_ops, 0);
  }

  // Case 2: When `require_registration = true` is configured, unregistered
  // 4KB-aligned slices take the 4KB bounce-buffer fallback (`bounced_ops`)
  // until registered.
  {
    absl::flat_hash_map<std::string, std::string> strict_props = {
        {"root_dir", test_dir_},
        {"require_registration", "true"},
    };
    TdsKVBackend strict_backend("tds_strict_registration", strict_props);
    EXPECT_TRUE(strict_backend.options().require_registration);

    BlockKey key{
        .block_hash = "strict_bounced",
        .resolved_key = absl::StrCat(test_dir_, "/strict_bounced.bin"),
        .offset = 0,
        .size = static_cast<int64_t>(kSliceSize),
    };
    HostBufferDescriptor write_slice{.ptr = src.data(), .size = kSliceSize};
    absl::Notification write_done;
    absl::Status write_status;
    strict_backend.WriteAsync(key, {write_slice}, kSliceSize,
                              [&](absl::Status s) {
                                write_status = std::move(s);
                                write_done.Notify();
                              });
    write_done.WaitForNotification();
    ABSL_ASSERT_OK(write_status);
    EXPECT_EQ(strict_backend.stats().direct_ops, 0);
    EXPECT_EQ(strict_backend.stats().bounced_ops, 1);
  }
}

TEST_F(TdsKVBackendTest,
       PosixPathMapperAndKVCacheStoreBackendFactoryIntegration) {
  absl::flat_hash_map<std::string, std::string> props = {
      {"root_dir", test_dir_},
      {"model_name", "gemini_ultra"},
      {"tp_size", "8"},
      {"tp_rank", "3"},
  };
  TdsKVBackend backend("tds_mapper", props);
  ASSERT_NE(backend.mapper(), nullptr);
  EXPECT_EQ(backend.mapper()->tp_size(), 8);

  // Construct a binary block_hash containing embedded NUL (0x00) and '/' (0x2F)
  // bytes to verify filesystem-safe hex encoding via PosixPathMapper.
  const std::string binary_hash("\xab\xcd\x00\x2f\xef\x01", 6);
  absl::StatusOr<BlockKey> mapped = backend.mapper()->MapKey(binary_hash);
  ABSL_ASSERT_OK(mapped);
  EXPECT_EQ(mapped->block_hash, binary_hash);
  EXPECT_EQ(mapped->resolved_key,
            absl::StrCat(test_dir_,
                         "/gemini_ultra/tp8_r3/abc/d0/abcd002fef01.bin"));

  // Verify that "tds" (kTdsBackendName) is registered in
  // KVCacheStoreBackendFactory and creates a working PosixKVCacheStoreBackend
  // backed by TdsKVBackend.
  BackendConfig store_cfg;
  store_cfg.type = std::string(kTdsBackendName);
  store_cfg.properties = {
      {"root_dir", test_dir_},
      {"model_name", "gemini_ultra"},
      {"tp_size", "1"},
      {"capacity_bytes", "1048576"},
  };
  absl::StatusOr<std::shared_ptr<KVCacheStoreBackend>> store_backend =
      KVCacheStoreBackendFactory::Instance().Create(store_cfg,
                                                    /*controller=*/nullptr);
  ABSL_ASSERT_OK(store_backend);
  ASSERT_NE(*store_backend, nullptr);
  EXPECT_EQ((*store_backend)->name(), kTdsBackendName);
  EXPECT_EQ((*store_backend)->GetCapacity(), 1048576);

  // Write a block at rank 0 via TdsKVBackend and verify that
  // PosixKVCacheStoreBackend::Lookup discovers it via TdsKVBackend's
  // BatchExistsAsync and returns BlockStatus::SHARED_STORAGE.
  TdsKVBackend rank0_writer(
      "tds_rank0",
      {{"root_dir", test_dir_}, {"model_name", "gemini_ultra"}, {"tp_size", "1"}, {"tp_rank", "0"}});
  AlignedBuffer buf(4096);
  std::memset(buf.data(), 0x5A, 4096);
  absl::Notification write_done;
  absl::Status write_status;
  rank0_writer.WriteAsync(
      BlockKey{.block_hash = binary_hash, .resolved_key = "", .offset = 0, .size = 4096},
      {HostBufferDescriptor{.ptr = buf.data(), .size = 4096}}, 4096,
      [&](absl::Status s) {
        write_status = std::move(s);
        write_done.Notify();
      });
  write_done.WaitForNotification();
  ABSL_ASSERT_OK(write_status);

  absl::StatusOr<BlockSliceList> lookup_hits =
      (*store_backend)->Lookup({binary_hash, "missing_hash"});
  ABSL_ASSERT_OK(lookup_hits);
  ASSERT_EQ(lookup_hits->size(), 1);
  EXPECT_EQ((*lookup_hits)[0].first, binary_hash);
  EXPECT_EQ((*lookup_hits)[0].second.status, BlockStatus::SHARED_STORAGE);
}

TEST_F(TdsKVBackendTest, ConcurrentMultiSliceWriteLookupAndRead) {
  constexpr int kNumBlocks = 12;
  constexpr size_t kSliceBytes = 4096;
  constexpr size_t kSlicesPerBlock = 3;
  constexpr size_t kBlockBytes = kSliceBytes * kSlicesPerBlock;

  BackendConfig store_cfg;
  store_cfg.type = std::string(kTdsBackendName);
  store_cfg.properties = {
      {"root_dir", test_dir_},
      {"model_name", "concurrent_model"},
      {"tp_size", "1"},
      {"storage_io_thread_pool_size", "8"},
  };
  absl::StatusOr<std::shared_ptr<KVCacheStoreBackend>> store_backend =
      KVCacheStoreBackendFactory::Instance().Create(store_cfg,
                                                    /*controller=*/nullptr);
  ABSL_ASSERT_OK(store_backend);

  TdsKVBackend worker_backend("tds_worker", store_cfg.properties);

  std::vector<std::unique_ptr<AlignedBuffer>> write_arena;
  std::vector<std::string> block_hashes;
  for (int b = 0; b < kNumBlocks; ++b) {
    block_hashes.push_back(absl::StrCat("concurrent_block_hash_", b));
    for (size_t s = 0; s < kSlicesPerBlock; ++s) {
      auto buf = std::make_unique<AlignedBuffer>(kSliceBytes);
      std::vector<uint8_t> payload = MakeDeterministicPayload(
          kSliceBytes, static_cast<uint8_t>(b * 7 + s + 1));
      std::memcpy(buf->data(), payload.data(), kSliceBytes);
      write_arena.push_back(std::move(buf));
    }
  }

  // Issue all 12 multi-slice block writes concurrently.
  std::atomic<int> remaining_writes{kNumBlocks};
  absl::Notification all_writes_done;
  std::vector<absl::Status> write_statuses(kNumBlocks);
  for (int b = 0; b < kNumBlocks; ++b) {
    std::vector<HostBufferDescriptor> slices;
    for (size_t s = 0; s < kSlicesPerBlock; ++s) {
      slices.push_back(HostBufferDescriptor{
          .ptr = write_arena[b * kSlicesPerBlock + s]->data(),
          .size = kSliceBytes});
    }
    BlockKey key{.block_hash = block_hashes[b],
                 .resolved_key = "",
                 .offset = 0,
                 .size = static_cast<int64_t>(kBlockBytes)};
    worker_backend.WriteAsync(
        key, slices, kBlockBytes, [&, b](absl::Status status) {
          write_statuses[b] = std::move(status);
          if (remaining_writes.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            all_writes_done.Notify();
          }
        });
  }
  all_writes_done.WaitForNotification();
  for (int b = 0; b < kNumBlocks; ++b) {
    ABSL_ASSERT_OK(write_statuses[b]);
  }

  // Verify Coordinator Lookup finds all 12 blocks in SHARED_STORAGE.
  absl::StatusOr<BlockSliceList> hits = (*store_backend)->Lookup(block_hashes);
  ABSL_ASSERT_OK(hits);
  ASSERT_EQ(hits->size(), kNumBlocks);
  for (int b = 0; b < kNumBlocks; ++b) {
    EXPECT_EQ((*hits)[b].first, block_hashes[b]);
    EXPECT_EQ((*hits)[b].second.status, BlockStatus::SHARED_STORAGE);
  }

  // Issue all 12 multi-slice block reads concurrently and verify payloads.
  std::vector<std::unique_ptr<AlignedBuffer>> read_arena;
  for (size_t i = 0; i < kNumBlocks * kSlicesPerBlock; ++i) {
    auto buf = std::make_unique<AlignedBuffer>(kSliceBytes);
    std::memset(buf->data(), 0, kSliceBytes);
    read_arena.push_back(std::move(buf));
  }

  std::atomic<int> remaining_reads{kNumBlocks};
  absl::Notification all_reads_done;
  std::vector<absl::Status> read_statuses(kNumBlocks);
  for (int b = 0; b < kNumBlocks; ++b) {
    std::vector<HostBufferDescriptor> slices;
    for (size_t s = 0; s < kSlicesPerBlock; ++s) {
      slices.push_back(HostBufferDescriptor{
          .ptr = read_arena[b * kSlicesPerBlock + s]->data(),
          .size = kSliceBytes});
    }
    BlockKey key{.block_hash = block_hashes[b],
                 .resolved_key = "",
                 .offset = 0,
                 .size = static_cast<int64_t>(kBlockBytes)};
    worker_backend.ReadAsync(
        key, slices, kBlockBytes, [&, b](absl::Status status) {
          read_statuses[b] = std::move(status);
          if (remaining_reads.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            all_reads_done.Notify();
          }
        });
  }
  all_reads_done.WaitForNotification();
  for (int b = 0; b < kNumBlocks; ++b) {
    ABSL_ASSERT_OK(read_statuses[b]);
    for (size_t s = 0; s < kSlicesPerBlock; ++s) {
      const size_t idx = b * kSlicesPerBlock + s;
      EXPECT_EQ(std::memcmp(read_arena[idx]->data(), write_arena[idx]->data(),
                            kSliceBytes),
                0);
    }
  }

  EXPECT_EQ(worker_backend.stats().direct_ops, 2 * kNumBlocks);
  EXPECT_EQ(worker_backend.stats().bounced_ops, 0);
}

TEST_F(TdsKVBackendTest, ErrorHandlingAndStatsInvariant) {
  TdsBackendOptions options;
  options.root_dir = test_dir_;
  TdsKVBackend backend("tds_errors", options);

  AlignedBuffer buf(4096);

  // 1. Reading a non-existent file returns kNotFound and increments failed_ops.
  {
    absl::Notification done;
    absl::Status status;
    backend.ReadAsync(
        BlockKey{.block_hash = "non_existent",
                 .resolved_key = absl::StrCat(test_dir_, "/missing.bin"),
                 .offset = 0,
                 .size = 4096},
        {HostBufferDescriptor{.ptr = buf.data(), .size = 4096}}, 4096,
        [&](absl::Status s) {
          status = std::move(s);
          done.Notify();
        });
    done.WaitForNotification();
    EXPECT_THAT(status, StatusIs(absl::StatusCode::kNotFound));
  }

  // 2. Writing with mismatched total_bytes returns kInvalidArgument and
  // increments failed_ops.
  {
    absl::Notification done;
    absl::Status status;
    backend.WriteAsync(
        BlockKey{.block_hash = "mismatch",
                 .resolved_key = absl::StrCat(test_dir_, "/mismatch.bin"),
                 .offset = 0,
                 .size = 8192},
        {HostBufferDescriptor{.ptr = buf.data(), .size = 4096}}, 8192,
        [&](absl::Status s) {
          status = std::move(s);
          done.Notify();
        });
    done.WaitForNotification();
    EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // 3. Register/Unregister invalid arguments.
  EXPECT_THAT(backend.RegisterBuffer(nullptr, 4096),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(backend.RegisterBuffer(buf.data(), 0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(backend.UnregisterBuffer(buf.data()),
              StatusIs(absl::StatusCode::kNotFound));

  TdsStats stats = backend.stats();
  EXPECT_EQ(stats.ops, 2);
  EXPECT_EQ(stats.failed_ops, 2);
  EXPECT_EQ(stats.direct_ops, 0);
  EXPECT_EQ(stats.bounced_ops, 0);
  EXPECT_EQ(stats.ops, stats.direct_ops + stats.bounced_ops + stats.failed_ops);
}

}  // namespace
}  // namespace storage
}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden
