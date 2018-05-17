// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-
// vim: ts=8 sw=2 smarttab

#include "test/librbd/test_mock_fixture.h"
#include "test/librbd/test_support.h"
#include "test/librbd/mock/MockImageCtx.h"
#include "test/librbd/mock/MockContextWQ.h"
#include "test/librados_test_stub/MockTestMemIoCtxImpl.h"
#include "test/librados_test_stub/MockTestMemRadosClient.h"
#include "librbd/ImageState.h"
#include "librbd/Operations.h"
#include "librbd/image/TypeTraits.h"
#include "librbd/image/CreateRequest.h"
#include "librbd/image/RemoveRequest.h"
#include "librbd/image/RefreshRequest.h"
#include "librbd/mirror/EnableRequest.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace librbd {
namespace {

struct MockTestImageCtx : public MockImageCtx {
  static MockTestImageCtx* s_instance;
  static MockTestImageCtx* create(const std::string &image_name,
                                  const std::string &image_id,
                                  const char *snap, librados::IoCtx& p,
                                  bool read_only) {
    assert(s_instance != nullptr);
    return s_instance;
  }

  MockTestImageCtx(ImageCtx &image_ctx) : MockImageCtx(image_ctx) {
    s_instance = this;
  }
};

MockTestImageCtx* MockTestImageCtx::s_instance = nullptr;

} // anonymous namespace

namespace image {

template <>
struct CreateRequest<MockTestImageCtx> {
  Context* on_finish = nullptr;
  static CreateRequest* s_instance;
  static CreateRequest* create(IoCtx &ioctx, const std::string &image_name,
                               const std::string &image_id, uint64_t size,
                               const ImageOptions &image_options,
                               const std::string &non_primary_global_image_id,
                               const std::string &primary_mirror_uuid,
                               bool skip_mirror_enable,
                               ContextWQ *op_work_queue,
                               Context *on_finish) {
    assert(s_instance != nullptr);
    s_instance->on_finish = on_finish;
    return s_instance;
  }

  MOCK_METHOD0(send, void());

  CreateRequest() {
    s_instance = this;
  }
};

CreateRequest<MockTestImageCtx>* CreateRequest<MockTestImageCtx>::s_instance = nullptr;

template <>
struct RefreshRequest<MockTestImageCtx> {
  Context* on_finish = nullptr;
  static RefreshRequest* s_instance;
  static RefreshRequest* create(MockTestImageCtx &image_ctx,
                                bool acquiring_lock, bool skip_open_parent,
                                Context *on_finish) {
    assert(s_instance != nullptr);
    s_instance->on_finish = on_finish;
    return s_instance;
  }

  MOCK_METHOD0(send, void());

  RefreshRequest() {
    s_instance = this;
  }
};

RefreshRequest<MockTestImageCtx>* RefreshRequest<MockTestImageCtx>::s_instance = nullptr;

template <>
struct RemoveRequest<MockTestImageCtx> {
  Context* on_finish = nullptr;
  static RemoveRequest* s_instance;
  static RemoveRequest* create(librados::IoCtx &ioctx,
                               const std::string &image_name,
                               const std::string &image_id,
                               bool force, bool from_trash_remove,
                               ProgressContext &prog_ctx,
                               ContextWQ *op_work_queue,
                               Context *on_finish) {
    assert(s_instance != nullptr);
    s_instance->on_finish = on_finish;
    return s_instance;
  }

  MOCK_METHOD0(send, void());

  RemoveRequest() {
    s_instance = this;
  }
};

RemoveRequest<MockTestImageCtx>* RemoveRequest<MockTestImageCtx>::s_instance = nullptr;

} // namespace image

namespace mirror {

template <>
struct EnableRequest<MockTestImageCtx> {
  Context* on_finish = nullptr;
  static EnableRequest* s_instance;
  static EnableRequest* create(librados::IoCtx &io_ctx,
                               const std::string &image_id,
                               const std::string &non_primary_global_image_id,
                               MockContextWQ *op_work_queue,
                               Context *on_finish) {
    assert(s_instance != nullptr);
    s_instance->on_finish = on_finish;
    return s_instance;
  }

  MOCK_METHOD0(send, void());

  EnableRequest() {
    s_instance = this;
  }
};

EnableRequest<MockTestImageCtx>* EnableRequest<MockTestImageCtx>::s_instance = nullptr;

} // namespace mirror
} // namespace librbd

// template definitions
#include "librbd/image/CloneRequest.cc"

namespace librbd {
namespace image {

using ::testing::_;
using ::testing::Invoke;
using ::testing::InvokeWithoutArgs;
using ::testing::InSequence;
using ::testing::Return;
using ::testing::StrEq;
using ::testing::WithArg;

class TestMockImageCloneRequest : public TestMockFixture {
public:
  typedef CloneRequest<MockTestImageCtx> MockCloneRequest;
  typedef CreateRequest<MockTestImageCtx> MockCreateRequest;
  typedef RefreshRequest<MockTestImageCtx> MockRefreshRequest;
  typedef RemoveRequest<MockTestImageCtx> MockRemoveRequest;
  typedef mirror::EnableRequest<MockTestImageCtx> MockMirrorEnableRequest;

  void SetUp() override {
    TestMockFixture::SetUp();

    ASSERT_EQ(0, _rados.conf_set("rbd_default_clone_format", "2"));

    ASSERT_EQ(0, open_image(m_image_name, &image_ctx));
    ASSERT_EQ(0, image_ctx->operations->snap_create(
                   cls::rbd::UserSnapshotNamespace{}, "snap"));
    if (is_feature_enabled(RBD_FEATURE_LAYERING)) {
      ASSERT_EQ(0, image_ctx->operations->snap_protect(
                     cls::rbd::UserSnapshotNamespace{}, "snap"));

      uint64_t snap_id = image_ctx->snap_ids[
        {cls::rbd::UserSnapshotNamespace{}, "snap"}];
      ASSERT_NE(CEPH_NOSNAP, snap_id);

      C_SaferCond ctx;
      image_ctx->state->snap_set(snap_id, &ctx);
      ASSERT_EQ(0, ctx.wait());
    }
  }

  void expect_get_min_compat_client(int8_t min_compat_client, int r) {
    auto mock_rados_client = get_mock_io_ctx(m_ioctx).get_mock_rados_client();
    EXPECT_CALL(*mock_rados_client, get_min_compatible_client(_, _))
      .WillOnce(Invoke([min_compat_client, r](int8_t* min, int8_t* required_min) {
                  *min = min_compat_client;
                  *required_min = min_compat_client;
                  return r;
                }));
  }

  void expect_get_image_size(MockTestImageCtx &mock_image_ctx, uint64_t snap_id,
                             uint64_t size) {
    EXPECT_CALL(mock_image_ctx, get_image_size(snap_id))
      .WillOnce(Return(size));
  }

  void expect_is_snap_protected(MockImageCtx &mock_image_ctx, bool is_protected,
                                int r) {
    EXPECT_CALL(mock_image_ctx, is_snap_protected(_, _))
      .WillOnce(WithArg<1>(Invoke([is_protected, r](bool* is_prot) {
                             *is_prot = is_protected;
                             return r;
                           })));
  }

  void expect_create(MockCreateRequest& mock_create_request, int r) {
    EXPECT_CALL(mock_create_request, send())
      .WillOnce(Invoke([this, &mock_create_request, r]() {
                  image_ctx->op_work_queue->queue(mock_create_request.on_finish, r);
                }));
  }

  void expect_open(MockImageCtx &mock_image_ctx, int r) {
    EXPECT_CALL(*mock_image_ctx.state, open(true, _))
      .WillOnce(WithArg<1>(Invoke([this, r](Context* ctx) {
                             image_ctx->op_work_queue->queue(ctx, r);
                           })));
  }

  void expect_set_parent(MockImageCtx &mock_image_ctx, int r) {
    EXPECT_CALL(get_mock_io_ctx(mock_image_ctx.md_ctx),
                exec(mock_image_ctx.header_oid, _, StrEq("rbd"),
                     StrEq("set_parent"), _, _, _))
      .WillOnce(InvokeWithoutArgs([r]() {
                  return r;
                }));
  }

  void expect_op_features_set(librados::IoCtx& io_ctx,
                              const std::string& clone_id, int r) {
    bufferlist bl;
    encode(static_cast<uint64_t>(RBD_OPERATION_FEATURE_CLONE_CHILD), bl);
    encode(static_cast<uint64_t>(RBD_OPERATION_FEATURE_CLONE_CHILD), bl);

    EXPECT_CALL(get_mock_io_ctx(io_ctx),
                exec(util::header_name(clone_id), _, StrEq("rbd"),
                     StrEq("op_features_set"), ContentsEqual(bl), _, _))
      .WillOnce(Return(r));
  }

  void expect_child_attach(MockImageCtx &mock_image_ctx, int r) {
    bufferlist bl;
    encode(mock_image_ctx.snap_id, bl);
    encode(cls::rbd::ChildImageSpec{m_ioctx.get_id(), mock_image_ctx.id}, bl);

    EXPECT_CALL(get_mock_io_ctx(mock_image_ctx.md_ctx),
                exec(mock_image_ctx.header_oid, _, StrEq("rbd"),
                     StrEq("child_attach"), ContentsEqual(bl), _, _))
      .WillOnce(Return(r));
  }

  void expect_add_child(librados::IoCtx& io_ctx, int r) {
    EXPECT_CALL(get_mock_io_ctx(io_ctx),
                exec(RBD_CHILDREN, _, StrEq("rbd"), StrEq("add_child"), _, _, _))
      .WillOnce(Return(r));
  }

  void expect_refresh(MockRefreshRequest& mock_refresh_request, int r) {
    EXPECT_CALL(mock_refresh_request, send())
      .WillOnce(Invoke([this, &mock_refresh_request, r]() {
                  image_ctx->op_work_queue->queue(mock_refresh_request.on_finish, r);
                }));
  }

  void expect_metadata_list(MockTestImageCtx &mock_image_ctx,
                            const std::map<std::string, bufferlist>& metadata,
                            int r) {
    bufferlist out_bl;
    encode(metadata, out_bl);

    EXPECT_CALL(get_mock_io_ctx(mock_image_ctx.md_ctx),
                exec(mock_image_ctx.header_oid, _, StrEq("rbd"), StrEq("metadata_list"), _, _, _))
      .WillOnce(WithArg<5>(Invoke([out_bl, r](bufferlist *out) {
                             *out = out_bl;
                             return r;
                           })));
  }

  void expect_metadata_set(librados::IoCtx& io_ctx,
                           MockTestImageCtx& mock_image_ctx,
                           const std::map<std::string, bufferlist>& metadata,
                           int r) {
    bufferlist in_bl;
    encode(metadata, in_bl);

    EXPECT_CALL(get_mock_io_ctx(io_ctx),
                exec(mock_image_ctx.header_oid, _, StrEq("rbd"), StrEq("metadata_set"),
                     ContentsEqual(in_bl), _, _))
      .WillOnce(Return(r));
  }

  void expect_test_features(MockTestImageCtx &mock_image_ctx,
                            uint64_t features, bool enabled) {
    EXPECT_CALL(mock_image_ctx, test_features(features))
      .WillOnce(Return(enabled));
  }

  void expect_mirror_mode_get(MockTestImageCtx &mock_image_ctx,
                              cls::rbd::MirrorMode mirror_mode, int r) {
    bufferlist out_bl;
    encode(static_cast<uint32_t>(mirror_mode), out_bl);

    EXPECT_CALL(get_mock_io_ctx(mock_image_ctx.md_ctx),
                exec(RBD_MIRRORING, _, StrEq("rbd"), StrEq("mirror_mode_get"),
                     _, _, _))
      .WillOnce(WithArg<5>(Invoke([out_bl, r](bufferlist* out) {
                             *out = out_bl;
                             return r;
                           })));
  }

  void expect_mirror_enable(MockMirrorEnableRequest& mock_mirror_enable_request,
                            int r) {
    EXPECT_CALL(mock_mirror_enable_request, send())
      .WillOnce(Invoke([this, &mock_mirror_enable_request, r]() {
                  image_ctx->op_work_queue->queue(mock_mirror_enable_request.on_finish, r);
                }));
  }

  void expect_close(MockImageCtx &mock_image_ctx, int r) {
    EXPECT_CALL(*mock_image_ctx.state, close(_))
      .WillOnce(Invoke([this, r](Context* ctx) {
                  image_ctx->op_work_queue->queue(ctx, r);
                }));
    EXPECT_CALL(mock_image_ctx, destroy());
  }

  void expect_remove(MockRemoveRequest& mock_remove_request, int r) {
    EXPECT_CALL(mock_remove_request, send())
      .WillOnce(Invoke([this, &mock_remove_request, r]() {
                  image_ctx->op_work_queue->queue(mock_remove_request.on_finish, r);
                }));
  }

  librbd::ImageCtx *image_ctx;
};

TEST_F(TestMockImageCloneRequest, SuccessV1) {
  REQUIRE_FEATURE(RBD_FEATURE_LAYERING);
  ASSERT_EQ(0, _rados.conf_set("rbd_default_clone_format", "1"));

  MockTestImageCtx mock_image_ctx(*image_ctx);
  expect_op_work_queue(mock_image_ctx);

  InSequence seq;
  expect_get_image_size(mock_image_ctx, mock_image_ctx.snaps.front(), 123);
  expect_is_snap_protected(mock_image_ctx, true, 0);

  MockCreateRequest mock_create_request;
  expect_create(mock_create_request, 0);

  expect_open(mock_image_ctx, 0);
  expect_set_parent(mock_image_ctx, 0);
  expect_add_child(m_ioctx, 0);

  MockRefreshRequest mock_refresh_request;
  expect_refresh(mock_refresh_request, 0);
  expect_is_snap_protected(mock_image_ctx, true, 0);

  expect_metadata_list(mock_image_ctx, {{"key", {}}}, 0);
  expect_metadata_set(m_ioctx, mock_image_ctx, {{"key", {}}}, 0);

  MockMirrorEnableRequest mock_mirror_enable_request;
  if (is_feature_enabled(RBD_FEATURE_JOURNALING)) {
    expect_test_features(mock_image_ctx, RBD_FEATURE_JOURNALING, true);
    expect_mirror_mode_get(mock_image_ctx, cls::rbd::MIRROR_MODE_POOL, 0);

    expect_mirror_enable(mock_mirror_enable_request, 0);
  } else {
    expect_test_features(mock_image_ctx, RBD_FEATURE_JOURNALING, false);
  }

  expect_close(mock_image_ctx, 0);

  C_SaferCond ctx;
  ImageOptions clone_opts;
  auto req = new MockCloneRequest(&mock_image_ctx, m_ioctx, "clone name",
                                  "clone id", clone_opts, "", "",
                                  image_ctx->op_work_queue, &ctx);
  req->send();
  ASSERT_EQ(0, ctx.wait());
}

TEST_F(TestMockImageCloneRequest, SuccessV2) {
  REQUIRE_FEATURE(RBD_FEATURE_LAYERING);

  MockTestImageCtx mock_image_ctx(*image_ctx);
  expect_op_work_queue(mock_image_ctx);

  InSequence seq;
  expect_get_image_size(mock_image_ctx, mock_image_ctx.snaps.front(), 123);
  expect_is_snap_protected(mock_image_ctx, true, 0);

  MockCreateRequest mock_create_request;
  expect_create(mock_create_request, 0);

  expect_open(mock_image_ctx, 0);
  expect_set_parent(mock_image_ctx, 0);

  expect_op_features_set(m_ioctx, mock_image_ctx.id, 0);
  expect_child_attach(mock_image_ctx, 0);

  expect_metadata_list(mock_image_ctx, {{"key", {}}}, 0);
  expect_metadata_set(m_ioctx, mock_image_ctx, {{"key", {}}}, 0);

  MockMirrorEnableRequest mock_mirror_enable_request;
  if (is_feature_enabled(RBD_FEATURE_JOURNALING)) {
    expect_test_features(mock_image_ctx, RBD_FEATURE_JOURNALING, true);
    expect_mirror_mode_get(mock_image_ctx, cls::rbd::MIRROR_MODE_POOL, 0);

    expect_mirror_enable(mock_mirror_enable_request, 0);
  } else {
    expect_test_features(mock_image_ctx, RBD_FEATURE_JOURNALING, false);
  }

  expect_close(mock_image_ctx, 0);

  C_SaferCond ctx;
  ImageOptions clone_opts;
  auto req = new MockCloneRequest(&mock_image_ctx, m_ioctx, "clone name",
                                  "clone id", clone_opts, "", "",
                                  image_ctx->op_work_queue, &ctx);
  req->send();
  ASSERT_EQ(0, ctx.wait());
}

TEST_F(TestMockImageCloneRequest, SuccessAuto) {
  REQUIRE_FEATURE(RBD_FEATURE_LAYERING);
  ASSERT_EQ(0, _rados.conf_set("rbd_default_clone_format", "auto"));

  MockTestImageCtx mock_image_ctx(*image_ctx);
  expect_op_work_queue(mock_image_ctx);

  InSequence seq;
  expect_get_min_compat_client(1, 0);
  expect_get_image_size(mock_image_ctx, mock_image_ctx.snaps.front(), 123);
  expect_is_snap_protected(mock_image_ctx, true, 0);

  MockCreateRequest mock_create_request;
  expect_create(mock_create_request, 0);

  expect_open(mock_image_ctx, 0);
  expect_set_parent(mock_image_ctx, 0);
  expect_add_child(m_ioctx, 0);

  MockRefreshRequest mock_refresh_request;
  expect_refresh(mock_refresh_request, 0);
  expect_is_snap_protected(mock_image_ctx, true, 0);

  expect_metadata_list(mock_image_ctx, {{"key", {}}}, 0);
  expect_metadata_set(m_ioctx, mock_image_ctx, {{"key", {}}}, 0);

  MockMirrorEnableRequest mock_mirror_enable_request;
  if (is_feature_enabled(RBD_FEATURE_JOURNALING)) {
    expect_test_features(mock_image_ctx, RBD_FEATURE_JOURNALING, true);
    expect_mirror_mode_get(mock_image_ctx, cls::rbd::MIRROR_MODE_POOL, 0);

    expect_mirror_enable(mock_mirror_enable_request, 0);
  } else {
    expect_test_features(mock_image_ctx, RBD_FEATURE_JOURNALING, false);
  }

  expect_close(mock_image_ctx, 0);

  C_SaferCond ctx;
  ImageOptions clone_opts;
  auto req = new MockCloneRequest(&mock_image_ctx, m_ioctx, "clone name",
                                  "clone id", clone_opts, "", "",
                                  image_ctx->op_work_queue, &ctx);
  req->send();
  ASSERT_EQ(0, ctx.wait());
}

TEST_F(TestMockImageCloneRequest, CreateError) {
  REQUIRE_FEATURE(RBD_FEATURE_LAYERING);

  MockTestImageCtx mock_image_ctx(*image_ctx);
  expect_op_work_queue(mock_image_ctx);

  InSequence seq;
  expect_get_image_size(mock_image_ctx, mock_image_ctx.snaps.front(), 123);
  expect_is_snap_protected(mock_image_ctx, true, 0);

  MockCreateRequest mock_create_request;
  expect_create(mock_create_request, -EINVAL);

  C_SaferCond ctx;
  ImageOptions clone_opts;
  auto req = new MockCloneRequest(&mock_image_ctx, m_ioctx, "clone name",
                                  "clone id", clone_opts, "", "",
                                  image_ctx->op_work_queue, &ctx);
  req->send();
  ASSERT_EQ(-EINVAL, ctx.wait());
}

TEST_F(TestMockImageCloneRequest, OpenError) {
  REQUIRE_FEATURE(RBD_FEATURE_LAYERING);

  MockTestImageCtx mock_image_ctx(*image_ctx);
  expect_op_work_queue(mock_image_ctx);

  InSequence seq;
  expect_get_image_size(mock_image_ctx, mock_image_ctx.snaps.front(), 123);
  expect_is_snap_protected(mock_image_ctx, true, 0);

  MockCreateRequest mock_create_request;
  expect_create(mock_create_request, 0);

  expect_open(mock_image_ctx, -EINVAL);

  MockRemoveRequest mock_remove_request;
  expect_remove(mock_remove_request, 0);

  C_SaferCond ctx;
  ImageOptions clone_opts;
  auto req = new MockCloneRequest(&mock_image_ctx, m_ioctx, "clone name",
                                  "clone id", clone_opts, "", "",
                                  image_ctx->op_work_queue, &ctx);
  req->send();
  ASSERT_EQ(-EINVAL, ctx.wait());
}

TEST_F(TestMockImageCloneRequest, SetParentError) {
  REQUIRE_FEATURE(RBD_FEATURE_LAYERING);

  MockTestImageCtx mock_image_ctx(*image_ctx);
  expect_op_work_queue(mock_image_ctx);

  InSequence seq;
  expect_get_image_size(mock_image_ctx, mock_image_ctx.snaps.front(), 123);
  expect_is_snap_protected(mock_image_ctx, true, 0);

  MockCreateRequest mock_create_request;
  expect_create(mock_create_request, 0);

  expect_open(mock_image_ctx, 0);
  expect_set_parent(mock_image_ctx, -EINVAL);
  expect_close(mock_image_ctx, 0);

  MockRemoveRequest mock_remove_request;
  expect_remove(mock_remove_request, 0);

  C_SaferCond ctx;
  ImageOptions clone_opts;
  auto req = new MockCloneRequest(&mock_image_ctx, m_ioctx, "clone name",
                                  "clone id", clone_opts, "", "",
                                  image_ctx->op_work_queue, &ctx);
  req->send();
  ASSERT_EQ(-EINVAL, ctx.wait());
}

TEST_F(TestMockImageCloneRequest, AddChildError) {
  REQUIRE_FEATURE(RBD_FEATURE_LAYERING);
  ASSERT_EQ(0, _rados.conf_set("rbd_default_clone_format", "1"));

  MockTestImageCtx mock_image_ctx(*image_ctx);
  expect_op_work_queue(mock_image_ctx);

  InSequence seq;
  expect_get_image_size(mock_image_ctx, mock_image_ctx.snaps.front(), 123);
  expect_is_snap_protected(mock_image_ctx, true, 0);

  MockCreateRequest mock_create_request;
  expect_create(mock_create_request, 0);

  expect_open(mock_image_ctx, 0);
  expect_set_parent(mock_image_ctx, 0);
  expect_add_child(m_ioctx, -EINVAL);
  expect_close(mock_image_ctx, 0);

  MockRemoveRequest mock_remove_request;
  expect_remove(mock_remove_request, 0);

  C_SaferCond ctx;
  ImageOptions clone_opts;
  auto req = new MockCloneRequest(&mock_image_ctx, m_ioctx, "clone name",
                                  "clone id", clone_opts, "", "",
                                  image_ctx->op_work_queue, &ctx);
  req->send();
  ASSERT_EQ(-EINVAL, ctx.wait());
}

TEST_F(TestMockImageCloneRequest, RefreshError) {
  REQUIRE_FEATURE(RBD_FEATURE_LAYERING);
  ASSERT_EQ(0, _rados.conf_set("rbd_default_clone_format", "1"));

  MockTestImageCtx mock_image_ctx(*image_ctx);
  expect_op_work_queue(mock_image_ctx);

  InSequence seq;
  expect_get_image_size(mock_image_ctx, mock_image_ctx.snaps.front(), 123);
  expect_is_snap_protected(mock_image_ctx, true, 0);

  MockCreateRequest mock_create_request;
  expect_create(mock_create_request, 0);

  expect_open(mock_image_ctx, 0);
  expect_set_parent(mock_image_ctx, 0);
  expect_add_child(m_ioctx, 0);

  MockRefreshRequest mock_refresh_request;
  expect_refresh(mock_refresh_request, -EINVAL);

  expect_close(mock_image_ctx, 0);

  MockRemoveRequest mock_remove_request;
  expect_remove(mock_remove_request, 0);

  C_SaferCond ctx;
  ImageOptions clone_opts;
  auto req = new MockCloneRequest(&mock_image_ctx, m_ioctx, "clone name",
                                  "clone id", clone_opts, "", "",
                                  image_ctx->op_work_queue, &ctx);
  req->send();
  ASSERT_EQ(-EINVAL, ctx.wait());
}

TEST_F(TestMockImageCloneRequest, MetadataListError) {
  REQUIRE_FEATURE(RBD_FEATURE_LAYERING);
  ASSERT_EQ(0, _rados.conf_set("rbd_default_clone_format", "1"));

  MockTestImageCtx mock_image_ctx(*image_ctx);
  expect_op_work_queue(mock_image_ctx);

  InSequence seq;
  expect_get_image_size(mock_image_ctx, mock_image_ctx.snaps.front(), 123);
  expect_is_snap_protected(mock_image_ctx, true, 0);

  MockCreateRequest mock_create_request;
  expect_create(mock_create_request, 0);

  expect_open(mock_image_ctx, 0);
  expect_set_parent(mock_image_ctx, 0);
  expect_add_child(m_ioctx, 0);

  MockRefreshRequest mock_refresh_request;
  expect_refresh(mock_refresh_request, 0);
  expect_is_snap_protected(mock_image_ctx, true, 0);

  expect_metadata_list(mock_image_ctx, {{"key", {}}}, -EINVAL);

  expect_close(mock_image_ctx, 0);

  MockRemoveRequest mock_remove_request;
  expect_remove(mock_remove_request, 0);

  C_SaferCond ctx;
  ImageOptions clone_opts;
  auto req = new MockCloneRequest(&mock_image_ctx, m_ioctx, "clone name",
                                  "clone id", clone_opts, "", "",
                                  image_ctx->op_work_queue, &ctx);
  req->send();
  ASSERT_EQ(-EINVAL, ctx.wait());
}

TEST_F(TestMockImageCloneRequest, MetadataSetError) {
  REQUIRE_FEATURE(RBD_FEATURE_LAYERING);

  MockTestImageCtx mock_image_ctx(*image_ctx);
  expect_op_work_queue(mock_image_ctx);

  InSequence seq;
  expect_get_image_size(mock_image_ctx, mock_image_ctx.snaps.front(), 123);
  expect_is_snap_protected(mock_image_ctx, true, 0);

  MockCreateRequest mock_create_request;
  expect_create(mock_create_request, 0);

  expect_open(mock_image_ctx, 0);
  expect_set_parent(mock_image_ctx, 0);

  expect_op_features_set(m_ioctx, mock_image_ctx.id, 0);
  expect_child_attach(mock_image_ctx, 0);

  expect_metadata_list(mock_image_ctx, {{"key", {}}}, 0);
  expect_metadata_set(m_ioctx, mock_image_ctx, {{"key", {}}}, -EINVAL);

  expect_close(mock_image_ctx, 0);

  MockRemoveRequest mock_remove_request;
  expect_remove(mock_remove_request, 0);

  C_SaferCond ctx;
  ImageOptions clone_opts;
  auto req = new MockCloneRequest(&mock_image_ctx, m_ioctx, "clone name",
                                  "clone id", clone_opts, "", "",
                                  image_ctx->op_work_queue, &ctx);
  req->send();
  ASSERT_EQ(-EINVAL, ctx.wait());
}

TEST_F(TestMockImageCloneRequest, GetMirrorModeError) {
  REQUIRE_FEATURE(RBD_FEATURE_LAYERING | RBD_FEATURE_JOURNALING);
  ASSERT_EQ(0, _rados.conf_set("rbd_default_clone_format", "1"));

  MockTestImageCtx mock_image_ctx(*image_ctx);
  expect_op_work_queue(mock_image_ctx);

  InSequence seq;
  expect_get_image_size(mock_image_ctx, mock_image_ctx.snaps.front(), 123);
  expect_is_snap_protected(mock_image_ctx, true, 0);

  MockCreateRequest mock_create_request;
  expect_create(mock_create_request, 0);

  expect_open(mock_image_ctx, 0);
  expect_set_parent(mock_image_ctx, 0);
  expect_add_child(m_ioctx, 0);

  MockRefreshRequest mock_refresh_request;
  expect_refresh(mock_refresh_request, 0);
  expect_is_snap_protected(mock_image_ctx, true, 0);

  expect_metadata_list(mock_image_ctx, {}, 0);

  expect_test_features(mock_image_ctx, RBD_FEATURE_JOURNALING, true);
  expect_mirror_mode_get(mock_image_ctx, cls::rbd::MIRROR_MODE_POOL, -EINVAL);

  expect_close(mock_image_ctx, 0);

  MockRemoveRequest mock_remove_request;
  expect_remove(mock_remove_request, 0);

  C_SaferCond ctx;
  ImageOptions clone_opts;
  auto req = new MockCloneRequest(&mock_image_ctx, m_ioctx, "clone name",
                                  "clone id", clone_opts, "", "",
                                  image_ctx->op_work_queue, &ctx);
  req->send();
  ASSERT_EQ(-EINVAL, ctx.wait());
}

TEST_F(TestMockImageCloneRequest, MirrorEnableError) {
  REQUIRE_FEATURE(RBD_FEATURE_LAYERING | RBD_FEATURE_JOURNALING);

  MockTestImageCtx mock_image_ctx(*image_ctx);
  expect_op_work_queue(mock_image_ctx);

  InSequence seq;
  expect_get_image_size(mock_image_ctx, mock_image_ctx.snaps.front(), 123);
  expect_is_snap_protected(mock_image_ctx, true, 0);

  MockCreateRequest mock_create_request;
  expect_create(mock_create_request, 0);

  expect_open(mock_image_ctx, 0);
  expect_set_parent(mock_image_ctx, 0);

  expect_op_features_set(m_ioctx, mock_image_ctx.id, 0);
  expect_child_attach(mock_image_ctx, 0);

  expect_metadata_list(mock_image_ctx, {}, 0);

  expect_test_features(mock_image_ctx, RBD_FEATURE_JOURNALING, true);
  expect_mirror_mode_get(mock_image_ctx, cls::rbd::MIRROR_MODE_POOL, 0);

  MockMirrorEnableRequest mock_mirror_enable_request;
  expect_mirror_enable(mock_mirror_enable_request, -EINVAL);

  expect_close(mock_image_ctx, 0);

  MockRemoveRequest mock_remove_request;
  expect_remove(mock_remove_request, 0);

  C_SaferCond ctx;
  ImageOptions clone_opts;
  auto req = new MockCloneRequest(&mock_image_ctx, m_ioctx, "clone name",
                                  "clone id", clone_opts, "", "",
                                  image_ctx->op_work_queue, &ctx);
  req->send();
  ASSERT_EQ(-EINVAL, ctx.wait());
}

TEST_F(TestMockImageCloneRequest, CloseError) {
  REQUIRE_FEATURE(RBD_FEATURE_LAYERING);

  MockTestImageCtx mock_image_ctx(*image_ctx);
  expect_op_work_queue(mock_image_ctx);

  InSequence seq;
  expect_get_image_size(mock_image_ctx, mock_image_ctx.snaps.front(), 123);
  expect_is_snap_protected(mock_image_ctx, true, 0);

  MockCreateRequest mock_create_request;
  expect_create(mock_create_request, 0);

  expect_open(mock_image_ctx, 0);
  expect_set_parent(mock_image_ctx, 0);

  expect_op_features_set(m_ioctx, mock_image_ctx.id, 0);
  expect_child_attach(mock_image_ctx, 0);

  expect_metadata_list(mock_image_ctx, {}, 0);
  expect_test_features(mock_image_ctx, RBD_FEATURE_JOURNALING, false);

  expect_close(mock_image_ctx, -EINVAL);

  C_SaferCond ctx;
  ImageOptions clone_opts;
  auto req = new MockCloneRequest(&mock_image_ctx, m_ioctx, "clone name",
                                  "clone id", clone_opts, "", "",
                                  image_ctx->op_work_queue, &ctx);
  req->send();
  ASSERT_EQ(-EINVAL, ctx.wait());
}

TEST_F(TestMockImageCloneRequest, RemoveError) {
  REQUIRE_FEATURE(RBD_FEATURE_LAYERING);

  MockTestImageCtx mock_image_ctx(*image_ctx);
  expect_op_work_queue(mock_image_ctx);

  InSequence seq;
  expect_get_image_size(mock_image_ctx, mock_image_ctx.snaps.front(), 123);
  expect_is_snap_protected(mock_image_ctx, true, 0);

  MockCreateRequest mock_create_request;
  expect_create(mock_create_request, 0);

  expect_open(mock_image_ctx, -EINVAL);

  MockRemoveRequest mock_remove_request;
  expect_remove(mock_remove_request, -EPERM);

  C_SaferCond ctx;
  ImageOptions clone_opts;
  auto req = new MockCloneRequest(&mock_image_ctx, m_ioctx, "clone name",
                                  "clone id", clone_opts, "", "",
                                  image_ctx->op_work_queue, &ctx);
  req->send();
  ASSERT_EQ(-EINVAL, ctx.wait());
}

} // namespace image
} // namespace librbd
