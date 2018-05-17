// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-
// vim: ts=8 sw=2 smarttab

#ifndef CEPH_LIBRBD_JOURNAL_OBJECT_DISPATCH_H
#define CEPH_LIBRBD_JOURNAL_OBJECT_DISPATCH_H

#include "include/int_types.h"
#include "include/buffer.h"
#include "include/rados/librados.hpp"
#include "common/snap_types.h"
#include "common/zipkin_trace.h"
#include "librbd/io/Types.h"
#include "librbd/io/ObjectDispatchInterface.h"

struct Context;

namespace librbd {

struct ImageCtx;
template <typename> class Journal;

namespace journal {

template <typename ImageCtxT = librbd::ImageCtx>
class ObjectDispatch : public io::ObjectDispatchInterface {
public:
  static ObjectDispatch* create(ImageCtxT* image_ctx,
                                Journal<ImageCtxT>* journal) {
    return new ObjectDispatch(image_ctx, journal);
  }

  ObjectDispatch(ImageCtxT* image_ctx, Journal<ImageCtxT>* journal);

  io::ObjectDispatchLayer get_object_dispatch_layer() const override {
    return io::OBJECT_DISPATCH_LAYER_JOURNAL;
  }

  void shut_down(Context* on_finish) override;

  bool read(
      const std::string &oid, uint64_t object_no, uint64_t object_off,
      uint64_t object_len, librados::snap_t snap_id, int op_flags,
      const ZTracer::Trace &parent_trace, ceph::bufferlist* read_data,
      io::ExtentMap* extent_map, int* object_dispatch_flags,
      io::DispatchResult* dispatch_result, Context** on_finish,
      Context* on_dispatched) {
    return false;
  }

  bool discard(
      const std::string &oid, uint64_t object_no, uint64_t object_off,
      uint64_t object_len, const ::SnapContext &snapc, int discard_flags,
      const ZTracer::Trace &parent_trace, int* object_dispatch_flags,
      uint64_t* journal_tid, io::DispatchResult* dispatch_result,
      Context** on_finish, Context* on_dispatched) override;

  bool write(
      const std::string &oid, uint64_t object_no, uint64_t object_off,
      ceph::bufferlist&& data, const ::SnapContext &snapc, int op_flags,
      const ZTracer::Trace &parent_trace, int* object_dispatch_flags,
      uint64_t* journal_tid, io::DispatchResult* dispatch_result,
      Context** on_finish, Context* on_dispatched) override;

  bool write_same(
      const std::string &oid, uint64_t object_no, uint64_t object_off,
      uint64_t object_len, io::Extents&& buffer_extents,
      ceph::bufferlist&& data, const ::SnapContext &snapc, int op_flags,
      const ZTracer::Trace &parent_trace, int* object_dispatch_flags,
      uint64_t* journal_tid, io::DispatchResult* dispatch_result,
      Context** on_finish, Context* on_dispatched) override;

  bool compare_and_write(
      const std::string &oid, uint64_t object_no, uint64_t object_off,
      ceph::bufferlist&& cmp_data, ceph::bufferlist&& write_data,
      const ::SnapContext &snapc, int op_flags,
      const ZTracer::Trace &parent_trace, uint64_t* mismatch_offset,
      int* object_dispatch_flags, uint64_t* journal_tid,
      io::DispatchResult* dispatch_result, Context** on_finish,
      Context* on_dispatched) override;

  bool flush(
      io::FlushSource flush_source, const ZTracer::Trace &parent_trace,
      io::DispatchResult* dispatch_result, Context** on_finish,
      Context* on_dispatched) override {
    return false;
  }

  bool invalidate_cache(Context* on_finish) override {
    return false;
  }
  bool reset_existence_cache(Context* on_finish) override {
    return false;
  }

  void extent_overwritten(
      uint64_t object_no, uint64_t object_off, uint64_t object_len,
      uint64_t journal_tid, uint64_t new_journal_tid) override;

private:
  ImageCtxT* m_image_ctx;
  Journal<ImageCtxT>* m_journal;

  void wait_or_flush_event(uint64_t journal_tid, int object_dispatch_flags,
                           Context* on_dispatched);

};

} // namespace journal
} // namespace librbd

extern template class librbd::journal::ObjectDispatch<librbd::ImageCtx>;

#endif // CEPH_LIBRBD_JOURNAL_OBJECT_DISPATCH_H
