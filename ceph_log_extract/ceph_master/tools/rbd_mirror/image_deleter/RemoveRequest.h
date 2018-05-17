// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-
// vim: ts=8 sw=2 smarttab

#ifndef CEPH_RBD_MIRROR_IMAGE_DELETER_REMOVE_REQUEST_H
#define CEPH_RBD_MIRROR_IMAGE_DELETER_REMOVE_REQUEST_H

#include "include/rados/librados.hpp"
#include "include/buffer.h"
#include "librbd/internal.h"
#include "tools/rbd_mirror/image_deleter/Types.h"
#include <string>
#include <vector>

class Context;
class ContextWQ;
namespace librbd { struct ImageCtx; }

namespace rbd {
namespace mirror {
namespace image_deleter {

template <typename ImageCtxT = librbd::ImageCtx>
class RemoveRequest {
public:
  static RemoveRequest* create(librados::IoCtx &io_ctx,
                               const std::string &image_id,
                               ErrorResult *error_result,
                               ContextWQ *op_work_queue, Context *on_finish) {
    return new RemoveRequest(io_ctx, image_id, error_result, op_work_queue,
                             on_finish);
  }

  RemoveRequest(librados::IoCtx &io_ctx, const std::string &image_id,
                ErrorResult *error_result, ContextWQ *op_work_queue,
                Context *on_finish)
    : m_io_ctx(io_ctx), m_image_id(image_id), m_error_result(error_result),
      m_op_work_queue(op_work_queue), m_on_finish(on_finish) {
  }

  void send();

private:
  /*
   * @verbatim
   *
   * <start>
   *    |
   *    v
   * GET_SNAP_CONTEXT
   *    |
   *    v
   * PURGE_SNAPSHOTS
   *    |
   *    v
   * REMOVE_IMAGE
   *    |
   *    v
   * <finish>
   *
   * @endverbatim
   */

  librados::IoCtx &m_io_ctx;
  std::string m_image_id;
  ErrorResult *m_error_result;
  ContextWQ *m_op_work_queue;
  Context *m_on_finish;

  ceph::bufferlist m_out_bl;
  bool m_has_snapshots = false;
  librbd::NoOpProgressContext m_progress_ctx;

  void get_snap_context();
  void handle_get_snap_context(int r);

  void purge_snapshots();
  void handle_purge_snapshots(int r);

  void remove_image();
  void handle_remove_image(int r);

  void finish(int r);

};

} // namespace image_deleter
} // namespace mirror
} // namespace rbd

extern template class rbd::mirror::image_deleter::RemoveRequest<librbd::ImageCtx>;

#endif // CEPH_RBD_MIRROR_IMAGE_DELETER_REMOVE_REQUEST_H
