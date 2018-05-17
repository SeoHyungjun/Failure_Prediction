// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*- 
// vim: ts=8 sw=2 smarttab
/*
 * Ceph - scalable distributed file system
 *
 * Copyright (C) 2004-2006 Sage Weil <sage@newdream.net>
 *
 * This is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License version 2.1, as published by the Free Software 
 * Foundation.  See file COPYING.
 * 
 */


#ifndef CEPH_MOSDPGLOG_H
#define CEPH_MOSDPGLOG_H

#include "messages/MOSDPeeringOp.h"

class MOSDPGLog : public MOSDPeeringOp {

  static const int HEAD_VERSION = 5;
  static const int COMPAT_VERSION = 5;

  epoch_t epoch = 0;
  /// query_epoch is the epoch of the query being responded to, or
  /// the current epoch if this is not being sent in response to a
  /// query. This allows the recipient to disregard responses to old
  /// queries.
  epoch_t query_epoch = 0;

public:
  shard_id_t to;
  shard_id_t from;
  pg_info_t info;
  pg_log_t log;
  pg_missing_t missing;
  PastIntervals past_intervals;

  epoch_t get_epoch() const { return epoch; }
  spg_t get_pgid() const { return spg_t(info.pgid.pgid, to); }
  epoch_t get_query_epoch() const { return query_epoch; }

  spg_t get_spg() const override {
    return spg_t(info.pgid.pgid, to);
  }
  epoch_t get_map_epoch() const override {
    return epoch;
  }
  epoch_t get_min_epoch() const override {
    return query_epoch;
  }

  PGPeeringEvent *get_event() override {
    return new PGPeeringEvent(
      epoch, query_epoch,
      MLogRec(pg_shard_t(get_source().num(), from),
	      this),
      true,
      new PGCreateInfo(
	get_spg(),
	epoch,
	info.history,
	past_intervals,
	false));
  }

  MOSDPGLog() : MOSDPeeringOp(MSG_OSD_PG_LOG, HEAD_VERSION, COMPAT_VERSION) {
    set_priority(CEPH_MSG_PRIO_HIGH); 
  }
  MOSDPGLog(shard_id_t to, shard_id_t from, version_t mv, pg_info_t& i)
    : MOSDPeeringOp(MSG_OSD_PG_LOG, HEAD_VERSION, COMPAT_VERSION),
      epoch(mv), query_epoch(mv),
      to(to), from(from),
      info(i)  {
    set_priority(CEPH_MSG_PRIO_HIGH); 
  }
  MOSDPGLog(shard_id_t to, shard_id_t from,
	    version_t mv, pg_info_t& i, epoch_t query_epoch)
    : MOSDPeeringOp(MSG_OSD_PG_LOG, HEAD_VERSION, COMPAT_VERSION),
      epoch(mv), query_epoch(query_epoch),
      to(to), from(from),
      info(i)  {
    set_priority(CEPH_MSG_PRIO_HIGH);
  }

private:
  ~MOSDPGLog() override {}

public:
  const char *get_type_name() const override { return "PGlog"; }
  void inner_print(ostream& out) const override {
    // NOTE: log is not const, but operator<< doesn't touch fields
    // swapped out by OSD code.
    out << "log " << log
	<< " pi " << past_intervals;
  }

  void encode_payload(uint64_t features) override {
    using ceph::encode;
    encode(epoch, payload);
    encode(info, payload);
    encode(log, payload);
    encode(missing, payload);
    encode(query_epoch, payload);
    encode(past_intervals, payload);
    encode(to, payload);
    encode(from, payload);
  }
  void decode_payload() override {
    bufferlist::iterator p = payload.begin();
    decode(epoch, p);
    decode(info, p);
    log.decode(p, info.pgid.pool());
    missing.decode(p, info.pgid.pool());
    decode(query_epoch, p);
    decode(past_intervals, p);
    decode(to, p);
    decode(from, p);
  }
};

#endif
