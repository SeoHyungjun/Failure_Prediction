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

#ifndef CEPH_MMGRBEACON_H
#define CEPH_MMGRBEACON_H

#include "messages/PaxosServiceMessage.h"
#include "mon/MonCommand.h"
#include "mon/MgrMap.h"

#include "include/types.h"



class MMgrBeacon : public PaxosServiceMessage {

  static const int HEAD_VERSION = 7;
  static const int COMPAT_VERSION = 1;

protected:
  uint64_t gid;
  entity_addr_t server_addr;
  bool available;
  std::string name;
  uuid_d fsid;

  // From active daemon to populate MgrMap::services
  std::map<std::string, std::string> services;

  // Only populated during activation
  std::vector<MonCommand> command_descs;

  // Information about the modules found locally on this daemon
  std::vector<MgrMap::ModuleInfo> modules;

  map<string,string> metadata; ///< misc metadata about this osd

public:
  MMgrBeacon()
    : PaxosServiceMessage(MSG_MGR_BEACON, 0, HEAD_VERSION, COMPAT_VERSION),
      gid(0), available(false)
  {
  }

  MMgrBeacon(const uuid_d& fsid_, uint64_t gid_, const std::string &name_,
             entity_addr_t server_addr_, bool available_,
	     std::vector<MgrMap::ModuleInfo>&& modules_,
	     map<string,string>&& metadata_)
    : PaxosServiceMessage(MSG_MGR_BEACON, 0, HEAD_VERSION, COMPAT_VERSION),
      gid(gid_), server_addr(server_addr_), available(available_), name(name_),
      fsid(fsid_), modules(std::move(modules_)), metadata(std::move(metadata_))
  {
  }

  uint64_t get_gid() const { return gid; }
  entity_addr_t get_server_addr() const { return server_addr; }
  bool get_available() const { return available; }
  const std::string& get_name() const { return name; }
  const uuid_d& get_fsid() const { return fsid; }
  std::vector<MgrMap::ModuleInfo>& get_modules() { return modules; }
  const std::map<std::string,std::string>& get_metadata() const {
    return metadata;
  }

  const std::map<std::string,std::string>& get_services() const {
    return services;
  }

  void set_services(const std::map<std::string, std::string> &svcs)
  {
    services = svcs;
  }

  void set_command_descs(const std::vector<MonCommand> &cmds)
  {
    command_descs = cmds;
  }

  const std::vector<MonCommand> &get_command_descs()
  {
    return command_descs;
  }

  const std::vector<MgrMap::ModuleInfo> &get_available_modules() const
  {
    return modules;
  }

private:
  ~MMgrBeacon() override {}

public:

  const char *get_type_name() const override { return "mgrbeacon"; }

  void print(ostream& out) const override {
    out << get_type_name() << " mgr." << name << "(" << fsid << ","
	<< gid << ", " << server_addr << ", " << available
	<< ")";
  }

  void encode_payload(uint64_t features) override {
    using ceph::encode;
    paxos_encode();

    encode(server_addr, payload, features);
    encode(gid, payload);
    encode(available, payload);
    encode(name, payload);
    encode(fsid, payload);

    // Fill out old-style list of module names (deprecated by
    // later field of full ModuleInfos)
    std::set<std::string> module_names;
    for (const auto &info : modules) {
      module_names.insert(info.name);
    }
    encode(module_names, payload);

    encode(command_descs, payload);
    encode(metadata, payload);
    encode(services, payload);

    encode(modules, payload);
  }
  void decode_payload() override {
    bufferlist::iterator p = payload.begin();
    paxos_decode(p);
    decode(server_addr, p);
    decode(gid, p);
    decode(available, p);
    decode(name, p);
    if (header.version >= 2) {
      decode(fsid, p);
    }
    if (header.version >= 3) {
      std::set<std::string> module_name_list;
      decode(module_name_list, p);
      // Only need to unpack this field if we won't have the full
      // ModuleInfo structures added in v7
      if (header.version < 7) {
        for (const auto &i : module_name_list) {
          MgrMap::ModuleInfo info;
          info.name = i;
          modules.push_back(std::move(info));
        }
      }
    }
    if (header.version >= 4) {
      decode(command_descs, p);
    }
    if (header.version >= 5) {
      decode(metadata, p);
    }
    if (header.version >= 6) {
      decode(services, p);
    }
    if (header.version >= 7) {
      decode(modules, p);
    }
  }
};


#endif
