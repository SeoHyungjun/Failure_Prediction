// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-
// vim: ts=8 sw=2 smarttab
/*
 * Ceph - scalable distributed file system
 *
 * Copyright (C) 2017 John Spray <john.spray@redhat.com>
 *
 * This is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License version 2.1, as published by the Free Software
 * Foundation.  See file COPYING.
 */


#pragma once

// First because it includes Python.h
#include "PyModule.h"

#include <string>
#include <map>
#include <memory>

#include "common/LogClient.h"

#include "ActivePyModules.h"
#include "StandbyPyModules.h"

/**
 * This class is responsible for setting up the python runtime environment
 * and importing the python modules.
 *
 * It is *not* responsible for constructing instances of their BaseMgrModule
 * subclasses: that is the job of ActiveMgrModule, which consumes the class
 * references that we load here.
 */
class PyModuleRegistry
{
private:
  mutable Mutex lock{"PyModuleRegistry::lock"};
  LogChannelRef clog;

  std::map<std::string, PyModuleRef> modules;

  std::unique_ptr<ActivePyModules> active_modules;
  std::unique_ptr<StandbyPyModules> standby_modules;

  PyThreadState *pMainThreadState;

  // We have our own copy of MgrMap, because we are constructed
  // before ClusterState exists.
  MgrMap mgr_map;

  /**
   * Discover python modules from local disk
   */
  std::set<std::string> probe_modules() const;

  PyModuleConfig module_config;

public:
  void handle_config(const std::string &k, const std::string &v);

  /**
   * Get references to all modules (whether they have loaded and/or
   * errored) or not.
   */
  std::list<PyModuleRef> get_modules() const
  {
    Mutex::Locker l(lock);
    std::list<PyModuleRef> modules_out;
    for (const auto &i : modules) {
      modules_out.push_back(i.second);
    }

    return modules_out;
  }

  explicit PyModuleRegistry(LogChannelRef clog_)
    : clog(clog_)
  {}

  /**
   * @return true if the mgrmap has changed such that the service needs restart
   */
  bool handle_mgr_map(const MgrMap &mgr_map_);

  void init();

  void upgrade_config(
      MonClient *monc,
      const std::map<std::string, std::string> &old_config);

  void active_start(
                DaemonStateIndex &ds, ClusterState &cs,
                const std::map<std::string, std::string> &kv_store,
                MonClient &mc, LogChannelRef clog_, Objecter &objecter_,
                Client &client_, Finisher &f);
  void standby_start(MonClient &mc);

  bool is_standby_running() const
  {
    return standby_modules != nullptr;
  }

  void active_shutdown();
  void shutdown();

  template<typename Callback, typename...Args>
  void with_active_modules(Callback&& cb, Args&&...args) const
  {
    Mutex::Locker l(lock);
    assert(active_modules != nullptr);

    std::forward<Callback>(cb)(*active_modules, std::forward<Args>(args)...);
  }

  std::vector<MonCommand> get_commands() const;
  std::vector<ModuleCommand> get_py_commands() const;

  /**
   * module_name **must** exist, but does not have to be loaded
   * or runnable.
   */
  PyModuleRef get_module(const std::string &module_name)
  {
    Mutex::Locker l(lock);
    return modules.at(module_name);
  }

  /**
   * Pass through command to the named module for execution.
   *
   * The command must exist in the COMMANDS reported by the module.  If it
   * doesn't then this will abort.
   *
   * If ActivePyModules has not been instantiated yet then this will
   * return EAGAIN.
   */
  int handle_command(
    std::string const &module_name,
    const cmdmap_t &cmdmap,
    std::stringstream *ds,
    std::stringstream *ss);

  /**
   * Pass through health checks reported by modules, and report any
   * modules that have failed (i.e. unhandled exceptions in serve())
   */
  void get_health_checks(health_check_map_t *checks);

  // FIXME: breaking interface so that I don't have to go rewrite all
  // the places that call into these (for now)
  // >>>
  void notify_all(const std::string &notify_type,
                  const std::string &notify_id)
  {
    if (active_modules) {
      active_modules->notify_all(notify_type, notify_id);
    }
  }

  void notify_all(const LogEntry &log_entry)
  {
    if (active_modules) {
      active_modules->notify_all(log_entry);
    }
  }

  std::map<std::string, std::string> get_services() const
  {
    assert(active_modules);
    return active_modules->get_services();
  }
  // <<< (end of ActivePyModules cheeky call-throughs)
};
