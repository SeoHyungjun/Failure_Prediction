// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-
// vim: ts=8 sw=2 smarttab
/*
* Ceph - scalable distributed file system
*
* Copyright (C) 2012 Inktank, Inc.
*
* This is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License version 2.1, as published by the Free Software
* Foundation. See file COPYING.
*/
#include <map>
#include <set>
#include <string>
#include <fstream>

#include <boost/scoped_ptr.hpp>

#include "common/ceph_argparse.h"
#include "common/config.h"
#include "common/errno.h"
#include "common/strtol.h"
#include "global/global_context.h"
#include "global/global_init.h"
#include "include/stringify.h"
#include "common/Clock.h"
#include "kv/KeyValueDB.h"
#include "common/url_escape.h"

#ifdef WITH_BLUESTORE
#include "os/bluestore/BlueStore.h"
#endif


class StoreTool
{
#ifdef WITH_BLUESTORE
  struct Deleter {
    BlueStore *bluestore;
    Deleter()
      : bluestore(nullptr) {}
    Deleter(BlueStore *store)
      : bluestore(store) {}
    void operator()(KeyValueDB *db) {
      if (bluestore) {
	bluestore->umount();
	delete bluestore;
      } else {
	delete db;
      }
    }
  };
  std::unique_ptr<KeyValueDB, Deleter> db;
#else
  std::unique_ptr<KeyValueDB> db;
#endif

  string store_path;

  public:
  StoreTool(string type, const string &path, bool need_open_db=true) : store_path(path) {
    if (type == "bluestore-kv") {
#ifdef WITH_BLUESTORE
      auto bluestore = new BlueStore(g_ceph_context, path, need_open_db);
      KeyValueDB *db_ptr;
      int r = bluestore->start_kv_only(&db_ptr);
      if (r < 0) {
	exit(1);
      }
      db = decltype(db){db_ptr, Deleter(bluestore)};
#else
      cerr << "bluestore not compiled in" << std::endl;
      exit(1);
#endif
    } else {
      auto db_ptr = KeyValueDB::create(g_ceph_context, type, path);
      if (need_open_db) {
        int r = db_ptr->open(std::cerr);
        if (r < 0) {
          cerr << "failed to open type " << type << " path " << path << ": "
               << cpp_strerror(r) << std::endl;
          exit(1);
        }
        db.reset(db_ptr);
      }
    }
  }

  uint32_t traverse(const string &prefix,
                    const bool do_crc,
                    ostream *out) {
    KeyValueDB::WholeSpaceIterator iter = db->get_wholespace_iterator();

    if (prefix.empty())
      iter->seek_to_first();
    else
      iter->seek_to_first(prefix);

    uint32_t crc = -1;

    while (iter->valid()) {
      pair<string,string> rk = iter->raw_key();
      if (!prefix.empty() && (rk.first != prefix))
        break;

      if (out)
        *out << url_escape(rk.first) << "\t" << url_escape(rk.second);
      if (do_crc) {
        bufferlist bl;
        bl.append(rk.first);
        bl.append(rk.second);
        bl.append(iter->value());

        crc = bl.crc32c(crc);
        if (out) {
          *out << "\t" << bl.crc32c(0);
        }
      }
      if (out)
        *out << std::endl;
      iter->next();
    }

    return crc;
  }

  void list(const string &prefix, const bool do_crc) {
    traverse(prefix, do_crc, &std::cout);
  }

  bool exists(const string &prefix) {
    assert(!prefix.empty());
    KeyValueDB::WholeSpaceIterator iter = db->get_wholespace_iterator();
    iter->seek_to_first(prefix);
    return (iter->valid() && (iter->raw_key().first == prefix));
  }

  bool exists(const string &prefix, const string &key) {
    assert(!prefix.empty());

    if (key.empty()) {
      return exists(prefix);
    }

    bool exists = false;
    get(prefix, key, exists);
    return exists;
  }

  bufferlist get(const string &prefix, const string &key, bool &exists) {
    assert(!prefix.empty() && !key.empty());

    map<string,bufferlist> result;
    std::set<std::string> keys;
    keys.insert(key);
    db->get(prefix, keys, &result);

    if (result.count(key) > 0) {
      exists = true;
      return result[key];
    }
    exists = false;
    return bufferlist();
  }

  uint64_t get_size() {
    map<string,uint64_t> extras;
    uint64_t s = db->get_estimated_size(extras);
    for (map<string,uint64_t>::iterator p = extras.begin();
         p != extras.end(); ++p) {
      std::cout << p->first << " - " << p->second << std::endl;
    }
    std::cout << "total: " << s << std::endl;
    return s;
  }

  bool set(const string &prefix, const string &key, bufferlist &val) {
    assert(!prefix.empty());
    assert(!key.empty());
    assert(val.length() > 0);

    KeyValueDB::Transaction tx = db->get_transaction();
    tx->set(prefix, key, val);
    int ret = db->submit_transaction_sync(tx);

    return (ret == 0);
  }

  bool rm(const string& prefix, const string& key) {
    assert(!prefix.empty());
    assert(!key.empty());

    KeyValueDB::Transaction tx = db->get_transaction();
    tx->rmkey(prefix, key);
    int ret = db->submit_transaction_sync(tx);

    return (ret == 0);
  }

  bool rm_prefix(const string& prefix) {
    assert(!prefix.empty());

    KeyValueDB::Transaction tx = db->get_transaction();
    tx->rmkeys_by_prefix(prefix);
    int ret = db->submit_transaction_sync(tx);

    return (ret == 0);
  }

  int copy_store_to(string type, const string &other_path,
		    const int num_keys_per_tx, const string &other_type) {

    if (num_keys_per_tx <= 0) {
      std::cerr << "must specify a number of keys/tx > 0" << std::endl;
      return -EINVAL;
    }

    // open or create a leveldb store at @p other_path
    boost::scoped_ptr<KeyValueDB> other;
    KeyValueDB *other_ptr = KeyValueDB::create(g_ceph_context, other_type, other_path);
    int err = other_ptr->create_and_open(std::cerr);
    if (err < 0)
      return err;
    other.reset(other_ptr);

    KeyValueDB::WholeSpaceIterator it = db->get_wholespace_iterator();
    it->seek_to_first();
    uint64_t total_keys = 0;
    uint64_t total_size = 0;
    uint64_t total_txs = 0;

    auto started_at = coarse_mono_clock::now();

    do {
      int num_keys = 0;

      KeyValueDB::Transaction tx = other->get_transaction();


      while (it->valid() && num_keys < num_keys_per_tx) {
        pair<string,string> k = it->raw_key();
        bufferlist v = it->value();
        tx->set(k.first, k.second, v);

        num_keys ++;
        total_size += v.length();

        it->next();
      }

      total_txs ++;
      total_keys += num_keys;

      if (num_keys > 0)
        other->submit_transaction_sync(tx);

      auto cur_duration = std::chrono::duration<double>(coarse_mono_clock::now() - started_at);
      std::cout << "ts = " << cur_duration.count() << "s, copied " << total_keys
                << " keys so far (" << stringify(byte_u_t(total_size)) << ")"
                << std::endl;

    } while (it->valid());

    auto time_taken = std::chrono::duration<double>(coarse_mono_clock::now() - started_at);

    std::cout << "summary:" << std::endl;
    std::cout << "  copied " << total_keys << " keys" << std::endl;
    std::cout << "  used " << total_txs << " transactions" << std::endl;
    std::cout << "  total size " << stringify(byte_u_t(total_size)) << std::endl;
    std::cout << "  from '" << store_path << "' to '" << other_path << "'"
              << std::endl;
    std::cout << "  duration " << time_taken.count() << " seconds" << std::endl;

    return 0;
  }

  void compact() {
    db->compact();
  }
  void compact_prefix(string prefix) {
    db->compact_prefix(prefix);
  }
  void compact_range(string prefix, string start, string end) {
    db->compact_range(prefix, start, end);
  }

  int repair() {
    return db->repair(std::cout);
  }
};

void usage(const char *pname)
{
  std::cout << "Usage: " << pname << " <leveldb|rocksdb|bluestore-kv> <store path> command [args...]\n"
    << "\n"
    << "Commands:\n"
    << "  list [prefix]\n"
    << "  list-crc [prefix]\n"
    << "  exists <prefix> [key]\n"
    << "  get <prefix> <key> [out <file>]\n"
    << "  crc <prefix> <key>\n"
    << "  get-size [<prefix> <key>]\n"
    << "  set <prefix> <key> [ver <N>|in <file>]\n"
    << "  rm <prefix> <key>\n"
    << "  rm-prefix <prefix>\n"
    << "  store-copy <path> [num-keys-per-tx] [leveldb|rocksdb|...] \n"
    << "  store-crc <path>\n"
    << "  compact\n"
    << "  compact-prefix <prefix>\n"
    << "  compact-range <prefix> <start> <end>\n"
    << "  repair\n"
    << std::endl;
}

int main(int argc, const char *argv[])
{
  vector<const char*> args;
  argv_to_vec(argc, argv, args);
  if (args.empty()) {
    cerr << argv[0] << ": -h or --help for usage" << std::endl;
    exit(1);
  }
  if (ceph_argparse_need_usage(args)) {
    usage(argv[0]);
    exit(0);
  }

  map<string,string> defaults = {
    { "debug_rocksdb", "2" }
  };

  auto cct = global_init(
    &defaults, args,
    CEPH_ENTITY_TYPE_CLIENT, CODE_ENVIRONMENT_UTILITY,
    CINIT_FLAG_NO_DEFAULT_CONFIG_FILE);
  common_init_finish(g_ceph_context);


  if (args.size() < 3) {
    usage(argv[0]);
    return 1;
  }

  string type(args[0]);
  string path(args[1]);
  string cmd(args[2]);

  if (type != "leveldb" &&
      type != "rocksdb" &&
      type != "bluestore-kv")  {

    std::cerr << "Unrecognized type: " << args[0] << std::endl;
    usage(argv[0]);
    return 1;
  }

  bool need_open_db = (cmd != "repair");
  StoreTool st(type, path, need_open_db);

  if (cmd == "repair") {
    int ret = st.repair();
    if (!ret) {
      std::cout << "repair kvstore successfully" << std::endl;
    } else {
      std::cout << "repair kvstore failed" << std::endl;
    }
    return ret;
  } else if (cmd == "list" || cmd == "list-crc") {
    string prefix;
    if (argc > 4)
      prefix = url_unescape(argv[4]);

    bool do_crc = (cmd == "list-crc");

    st.list(prefix, do_crc);

  } else if (cmd == "exists") {
    string key;
    if (argc < 5) {
      usage(argv[0]);
      return 1;
    }
    string prefix(url_unescape(argv[4]));
    if (argc > 5)
      key = url_unescape(argv[5]);

    bool ret = st.exists(prefix, key);
    std::cout << "(" << url_escape(prefix) << ", " << url_escape(key) << ") "
      << (ret ? "exists" : "does not exist")
      << std::endl;
    return (ret ? 0 : 1);

  } else if (cmd == "get") {
    if (argc < 6) {
      usage(argv[0]);
      return 1;
    }
    string prefix(url_unescape(argv[4]));
    string key(url_unescape(argv[5]));

    bool exists = false;
    bufferlist bl = st.get(prefix, key, exists);
    std::cout << "(" << url_escape(prefix) << ", " << url_escape(key) << ")";
    if (!exists) {
      std::cout << " does not exist" << std::endl;
      return 1;
    }
    std::cout << std::endl;

    if (argc >= 7) {
      string subcmd(argv[6]);
      if (subcmd != "out") {
        std::cerr << "unrecognized subcmd '" << subcmd << "'"
                  << std::endl;
        return 1;
      }
      if (argc < 8) {
        std::cerr << "output path not specified" << std::endl;
        return 1;
      }
      string out(argv[7]);

      if (out.empty()) {
        std::cerr << "unspecified out file" << std::endl;
        return 1;
      }

      int err = bl.write_file(argv[7], 0644);
      if (err < 0) {
        std::cerr << "error writing value to '" << out << "': "
                  << cpp_strerror(err) << std::endl;
        return 1;
      }
    } else {
      ostringstream os;
      bl.hexdump(os);
      std::cout << os.str() << std::endl;
    }

  } else if (cmd == "crc") {
    if (argc < 6) {
      usage(argv[0]);
      return 1;
    }
    string prefix(url_unescape(argv[4]));
    string key(url_unescape(argv[5]));

    bool exists = false;
    bufferlist bl = st.get(prefix, key, exists);
    std::cout << "(" << url_escape(prefix) << ", " << url_escape(key) << ") ";
    if (!exists) {
      std::cout << " does not exist" << std::endl;
      return 1;
    }
    std::cout << " crc " << bl.crc32c(0) << std::endl;

  } else if (cmd == "get-size") {
    std::cout << "estimated store size: " << st.get_size() << std::endl;

    if (argc < 5)
      return 0;

    if (argc < 6) {
      usage(argv[0]);
      return 1;
    }
    string prefix(url_unescape(argv[4]));
    string key(url_unescape(argv[5]));

    bool exists = false;
    bufferlist bl = st.get(prefix, key, exists);
    if (!exists) {
      std::cerr << "(" << url_escape(prefix) << "," << url_escape(key)
                << ") does not exist" << std::endl;
      return 1;
    }
    std::cout << "(" << url_escape(prefix) << "," << url_escape(key)
              << ") size " << byte_u_t(bl.length()) << std::endl;

  } else if (cmd == "set") {
    if (argc < 8) {
      usage(argv[0]);
      return 1;
    }
    string prefix(url_unescape(argv[4]));
    string key(url_unescape(argv[5]));
    string subcmd(argv[6]);

    bufferlist val;
    string errstr;
    if (subcmd == "ver") {
      version_t v = (version_t) strict_strtoll(argv[7], 10, &errstr);
      if (!errstr.empty()) {
        std::cerr << "error reading version: " << errstr << std::endl;
        return 1;
      }
      encode(v, val);
    } else if (subcmd == "in") {
      int ret = val.read_file(argv[7], &errstr);
      if (ret < 0 || !errstr.empty()) {
        std::cerr << "error reading file: " << errstr << std::endl;
        return 1;
      }
    } else {
      std::cerr << "unrecognized subcommand '" << subcmd << "'" << std::endl;
      usage(argv[0]);
      return 1;
    }

    bool ret = st.set(prefix, key, val);
    if (!ret) {
      std::cerr << "error setting ("
                << url_escape(prefix) << "," << url_escape(key) << ")" << std::endl;
      return 1;
    }
  } else if (cmd == "rm") {
    if (argc < 6) {
      usage(argv[0]);
      return 1;
    }
    string prefix(url_unescape(argv[4]));
    string key(url_unescape(argv[5]));

    bool ret = st.rm(prefix, key);
    if (!ret) {
      std::cerr << "error removing ("
                << url_escape(prefix) << "," << url_escape(key) << ")"
		<< std::endl;
      return 1;
    }
  } else if (cmd == "rm-prefix") {
    if (argc < 5) {
      usage(argv[0]);
      return 1;
    }
    string prefix(url_unescape(argv[4]));

    bool ret = st.rm_prefix(prefix);
    if (!ret) {
      std::cerr << "error removing prefix ("
                << url_escape(prefix) << ")"
		<< std::endl;
      return 1;
    }
  } else if (cmd == "store-copy") {
    int num_keys_per_tx = 128; // magic number that just feels right.
    if (argc < 5) {
      usage(argv[0]);
      return 1;
    } else if (argc > 5) {
      string err;
      num_keys_per_tx = strict_strtol(argv[5], 10, &err);
      if (!err.empty()) {
        std::cerr << "invalid num_keys_per_tx: " << err << std::endl;
        return 1;
      }
    }
    string other_store_type = argv[1];
    if (argc > 6) {
      other_store_type = argv[6];
    }

    int ret = st.copy_store_to(argv[1], argv[4], num_keys_per_tx, other_store_type);
    if (ret < 0) {
      std::cerr << "error copying store to path '" << argv[4]
                << "': " << cpp_strerror(ret) << std::endl;
      return 1;
    }

  } else if (cmd == "store-crc") {
    if (argc < 4) {
      usage(argv[0]);
      return 1;
    }
    std::ofstream fs(argv[4]);
    uint32_t crc = st.traverse(string(), true, &fs);
    std::cout << "store at '" << argv[4] << "' crc " << crc << std::endl;

  } else if (cmd == "compact") {
    st.compact();
  } else if (cmd == "compact-prefix") {
    if (argc < 5) {
      usage(argv[0]);
      return 1;
    }
    string prefix(url_unescape(argv[4]));
    st.compact_prefix(prefix);
  } else if (cmd == "compact-range") {
    if (argc < 7) {
      usage(argv[0]);
      return 1;
    }
    string prefix(url_unescape(argv[4]));
    string start(url_unescape(argv[5]));
    string end(url_unescape(argv[6]));
    st.compact_range(prefix, start, end);
  } else {
    std::cerr << "Unrecognized command: " << cmd << std::endl;
    return 1;
  }

  return 0;
}
