import cherrypy
import json
import errno
import math
import os
import socket
import threading
from collections import OrderedDict
from mgr_module import MgrModule, MgrStandbyModule, CommandResult

# Defaults for the Prometheus HTTP server.  Can also set in config-key
# see https://github.com/prometheus/prometheus/wiki/Default-port-allocations
# for Prometheus exporter port registry

DEFAULT_ADDR = '::'
DEFAULT_PORT = 9283


# cherrypy likes to sys.exit on error.  don't let it take us down too!
def os_exit_noop(*args, **kwargs):
    pass


os._exit = os_exit_noop


# to access things in class Module from subclass Root.  Because
# it's a dict, the writer doesn't need to declare 'global' for access

_global_instance = {'plugin': None}


def global_instance():
    assert _global_instance['plugin'] is not None
    return _global_instance['plugin']


def health_status_to_number(status):

    if status == 'HEALTH_OK':
        return 0
    elif status == 'HEALTH_WARN':
        return 1
    elif status == 'HEALTH_ERR':
        return 2

PG_STATES = [
        "active",
        "clean",
        "down",
        "recovery_unfound",
        "backfill_unfound",
        "scrubbing",
        "degraded",
        "inconsistent",
        "peering",
        "repair",
        "recovering",
        "forced_recovery",
        "backfill_wait",
        "incomplete",
        "stale",
        "remapped",
        "deep",
        "backfilling",
        "forced_backfill",
        "backfill_toofull",
        "recovery_wait",
        "recovery_toofull",
        "undersized",
        "activating",
        "peered",
        "snaptrim",
        "snaptrim_wait",
        "snaptrim_error",
        "creating",
        "unknown"]

DF_CLUSTER = ['total_bytes', 'total_used_bytes', 'total_objects']

DF_POOL = ['max_avail', 'bytes_used', 'raw_bytes_used', 'objects', 'dirty',
           'quota_bytes', 'quota_objects', 'rd', 'rd_bytes', 'wr', 'wr_bytes']

OSD_FLAGS = ('noup', 'nodown', 'noout', 'noin', 'nobackfill', 'norebalance',
             'norecover', 'noscrub', 'nodeep-scrub')

FS_METADATA = ('data_pools', 'fs_id', 'metadata_pool', 'name')

MDS_METADATA = ('ceph_daemon', 'fs_id', 'hostname', 'public_addr', 'rank',
                'ceph_version')

MON_METADATA = ('ceph_daemon', 'hostname', 'public_addr', 'rank', 'ceph_version')

OSD_METADATA = ('ceph_daemon', 'cluster_addr', 'device_class', 'hostname',
                'public_addr', 'ceph_version')

OSD_STATUS = ['weight', 'up', 'in']

OSD_STATS = ['apply_latency_ms', 'commit_latency_ms']

POOL_METADATA = ('pool_id', 'name')

RGW_METADATA = ('ceph_daemon', 'hostname', 'ceph_version')

DISK_OCCUPATION = ( 'ceph_daemon', 'device','instance')

NUM_OBJECTS = ['degraded', 'misplaced', 'unfound']


class Metrics(object):
    def __init__(self):
        self.metrics = self._setup_static_metrics()
        self.pending = {}

    def set(self, key, value, labels=('',)):
        '''
        Set the value of a single Metrics. This should be used for static metrics,
        e.g. cluster health.
        '''
        self.metrics[key].set(value, labels)

    def append(self, key, value, labels = ('',)):
        '''
        Append a metrics to the staging area. Use this to aggregate daemon specific
        metrics that can appear and go away as daemons are added or removed.
        '''
        if key not in self.pending:
            self.pending[key] = []
        self.pending[key].append((labels, value))

    def reset(self):
        '''
        When metrics aggregation is done, call Metrics.reset() to apply the
        aggregated metric. This will remove all label -> value mappings for a
        metric and set the new mapping (from pending). This means daemon specific
        metrics os daemons that do no longer exist, are removed.
        '''
        for k, v in self.pending.items():
            self.metrics[k].reset(v)
        self.pending = {}

    def add_metric(self, path, metric):
        if path not in self.metrics:
            self.metrics[path] = metric


    def _setup_static_metrics(self):
        metrics = {}
        metrics['health_status'] = Metric(
            'untyped',
            'health_status',
            'Cluster health status'
        )
        metrics['mon_quorum_status'] = Metric(
            'gauge',
            'mon_quorum_status',
            'Monitors in quorum',
            ('ceph_daemon',)
        )
        metrics['fs_metadata'] = Metric(
            'untyped',
            'fs_metadata',
            'FS Metadata',
            FS_METADATA
        )
        metrics['mds_metadata'] = Metric(
            'untyped',
            'mds_metadata',
            'MDS Metadata',
            MDS_METADATA
        )
        metrics['mon_metadata'] = Metric(
            'untyped',
            'mon_metadata',
            'MON Metadata',
            MON_METADATA
        )
        metrics['osd_metadata'] = Metric(
            'untyped',
            'osd_metadata',
            'OSD Metadata',
            OSD_METADATA
        )

        # The reason for having this separate to OSD_METADATA is
        # so that we can stably use the same tag names that
        # the Prometheus node_exporter does
        metrics['disk_occupation'] = Metric(
            'untyped',
            'disk_occupation',
            'Associate Ceph daemon with disk used',
            DISK_OCCUPATION
        )

        metrics['pool_metadata'] = Metric(
            'untyped',
            'pool_metadata',
            'POOL Metadata',
            POOL_METADATA
        )

        metrics['rgw_metadata'] = Metric(
            'untyped',
            'rgw_metadata',
            'RGW Metadata',
            RGW_METADATA
        )

        metrics['pg_total'] = Metric(
            'gauge',
            'pg_total',
            'PG Total Count'
        )

        for flag in OSD_FLAGS:
            path = 'osd_flag_{}'.format(flag)
            metrics[path] = Metric(
                'untyped',
                path,
                'OSD Flag {}'.format(flag)
            )
        for state in OSD_STATUS:
            path = 'osd_{}'.format(state)
            metrics[path] = Metric(
                'untyped',
                path,
                'OSD status {}'.format(state),
                ('ceph_daemon',)
            )
        for stat in OSD_STATS:
            path = 'osd_{}'.format(stat)
            metrics[path] = Metric(
                'gauge',
                path,
                'OSD stat {}'.format(stat),
                ('ceph_daemon',)
            )
        for state in PG_STATES:
            path = 'pg_{}'.format(state)
            metrics[path] = Metric(
                'gauge',
                path,
                'PG {}'.format(state),
            )
        for state in DF_CLUSTER:
            path = 'cluster_{}'.format(state)
            metrics[path] = Metric(
                'gauge',
                path,
                'DF {}'.format(state),
            )
        for state in DF_POOL:
            path = 'pool_{}'.format(state)
            metrics[path] = Metric(
                'gauge',
                path,
                'DF pool {}'.format(state),
                ('pool_id',)
            )
        for state in NUM_OBJECTS:
            path = 'num_objects_{}'.format(state)
            metrics[path] = Metric(
                'gauge',
                path,
                'Number of {} objects'.format(state),
            )

        return metrics



class Metric(object):
    def __init__(self, mtype, name, desc, labels=None):
        self.mtype = mtype
        self.name = name
        self.desc = desc
        self.labelnames = labels    # tuple if present
        self.value = {}             # indexed by label values

    def set(self, value, labelvalues=None):
        # labelvalues must be a tuple
        labelvalues = labelvalues or ('',)
        self.value[labelvalues] = value

    def reset(self, values):
        self.value = {}
        for labelvalues, value in values:
            self.value[labelvalues] = value

    def str_expfmt(self):

        def promethize(path):
            ''' replace illegal metric name characters '''
            result = path.replace('.', '_').replace('+', '_plus').replace('::', '_')

            # Hyphens usually turn into underscores, unless they are
            # trailing
            if result.endswith("-"):
                result = result[0:-1] + "_minus"
            else:
                result = result.replace("-", "_")

            return "ceph_{0}".format(result)

        def floatstr(value):
            ''' represent as Go-compatible float '''
            if value == float('inf'):
                return '+Inf'
            if value == float('-inf'):
                return '-Inf'
            if math.isnan(value):
                return 'NaN'
            return repr(float(value))

        name = promethize(self.name)
        expfmt = '''
# HELP {name} {desc}
# TYPE {name} {mtype}'''.format(
            name=name,
            desc=self.desc,
            mtype=self.mtype,
        )

        for labelvalues, value in self.value.items():
            if self.labelnames:
                labels = zip(self.labelnames, labelvalues)
                labels = ','.join('%s="%s"' % (k, v) for k, v in labels)
            else:
                labels = ''
            if labels:
                fmtstr = '\n{name}{{{labels}}} {value}'
            else:
                fmtstr = '\n{name} {value}'
            expfmt += fmtstr.format(
                name=name,
                labels=labels,
                value=floatstr(value),
            )
        return expfmt


class Module(MgrModule):
    COMMANDS = [
        {
            "cmd": "prometheus self-test",
            "desc": "Run a self test on the prometheus module",
            "perm": "rw"
        },
        {
            "cmd": "prometheus file_sd_config",
            "desc": "Return file_sd compatible prometheus config for mgr cluster",
            "perm": "r"
        },
    ]

    OPTIONS = [
            {'name': 'server_addr'},
            {'name': 'server_port'},
    ]

    def __init__(self, *args, **kwargs):
        super(Module, self).__init__(*args, **kwargs)
        self.metrics = Metrics()
        self.schema = OrderedDict()
        self.shutdown_event = threading.Event()
        _global_instance['plugin'] = self

    def get_health(self):
        health = json.loads(self.get('health')['json'])
        self.metrics.set('health_status',
                         health_status_to_number(health['status'])
        )

    def get_df(self):
        # maybe get the to-be-exported metrics from a config?
        df = self.get('df')
        for stat in DF_CLUSTER:
            self.metrics.set('cluster_{}'.format(stat), df['stats'][stat])

        for pool in df['pools']:
            for stat in DF_POOL:
                self.metrics.append('pool_{}'.format(stat),
                                    pool['stats'][stat],
                                    (pool['id'],))

    def get_fs(self):
        fs_map = self.get('fs_map')
        servers = self.get_service_list()
        active_daemons = []
        for fs in fs_map['filesystems']:
            # collect fs metadata
            data_pools = ",".join([str(pool) for pool in fs['mdsmap']['data_pools']])
            self.metrics.append('fs_metadata', 1,
                                (data_pools,
                                 fs['id'],
                                 fs['mdsmap']['metadata_pool'],
                                 fs['mdsmap']['fs_name']))
            self.log.debug('mdsmap: {}'.format(fs['mdsmap']))
            for gid, daemon in fs['mdsmap']['info'].items():
                id_ = daemon['name']
                host_version = servers.get((id_, 'mds'), ('',''))
                self.metrics.append('mds_metadata', 1,
                                    ('mds.{}'.format(id_), fs['id'],
                                     host_version[0], daemon['addr'],
                                     daemon['rank'], host_version[1]))

    def get_quorum_status(self):
        mon_status = json.loads(self.get('mon_status')['json'])
        servers = self.get_service_list()
        for mon in mon_status['monmap']['mons']:
            rank = mon['rank']
            id_ = mon['name']
            host_version = servers.get((id_, 'mon'), ('',''))
            self.metrics.append('mon_metadata', 1,
                                ('mon.{}'.format(id_), host_version[0],
                                 mon['public_addr'].split(':')[0], rank,
                                 host_version[1]))
            in_quorum = int(rank in mon_status['quorum'])
            self.metrics.append('mon_quorum_status', in_quorum,
                                ('mon.{}'.format(id_),))

    def get_pg_status(self):
        # TODO add per pool status?
        pg_status = self.get('pg_status')

        # Set total count of PGs, first
        self.metrics.set('pg_total', pg_status['num_pgs'])

        reported_states = {}
        for pg in pg_status['pgs_by_state']:
            for state in pg['state_name'].split('+'):
                reported_states[state] =  reported_states.get(state, 0) + pg['count']

        for state in reported_states:
            path = 'pg_{}'.format(state)
            try:
                self.metrics.set(path, reported_states[state])
            except KeyError:
                self.log.warn("skipping pg in unknown state {}".format(state))

        for state in PG_STATES:
            if state not in reported_states:
                try:
                    self.metrics.set('pg_{}'.format(state), 0)
                except KeyError:
                    self.log.warn("skipping pg in unknown state {}".format(state))

    def get_osd_stats(self):
        osd_stats = self.get('osd_stats')
        for osd in osd_stats['osd_stats']:
            id_ = osd['osd']
            for stat in OSD_STATS:
                val = osd['perf_stat'][stat]
                self.metrics.append('osd_{}'.format(stat), val,
                                    ('osd.{}'.format(id_),))

    def get_service_list(self):
        ret = {}
        for server in self.list_servers():
            version = server.get('ceph_version', '')
            host = server.get('hostname', '')
            for service in server.get('services', []):
                ret.update({(service['id'], service['type']): (host, version)})
        return ret

    def get_metadata_and_osd_status(self):
        osd_map = self.get('osd_map')
        osd_flags = osd_map['flags'].split(',')
        for flag in OSD_FLAGS:
            self.metrics.set('osd_flag_{}'.format(flag),
                int(flag in osd_flags))

        osd_devices = self.get('osd_map_crush')['devices']
        servers = self.get_service_list()
        for osd in osd_map['osds']:
            # id can be used to link osd metrics and metadata
            id_ = osd['osd']
            # collect osd metadata
            p_addr = osd['public_addr'].split(':')[0]
            c_addr = osd['cluster_addr'].split(':')[0]
            if p_addr == "-" or c_addr == "-":
                self.log.info(
                    "Missing address metadata for osd {0}, skipping occupation"
                    " and metadata records for this osd".format(id_)
                )
                continue

            dev_class = None
            for osd_device in osd_devices:
                if osd_device['id'] == id_:
                    dev_class = osd_device.get('class', '')
                    break

            if dev_class is None:
                self.log.info(
                    "OSD {0} is missing from CRUSH map, skipping output".format(
                        id_))
                continue

            host_version = servers.get((str(id_), 'osd'), ('',''))

            self.metrics.append('osd_metadata', 1, (
                'osd.{}'.format(id_),
                c_addr,
                dev_class,
                host_version[0],
                p_addr, host_version[1]
            ))

            # collect osd status
            for state in OSD_STATUS:
                status = osd[state]
                self.metrics.append('osd_{}'.format(state), status,
                                    ('osd.{}'.format(id_),))

            # collect disk occupation metadata
            osd_metadata = self.get_metadata("osd", str(id_))
            if osd_metadata is None:
                continue
            dev_keys = ("backend_filestore_dev_node", "bluestore_bdev_dev_node")
            osd_dev_node = None
            for dev_key in dev_keys:
                val = osd_metadata.get(dev_key, None)
                if val and val != "unknown":
                    osd_dev_node = val
                    break
            osd_hostname = osd_metadata.get('hostname', None)
            if osd_dev_node and osd_hostname:
                self.log.debug("Got dev for osd {0}: {1}/{2}".format(
                    id_, osd_hostname, osd_dev_node))
                self.metrics.set('disk_occupation', 1, (
                    "osd.{0}".format(id_),
                    osd_hostname,
                    osd_dev_node
                ))
            else:
                self.log.info("Missing dev node metadata for osd {0}, skipping "
                               "occupation record for this osd".format(id_))

        pool_meta = []
        for pool in osd_map['pools']:
            self.metrics.append('pool_metadata', 1, (pool['pool'], pool['pool_name']))

        # Populate rgw_metadata
        for key, value in servers.items():
            service_id, service_type = key
            if service_type != 'rgw':
                continue
            hostname, version = value
            self.metrics.append(
                'rgw_metadata',
                1,
                ('{}.{}'.format(service_type, service_id), hostname, version)
            )

    def get_num_objects(self):
        pg_sum = self.get('pg_summary')['pg_stats_sum']['stat_sum']
        for obj in NUM_OBJECTS:
            stat = 'num_objects_{}'.format(obj)
            self.metrics.set(stat, pg_sum[stat])

    def collect(self):
        self.get_health()
        self.get_df()
        self.get_fs()
        self.get_osd_stats()
        self.get_quorum_status()
        self.get_metadata_and_osd_status()
        self.get_pg_status()
        self.get_num_objects()

        for daemon, counters in self.get_all_perf_counters().items():
            for path, counter_info in counters.items():
                stattype = self._stattype_to_str(counter_info['type'])
                # XXX simplify first effort: no histograms
                # averages are already collapsed to one value for us
                if not stattype or stattype == 'histogram':
                    self.log.debug('ignoring %s, type %s' % (path, stattype))
                    continue

                self.metrics.add_metric(path, Metric(
                        stattype,
                        path,
                        counter_info['description'],
                        ("ceph_daemon",),
                    ))

                value = self._perfvalue_to_value(counter_info['type'], counter_info['value'])
                self.metrics.append(path, value, (daemon,))
        # It is sufficient to reset the pending metrics once per scrape
        self.metrics.reset()

        return self.metrics.metrics

    def get_file_sd_config(self):
        servers = self.list_servers()
        targets = []
        for server in servers:
            hostname = server.get('hostname', '')
            for service in server.get('services', []):
                if service['type'] != 'mgr':
                    continue
                id_ = service['id']
                # get port for prometheus module at mgr with id_
                # TODO use get_config_prefix or get_config here once
                # https://github.com/ceph/ceph/pull/20458 is merged
                result = CommandResult("")
                global_instance().send_command(
                    result, "mon", '',
                    json.dumps({
                        "prefix": "config-key get",
                        'key': "config/mgr/mgr/prometheus/{}/server_port".format(id_),
                    }),
                                               "")
                r, outb, outs = result.wait()
                if r != 0:
                    global_instance().log.error("Failed to retrieve port for mgr {}: {}".format(id_, outs))
                    targets.append('{}:{}'.format(hostname, DEFAULT_PORT))
                else:
                    port = json.loads(outb)
                    targets.append('{}:{}'.format(hostname, port))

        ret = [
            {
                "targets": targets,
                "labels": {}
            }
        ]
        return 0, json.dumps(ret), ""

    def handle_command(self, cmd):
        if cmd['prefix'] == 'prometheus self-test':
            self.collect()
            self.get_file_sd_config()
            return 0, '', 'Self-test OK'
        elif cmd['prefix'] == 'prometheus file_sd_config':
            return self.get_file_sd_config()
        else:
            return (-errno.EINVAL, '',
                    "Command not found '{0}'".format(cmd['prefix']))

    def serve(self):

        class Root(object):

            # collapse everything to '/'
            def _cp_dispatch(self, vpath):
                cherrypy.request.path = ''
                return self

            def format_metrics(self, metrics):
                formatted = ''
                for m in metrics.values():
                    formatted += m.str_expfmt()
                return formatted + '\n'

            @cherrypy.expose
            def index(self):
                return '''<!DOCTYPE html>
<html>
	<head><title>Ceph Exporter</title></head>
	<body>
		<h1>Ceph Exporter</h1>
		<p><a href='/metrics'>Metrics</a></p>
	</body>
</html>'''

            @cherrypy.expose
            def metrics(self):
                if global_instance().have_mon_connection():
                    metrics = global_instance().collect()
                    cherrypy.response.headers['Content-Type'] = 'text/plain'
                    if metrics:
                        return self.format_metrics(metrics)
                else:
                    raise cherrypy.HTTPError(503, 'No MON connection')

        server_addr = self.get_localized_config('server_addr', DEFAULT_ADDR)
        server_port = self.get_localized_config('server_port', DEFAULT_PORT)
        self.log.info(
            "server_addr: %s server_port: %s" %
            (server_addr, server_port)
        )

        # Publish the URI that others may use to access the service we're
        # about to start serving
        self.set_uri('http://{0}:{1}/'.format(
            socket.getfqdn() if server_addr == '::' else server_addr,
            server_port
        ))

        cherrypy.config.update({
            'server.socket_host': server_addr,
            'server.socket_port': int(server_port),
            'engine.autoreload.on': False
        })
        cherrypy.tree.mount(Root(), "/")
        self.log.info('Starting engine...')
        cherrypy.engine.start()
        self.log.info('Engine started.')
        # wait for the shutdown event
        self.shutdown_event.wait()
        self.shutdown_event.clear()
        cherrypy.engine.stop()
        self.log.info('Engine stopped.')

    def shutdown(self):
        self.log.info('Stopping engine...')
        self.shutdown_event.set()


class StandbyModule(MgrStandbyModule):
    def __init__(self, *args, **kwargs):
        super(StandbyModule, self).__init__(*args, **kwargs)
        self.shutdown_event = threading.Event()

    def serve(self):
        server_addr = self.get_localized_config('server_addr', '::')
        server_port = self.get_localized_config('server_port', DEFAULT_PORT)
        self.log.info("server_addr: %s server_port: %s" % (server_addr, server_port))
        cherrypy.config.update({
            'server.socket_host': server_addr,
            'server.socket_port': int(server_port),
            'engine.autoreload.on': False
        })

        module = self

        class Root(object):

            @cherrypy.expose
            def index(self):
                active_uri = module.get_active_uri()
                return '''<!DOCTYPE html>
<html>
	<head><title>Ceph Exporter</title></head>
	<body>
		<h1>Ceph Exporter</h1>
        <p><a href='{}metrics'>Metrics</a></p>
	</body>
</html>'''.format(active_uri)

            @cherrypy.expose
            def metrics(self):
                cherrypy.response.headers['Content-Type'] = 'text/plain'
                return ''

        cherrypy.tree.mount(Root(), '/', {})
        self.log.info('Starting engine...')
        cherrypy.engine.start()
        self.log.info('Engine started.')
        # Wait for shutdown event
        self.shutdown_event.wait()
        self.shutdown_event.clear()
        cherrypy.engine.stop()
        self.log.info('Engine stopped.')

    def shutdown(self):
        self.log.info("Stopping engine...")
        self.shutdown_event.set()
        self.log.info("Stopped engine")
