import errno
import json
import socket
import time
from threading import Event

from telegraf.basesocket import BaseSocket
from telegraf.protocol import Line
from mgr_module import MgrModule

try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse


class Module(MgrModule):
    COMMANDS = [
        {
            "cmd": "telegraf config-set name=key,type=CephString "
                   "name=value,type=CephString",
            "desc": "Set a configuration value",
            "perm": "rw"
        },
        {
            "cmd": "telegraf config-show",
            "desc": "Show current configuration",
            "perm": "r"
        },
        {
            "cmd": "telegraf send",
            "desc": "Force sending data to Telegraf",
            "perm": "rw"
        },
        {
            "cmd": "telegraf self-test",
            "desc": "debug the module",
            "perm": "rw"
        },
    ]

    OPTIONS = [
        {
            'name': 'address',
            'default': 'unixgram:///tmp/telegraf.sock',
        },
        {
            'name': 'interval',
            'default': 15
        }
    ]

    ceph_health_mapping = {'HEALTH_OK': 0, 'HEALTH_WARN': 1, 'HEALTH_ERR': 2}

    @property
    def config_keys(self):
        return dict((o['name'], o.get('default', None)) for o in self.OPTIONS)

    def __init__(self, *args, **kwargs):
        super(Module, self).__init__(*args, **kwargs)
        self.event = Event()
        self.run = True
        self.fsid = None
        self.config = dict()

    def get_fsid(self):
        if not self.fsid:
            self.fsid = self.get('mon_map')['fsid']

        return self.fsid

    def get_pool_stats(self):
        df = self.get('df')
        data = []

        df_types = [
            'bytes_used',
            'kb_used',
            'dirty',
            'rd',
            'rd_bytes',
            'raw_bytes_used',
            'wr',
            'wr_bytes',
            'objects',
            'max_avail',
            'quota_objects',
            'quota_bytes'
        ]

        for df_type in df_types:
            for pool in df['pools']:
                point = {
                    'measurement': 'ceph_pool_stats',
                    'tags': {
                        'pool_name': pool['name'],
                        'pool_id': pool['id'],
                        'type_instance': df_type,
                        'fsid': self.get_fsid()
                    },
                    'value': pool['stats'][df_type],
                }
                data.append(point)
        return data

    def get_daemon_stats(self):
        data = []

        for daemon, counters in self.get_all_perf_counters().iteritems():
            svc_type, svc_id = daemon.split('.', 1)
            metadata = self.get_metadata(svc_type, svc_id)

            for path, counter_info in counters.items():
                if counter_info['type'] & self.PERFCOUNTER_HISTOGRAM:
                    continue

                data.append({
                    'measurement': 'ceph_daemon_stats',
                    'tags': {
                        'ceph_daemon': daemon,
                        'type_instance': path,
                        'host': metadata['hostname'],
                        'fsid': self.get_fsid()
                    },
                    'value': counter_info['value']
                })

        return data

    def get_cluster_stats(self):
        stats = dict()

        health = json.loads(self.get('health')['json'])
        stats['health'] = self.ceph_health_mapping.get(health['status'])

        mon_status = json.loads(self.get('mon_status')['json'])
        stats['num_mon'] = len(mon_status['monmap']['mons'])

        stats['mon_election_epoch'] = mon_status['election_epoch']
        stats['mon_outside_quorum'] = len(mon_status['outside_quorum'])
        stats['mon_quorum'] = len(mon_status['quorum'])

        osd_map = self.get('osd_map')
        stats['num_osd'] = len(osd_map['osds'])
        stats['num_pg_temp'] = len(osd_map['pg_temp'])
        stats['osd_epoch'] = osd_map['epoch']

        mgr_map = self.get('mgr_map')
        stats['mgr_available'] = int(mgr_map['available'])
        stats['num_mgr_standby'] = len(mgr_map['standbys'])
        stats['mgr_epoch'] = mgr_map['epoch']

        num_up = 0
        num_in = 0
        for osd in osd_map['osds']:
            if osd['up'] == 1:
                num_up += 1

            if osd['in'] == 1:
                num_in += 1

        stats['num_osd_up'] = num_up
        stats['num_osd_in'] = num_in

        fs_map = self.get('fs_map')
        stats['num_mds_standby'] = len(fs_map['standbys'])
        stats['num_fs'] = len(fs_map['filesystems'])
        stats['mds_epoch'] = fs_map['epoch']

        num_mds_up = 0
        for fs in fs_map['filesystems']:
            num_mds_up += len(fs['mdsmap']['up'])

        stats['num_mds_up'] = num_mds_up
        stats['num_mds'] = num_mds_up + stats['num_mds_standby']

        pg_status = self.get('pg_status')
        for key in ['bytes_total', 'data_bytes', 'bytes_used', 'bytes_avail',
                    'num_pgs', 'num_objects', 'num_pools']:
            stats[key] = pg_status[key]

        stats['num_pgs_active'] = 0
        stats['num_pgs_clean'] = 0
        stats['num_pgs_scrubbing'] = 0
        stats['num_pgs_peering'] = 0
        for state in pg_status['pgs_by_state']:
            states = state['state_name'].split('+')

            if 'active' in states:
                stats['num_pgs_active'] += state['count']

            if 'clean' in states:
                stats['num_pgs_clean'] += state['count']

            if 'peering' in states:
                stats['num_pgs_peering'] += state['count']

            if 'scrubbing' in states:
                stats['num_pgs_scrubbing'] += state['count']

        data = list()
        for key, value in stats.items():
            data.append({
                'measurement': 'ceph_cluster_stats',
                'tags': {
                    'type_instance': key,
                    'fsid': self.get_fsid()
                },
                'value': int(value)
            })

        return data

    def set_config_option(self, option, value):
        if option not in self.config_keys.keys():
            raise RuntimeError('{0} is a unknown configuration '
                               'option'.format(option))

        if option in ['interval']:
            try:
                value = int(value)
            except (ValueError, TypeError):
                raise RuntimeError('invalid {0} configured. Please specify '
                                   'a valid integer'.format(option))

        if option == 'interval' and value < 5:
            raise RuntimeError('interval should be set to at least 5 seconds')

        self.config[option] = value

    def init_module_config(self):
        self.config['address'] = \
            self.get_config("address", default=self.config_keys['address'])
        self.config['interval'] = \
            int(self.get_config("interval",
                                default=self.config_keys['interval']))

    def now(self):
        return int(round(time.time() * 1000000000))

    def gather_measurements(self):
        measurements = list()
        measurements += self.get_pool_stats()
        measurements += self.get_daemon_stats()
        measurements += self.get_cluster_stats()
        return measurements

    def send_to_telegraf(self):
        url = urlparse(self.config['address'])

        sock = BaseSocket(url)
        self.log.debug('Sending data to Telegraf at %s', sock.address)
        now = self.now()
        with sock as s:
            try:
                for measurement in self.gather_measurements():
                    self.log.debug(measurement)
                    line = Line(measurement['measurement'],
                                measurement['value'],
                                measurement['tags'], now)
                    self.log.debug(line.to_line_protocol())
                    s.send(line.to_line_protocol())
            except (socket.error, RuntimeError, errno, IOError):
                self.log.exception('Failed to send statistics to Telegraf:')

    def shutdown(self):
        self.log.info('Stopping Telegraf module')
        self.run = False
        self.event.set()

    def handle_command(self, cmd):
        if cmd['prefix'] == 'telegraf config-show':
            return 0, json.dumps(self.config), ''
        elif cmd['prefix'] == 'telegraf config-set':
            key = cmd['key']
            value = cmd['value']
            if not value:
                return -errno.EINVAL, '', 'Value should not be empty or None'

            self.log.debug('Setting configuration option %s to %s', key, value)
            self.set_config_option(key, value)
            self.set_config(key, value)
            return 0, 'Configuration option {0} updated'.format(key), ''
        elif cmd['prefix'] == 'telegraf send':
            self.send_to_telegraf()
            return 0, 'Sending data to Telegraf', ''
        if cmd['prefix'] == 'telegraf self-test':
            self.self_test()
            return 0, '', 'Self-test OK'

        return (-errno.EINVAL, '',
                "Command not found '{0}'".format(cmd['prefix']))

    def self_test(self):
        measurements = self.gather_measurements()
        if len(measurements) == 0:
            raise RuntimeError('No measurements found')

    def serve(self):
        self.log.info('Starting Telegraf module')
        self.init_module_config()
        self.run = True

        self.log.debug('Waiting 10 seconds before starting')
        self.event.wait(10)

        while self.run:
            start = self.now()
            self.send_to_telegraf()
            runtime = (self.now() - start) / 1000000
            self.log.debug('Sending data to Telegraf took %d ms', runtime)
            self.log.debug("Sleeping for %d seconds", self.config['interval'])
            self.event.wait(self.config['interval'])
