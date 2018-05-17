# -*- coding: utf-8 -*-
from __future__ import absolute_import

import sys
import inspect
import functools

import collections
from datetime import datetime, timedelta
import fnmatch
import time
import threading
import socket
from six.moves import urllib
import cherrypy

from . import logger
from .exceptions import ViewCacheNoDataException


class RequestLoggingTool(cherrypy.Tool):
    def __init__(self):
        cherrypy.Tool.__init__(self, 'before_handler', self.request_begin,
                               priority=95)

    def _setup(self):
        cherrypy.Tool._setup(self)
        cherrypy.request.hooks.attach('on_end_request', self.request_end,
                                      priority=5)
        cherrypy.request.hooks.attach('after_error_response', self.request_error,
                                      priority=5)

    def _get_user(self):
        if hasattr(cherrypy.serving, 'session'):
            return cherrypy.session.get(Session.USERNAME)
        return None

    def request_begin(self):
        req = cherrypy.request
        user = self._get_user()
        if user:
            logger.debug("[%s:%s] [%s] [%s] %s", req.remote.ip,
                         req.remote.port, req.method, user, req.path_info)
        else:
            logger.debug("[%s:%s] [%s] %s", req.remote.ip,
                         req.remote.port, req.method, req.path_info)

    def request_error(self):
        self._request_log(logger.error)
        logger.error(cherrypy.response.body)

    def request_end(self):
        status = cherrypy.response.status[:3]
        if status in ["401"]:
            # log unauthorized accesses
            self._request_log(logger.warning)
        else:
            self._request_log(logger.info)

    def _format_bytes(self, num):
        units = ['B', 'K', 'M', 'G']

        if isinstance(num, str):
            try:
                num = int(num)
            except ValueError:
                return "n/a"

        format_str = "{:.0f}{}"
        for i, unit in enumerate(units):
            div = 2**(10*i)
            if num < 2**(10*(i+1)):
                if num % div == 0:
                    format_str = "{}{}"
                else:
                    div = float(div)
                    format_str = "{:.1f}{}"
                return format_str.format(num/div, unit[0])

        # content-length bigger than 1T!! return value in bytes
        return "{}B".format(num)

    def _request_log(self, logger_fn):
        req = cherrypy.request
        res = cherrypy.response
        lat = time.time() - res.time
        user = self._get_user()
        status = res.status[:3] if isinstance(res.status, str) else res.status
        if 'Content-Length' in res.headers:
            length = self._format_bytes(res.headers['Content-Length'])
        else:
            length = self._format_bytes(0)
        if user:
            logger_fn("[%s:%s] [%s] [%s] [%s] [%s] [%s] %s", req.remote.ip,
                      req.remote.port, req.method, status,
                      "{0:.3f}s".format(lat), user, length, req.path_info)
        else:
            logger_fn("[%s:%s] [%s] [%s] [%s] [%s] %s", req.remote.ip,
                      req.remote.port, req.method, status,
                      "{0:.3f}s".format(lat), length, req.path_info)


# pylint: disable=too-many-instance-attributes
class ViewCache(object):
    VALUE_OK = 0
    VALUE_STALE = 1
    VALUE_NONE = 2

    class GetterThread(threading.Thread):
        def __init__(self, view, fn, args, kwargs):
            super(ViewCache.GetterThread, self).__init__()
            self._view = view
            self.event = threading.Event()
            self.fn = fn
            self.args = args
            self.kwargs = kwargs

        # pylint: disable=broad-except
        def run(self):
            t0 = 0.0
            t1 = 0.0
            try:
                t0 = time.time()
                logger.debug("VC: starting execution of %s", self.fn)
                val = self.fn(*self.args, **self.kwargs)
                t1 = time.time()
            except Exception as ex:
                with self._view.lock:
                    logger.exception("Error while calling fn=%s ex=%s", self.fn,
                                     str(ex))
                    self._view.value = None
                    self._view.value_when = None
                    self._view.getter_thread = None
                    self._view.exception = ex
            else:
                with self._view.lock:
                    self._view.latency = t1 - t0
                    self._view.value = val
                    self._view.value_when = datetime.now()
                    self._view.getter_thread = None
                    self._view.exception = None

            logger.debug("VC: execution of %s finished in: %s", self.fn,
                         t1 - t0)
            self.event.set()

    class RemoteViewCache(object):
        # Return stale data if
        STALE_PERIOD = 1.0

        def __init__(self, timeout):
            self.getter_thread = None
            # Consider data within 1s old to be sufficiently fresh
            self.timeout = timeout
            self.event = threading.Event()
            self.value_when = None
            self.value = None
            self.latency = 0
            self.exception = None
            self.lock = threading.Lock()

        def run(self, fn, args, kwargs):
            """
            If data less than `stale_period` old is available, return it
            immediately.
            If an attempt to fetch data does not complete within `timeout`, then
            return the most recent data available, with a status to indicate that
            it is stale.

            Initialization does not count towards the timeout, so the first call
            on one of these objects during the process lifetime may be slower
            than subsequent calls.

            :return: 2-tuple of value status code, value
            """
            with self.lock:
                now = datetime.now()
                if self.value_when and now - self.value_when < timedelta(
                        seconds=self.STALE_PERIOD):
                    return ViewCache.VALUE_OK, self.value

                if self.getter_thread is None:
                    self.getter_thread = ViewCache.GetterThread(self, fn, args,
                                                                kwargs)
                    self.getter_thread.start()
                else:
                    logger.debug("VC: getter_thread still alive for: %s", fn)

                ev = self.getter_thread.event

            success = ev.wait(timeout=self.timeout)

            with self.lock:
                if success:
                    # We fetched the data within the timeout
                    if self.exception:
                        # execution raised an exception
                        # pylint: disable=raising-bad-type
                        raise self.exception
                    return ViewCache.VALUE_OK, self.value
                elif self.value_when is not None:
                    # We have some data, but it doesn't meet freshness requirements
                    return ViewCache.VALUE_STALE, self.value
                # We have no data, not even stale data
                raise ViewCacheNoDataException()

    def __init__(self, timeout=5):
        self.timeout = timeout
        self.cache_by_args = {}

    def __call__(self, fn):
        def wrapper(*args, **kwargs):
            rvc = self.cache_by_args.get(args, None)
            if not rvc:
                rvc = ViewCache.RemoteViewCache(self.timeout)
                self.cache_by_args[args] = rvc
            return rvc.run(fn, args, kwargs)
        return wrapper


class Session(object):
    """
    This class contains all relevant settings related to cherrypy.session.
    """
    NAME = 'session_id'

    # The keys used to store the information in the cherrypy.session.
    USERNAME = '_username'
    TS = '_ts'
    EXPIRE_AT_BROWSER_CLOSE = '_expire_at_browser_close'

    # The default values.
    DEFAULT_EXPIRE = 1200.0


class SessionExpireAtBrowserCloseTool(cherrypy.Tool):
    """
    A CherryPi Tool which takes care that the cookie does not expire
    at browser close if the 'Keep me logged in' checkbox was selected
    on the login page.
    """
    def __init__(self):
        cherrypy.Tool.__init__(self, 'before_finalize', self._callback)

    def _callback(self):
        # Shall the cookie expire at browser close?
        expire_at_browser_close = cherrypy.session.get(
            Session.EXPIRE_AT_BROWSER_CLOSE, True)
        logger.debug("expire at browser close: %s", expire_at_browser_close)
        if expire_at_browser_close:
            # Get the cookie and its name.
            cookie = cherrypy.response.cookie
            name = cherrypy.request.config.get(
                'tools.sessions.name', Session.NAME)
            # Make the cookie a session cookie by purging the
            # fields 'expires' and 'max-age'.
            logger.debug("expire at browser close: removing 'expires' and 'max-age'")
            if name in cookie:
                del cookie[name]['expires']
                del cookie[name]['max-age']


class NotificationQueue(threading.Thread):
    _ALL_TYPES_ = '__ALL__'
    _listeners = collections.defaultdict(set)
    _lock = threading.Lock()
    _cond = threading.Condition()
    _queue = collections.deque()
    _running = False
    _instance = None

    def __init__(self):
        super(NotificationQueue, self).__init__()

    @classmethod
    def start_queue(cls):
        with cls._lock:
            if cls._instance:
                # the queue thread is already running
                return
            cls._running = True
            cls._instance = NotificationQueue()
        logger.debug("starting notification queue")
        cls._instance.start()

    @classmethod
    def stop(cls):
        with cls._lock:
            if not cls._instance:
                # the queue thread was not started
                return
            instance = cls._instance
            cls._instance = None
            cls._running = False
        with cls._cond:
            cls._cond.notify()
        logger.debug("waiting for notification queue to finish")
        instance.join()
        logger.debug("notification queue stopped")

    @classmethod
    def _registered_handler(cls, func, n_types):
        for _, reg_func in cls._listeners[n_types]:
            if reg_func == func:
                return True
        return False

    @classmethod
    def register(cls, func, n_types=None, priority=1):
        """Registers function to listen for notifications

        If the second parameter `n_types` is omitted, the function in `func`
        parameter will be called for any type of notifications.

        Args:
            func (function): python function ex: def foo(val)
            n_types (str|list): the single type to listen, or a list of types
            priority (int): the priority level (1=max, +inf=min)
        """
        with cls._lock:
            if not n_types:
                n_types = [cls._ALL_TYPES_]
            elif isinstance(n_types, str):
                n_types = [n_types]
            elif not isinstance(n_types, list):
                raise Exception("n_types param is neither a string nor a list")
            for ev_type in n_types:
                if not cls._registered_handler(func, ev_type):
                    cls._listeners[ev_type].add((priority, func))
                    logger.debug("NQ: function %s was registered for events of"
                                 " type %s", func, ev_type)

    @classmethod
    def deregister(cls, func, n_types=None):
        """Removes the listener function from this notification queue

        If the second parameter `n_types` is ommitted, the function is removed
        from all event types, otherwise the function is removed only for the
        specified event types.

        Args:
            func (function): python function
            n_types (str|list): the single event type, or a list of event types
        """
        with cls._lock:
            if not n_types:
                n_types = list(cls._listeners.keys())
            elif isinstance(n_types, str):
                n_types = [n_types]
            elif not isinstance(n_types, list):
                raise Exception("n_types param is neither a string nor a list")
            for ev_type in n_types:
                listeners = cls._listeners[ev_type]
                toRemove = None
                for pr, fn in listeners:
                    if fn == func:
                        toRemove = (pr, fn)
                        break
                if toRemove:
                    listeners.discard(toRemove)
                    logger.debug("NQ: function %s was deregistered for events "
                                 "of type %s", func, ev_type)

    @classmethod
    def new_notification(cls, notify_type, notify_value):
        with cls._cond:
            cls._queue.append((notify_type, notify_value))
            cls._cond.notify()

    @classmethod
    def _notify_listeners(cls, events):
        for ev in events:
            notify_type, notify_value = ev
            with cls._lock:
                listeners = list(cls._listeners[notify_type])
                listeners.extend(cls._listeners[cls._ALL_TYPES_])
            listeners.sort(key=lambda lis: lis[0])
            for listener in listeners:
                listener[1](notify_value)

    def run(self):
        logger.debug("notification queue started")
        while self._running:
            private_buffer = []
            logger.debug("NQ: processing queue: %s", len(self._queue))
            try:
                while True:
                    private_buffer.append(self._queue.popleft())
            except IndexError:
                pass
            self._notify_listeners(private_buffer)
            with self._cond:
                while self._running and not self._queue:
                    self._cond.wait()
        # flush remaining events
        logger.debug("NQ: flush remaining events: %s", len(self._queue))
        self._notify_listeners(self._queue)
        self._queue.clear()
        logger.debug("notification queue finished")


# pylint: disable=too-many-arguments, protected-access
class TaskManager(object):
    FINISHED_TASK_SIZE = 10
    FINISHED_TASK_TTL = 60.0

    VALUE_DONE = "done"
    VALUE_EXECUTING = "executing"

    _executing_tasks = set()
    _finished_tasks = []
    _lock = threading.Lock()

    _task_local_data = threading.local()

    @classmethod
    def init(cls):
        NotificationQueue.register(cls._handle_finished_task, 'cd_task_finished')

    @classmethod
    def _handle_finished_task(cls, task):
        logger.info("TM: finished %s", task)
        with cls._lock:
            cls._executing_tasks.remove(task)
            cls._finished_tasks.append(task)

    @classmethod
    def run(cls, name, metadata, fn, args=None, kwargs=None, executor=None,
            exception_handler=None):
        if not args:
            args = []
        if not kwargs:
            kwargs = {}
        if not executor:
            executor = ThreadedExecutor()
        task = Task(name, metadata, fn, args, kwargs, executor,
                    exception_handler)
        with cls._lock:
            if task in cls._executing_tasks:
                logger.debug("TM: task already executing: %s", task)
                for t in cls._executing_tasks:
                    if t == task:
                        return t
            logger.debug("TM: created %s", task)
            cls._executing_tasks.add(task)
        logger.info("TM: running %s", task)
        task._run()
        return task

    @classmethod
    def current_task(cls):
        """
        Returns the current task object.
        This method should only be called from a threaded task operation code.
        """
        return cls._task_local_data.task

    @classmethod
    def _cleanup_old_tasks(cls, task_list):
        """
        The cleanup rule is: maintain the FINISHED_TASK_SIZE more recent
        finished tasks, and the rest is maintained up to the FINISHED_TASK_TTL
        value.
        """
        now = datetime.now()
        for idx, t in enumerate(task_list):
            if idx < cls.FINISHED_TASK_SIZE:
                continue
            if now - datetime.fromtimestamp(t[1].end_time) > \
                    timedelta(seconds=cls.FINISHED_TASK_TTL):
                del cls._finished_tasks[t[0]]

    @classmethod
    def list(cls, name_glob=None):
        executing_tasks = []
        finished_tasks = []
        with cls._lock:
            for task in cls._executing_tasks:
                if not name_glob or fnmatch.fnmatch(task.name, name_glob):
                    executing_tasks.append(task)
            for idx, task in enumerate(cls._finished_tasks):
                if not name_glob or fnmatch.fnmatch(task.name, name_glob):
                    finished_tasks.append((idx, task))
            finished_tasks.sort(key=lambda t: t[1].end_time, reverse=True)
            cls._cleanup_old_tasks(finished_tasks)
        executing_tasks.sort(key=lambda t: t.begin_time, reverse=True)
        return executing_tasks, [t[1] for t in finished_tasks]

    @classmethod
    def list_serializable(cls, ns_glob=None):
        ex_t, fn_t = cls.list(ns_glob)
        return [{
            'name': t.name,
            'metadata': t.metadata,
            'begin_time': "{}Z".format(datetime.fromtimestamp(t.begin_time).isoformat()),
            'progress': t.progress
        } for t in ex_t if t.begin_time], [{
            'name': t.name,
            'metadata': t.metadata,
            'begin_time': "{}Z".format(datetime.fromtimestamp(t.begin_time).isoformat()),
            'end_time': "{}Z".format(datetime.fromtimestamp(t.end_time).isoformat()),
            'duration': t.duration,
            'progress': t.progress,
            'success': not t.exception,
            'ret_value': t.ret_value if not t.exception else None,
            'exception': t.ret_value if t.exception and t.ret_value else (
                {'detail': str(t.exception)} if t.exception else None)
        } for t in fn_t]


# pylint: disable=protected-access
class TaskExecutor(object):
    def __init__(self):
        self.task = None

    def init(self, task):
        self.task = task

    # pylint: disable=broad-except
    def start(self):
        logger.debug("EX: executing task %s", self.task)
        try:
            self.task.fn(*self.task.fn_args, **self.task.fn_kwargs)
        except Exception as ex:
            logger.exception("Error while calling %s", self.task)
            self.finish(None, ex)

    def finish(self, ret_value, exception):
        if not exception:
            logger.debug("EX: successfully finished task: %s", self.task)
        else:
            logger.debug("EX: task finished with exception: %s", self.task)
        self.task._complete(ret_value, exception)


# pylint: disable=protected-access
class ThreadedExecutor(TaskExecutor):
    def __init__(self):
        super(ThreadedExecutor, self).__init__()
        self._thread = threading.Thread(target=self._run)

    def start(self):
        self._thread.start()

    # pylint: disable=broad-except
    def _run(self):
        TaskManager._task_local_data.task = self.task
        try:
            logger.debug("TEX: executing task %s", self.task)
            val = self.task.fn(*self.task.fn_args, **self.task.fn_kwargs)
        except Exception as ex:
            logger.exception("Error while calling %s", self.task)
            self.finish(None, ex)
        else:
            self.finish(val, None)


class Task(object):
    def __init__(self, name, metadata, fn, args, kwargs, executor,
                 exception_handler=None):
        self.name = name
        self.metadata = metadata
        self.fn = fn
        self.fn_args = args
        self.fn_kwargs = kwargs
        self.executor = executor
        self.ex_handler = exception_handler
        self.running = False
        self.event = threading.Event()
        self.progress = None
        self.ret_value = None
        self.begin_time = None
        self.end_time = None
        self.duration = 0
        self.exception = None
        self.lock = threading.Lock()

    def __hash__(self):
        return hash((self.name, tuple(sorted(self.metadata.items()))))

    def __eq__(self, other):
        return self.name == self.name and self.metadata == self.metadata

    def __str__(self):
        return "Task(ns={}, md={})" \
               .format(self.name, self.metadata)

    def __repr__(self):
        return str(self)

    def _run(self):
        with self.lock:
            assert not self.running
            self.executor.init(self)
            self.set_progress(0, in_lock=True)
            self.begin_time = time.time()
            self.running = True
        self.executor.start()

    def _complete(self, ret_value, exception=None):
        now = time.time()
        if exception and self.ex_handler:
            # pylint: disable=broad-except
            try:
                ret_value = self.ex_handler(exception, task=self)
            except Exception as ex:
                exception = ex
        with self.lock:
            assert self.running, "_complete cannot be called before _run"
            self.end_time = now
            self.ret_value = ret_value
            self.exception = exception
            self.duration = now - self.begin_time
            if not self.exception:
                self.set_progress(100, True)
        NotificationQueue.new_notification('cd_task_finished', self)
        self.event.set()
        logger.debug("TK: execution of %s finished in: %s s", self,
                     self.duration)

    def wait(self, timeout=None):
        with self.lock:
            assert self.running, "wait cannot be called before _run"
            ev = self.event

        success = ev.wait(timeout=timeout)
        with self.lock:
            if success:
                # the action executed within the timeout
                if self.exception:
                    # pylint: disable=raising-bad-type
                    # execution raised an exception
                    raise self.exception
                return TaskManager.VALUE_DONE, self.ret_value
            # the action is still executing
            return TaskManager.VALUE_EXECUTING, None

    def inc_progress(self, delta, in_lock=False):
        if not isinstance(delta, int) or delta < 0:
            raise Exception("Progress delta value must be a positive integer")
        if not in_lock:
            self.lock.acquire()
        prog = self.progress + delta
        self.progress = prog if prog <= 100 else 100
        if not in_lock:
            self.lock.release()

    def set_progress(self, percentage, in_lock=False):
        if not isinstance(percentage, int) or percentage < 0 or percentage > 100:
            raise Exception("Progress value must be in percentage "
                            "(0 <= percentage <= 100)")
        if not in_lock:
            self.lock.acquire()
        self.progress = percentage
        if not in_lock:
            self.lock.release()


def is_valid_ipv6_address(addr):
    try:
        socket.inet_pton(socket.AF_INET6, addr)
        return True
    except socket.error:
        return False


def build_url(host, scheme=None, port=None):
    """
    Build a valid URL. IPv6 addresses specified in host will be enclosed in brackets
    automatically.

    >>> build_url('example.com', 'https', 443)
    'https://example.com:443'

    >>> build_url(host='example.com', port=443)
    '//example.com:443'

    >>> build_url('fce:9af7:a667:7286:4917:b8d3:34df:8373', port=80, scheme='http')
    'http://[fce:9af7:a667:7286:4917:b8d3:34df:8373]:80'

    :param scheme: The scheme, e.g. http, https or ftp.
    :type scheme: str
    :param host: Consisting of either a registered name (including but not limited to
                 a hostname) or an IP address.
    :type host: str
    :type port: int
    :rtype: str
    """
    netloc = host if not is_valid_ipv6_address(host) else '[{}]'.format(host)
    if port:
        netloc += ':{}'.format(port)
    pr = urllib.parse.ParseResult(
        scheme=scheme if scheme else '',
        netloc=netloc,
        path='',
        params='',
        query='',
        fragment='')
    return pr.geturl()


def dict_contains_path(dct, keys):
    """
    Tests whether the keys exist recursively in `dictionary`.

    :type dct: dict
    :type keys: list
    :rtype: bool
    """
    if keys:
        if not isinstance(dct, dict):
            return False
        key = keys.pop(0)
        if key in dct:
            dct = dct[key]
            return dict_contains_path(dct, keys)
        return False
    return True


if sys.version_info > (3, 0):
    wraps = functools.wraps
    _getargspec = inspect.getfullargspec
else:
    def wraps(func):
        def decorator(wrapper):
            new_wrapper = functools.wraps(func)(wrapper)
            new_wrapper.__wrapped__ = func  # set __wrapped__ even for Python 2
            return new_wrapper
        return decorator

    _getargspec = inspect.getargspec


def getargspec(func):
    try:
        while True:
            func = func.__wrapped__
    except AttributeError:
        pass
    return _getargspec(func)
