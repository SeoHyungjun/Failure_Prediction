# -*- coding: utf-8 -*-
"""
 *   Copyright (c) 2017 SUSE LLC
 *
 *  openATTIC is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; version 2.
 *
 *  This package is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
"""
from __future__ import absolute_import

from .tools import build_url
import inspect
import re
import requests
from requests.exceptions import ConnectionError, InvalidURL
from . import logger

try:
    from requests.packages.urllib3.exceptions import SSLError
except ImportError:
    from urllib3.exceptions import SSLError


class RequestException(Exception):
    def __init__(self,
                 message,
                 status_code=None,
                 content=None,
                 conn_errno=None,
                 conn_strerror=None):
        super(RequestException, self).__init__(message)
        self.status_code = status_code
        self.content = content
        self.conn_errno = conn_errno
        self.conn_strerror = conn_strerror


class BadResponseFormatException(RequestException):
    def __init__(self, message):
        super(BadResponseFormatException, self).__init__(
            "Bad response format" if message is None else message, None)


class _ResponseValidator(object):
    """Simple JSON schema validator

    This class implements a very simple validator for the JSON formatted
    messages received by request responses from a RestClient instance.

    The validator validates the JSON response against a "structure" string that
    specifies the structure that the JSON response must comply.  The validation
    procedure raises a BadResponseFormatException in case of a validation
    failure.

    The structure syntax is given by the following grammar:

    Structure  ::=  Level
    Level      ::=  Path | Path '&' Level
    Path       ::=  Step | Step '>'+ Path
    Step       ::=  Key  | '?' Key | '*' | '(' Level ')'
    Key        ::=  <string> | Array+
    Array      ::=  '[' <int> ']' | '[' '*' ']' | '[' '+' ']'

    The symbols enclosed in ' ' are tokens of the language, and the + symbol
    denotes repetition of of the preceding token at least once.

    Examples of usage:

    Example 1:
        Validator args:
            structure = "return > *"
            response = { 'return': { ... } }

        In the above example the structure will validate against any response
        that contains a key named "return" in the root of the response
        dictionary and its value is also a dictionary.

    Example 2:
        Validator args:
            structure = "[*]"
            response = [...]

        In the above example the structure will validate against any response
        that is an array of any size.

    Example 3:
        Validator args:
            structure = "return[*]"
            response = { 'return': [....] }

        In the above example the structure will validate against any response
        that contains a key named "return" in the root of the response
        dictionary and its value is an array.

    Example 4:
        Validator args:
            structure = "return[0] > token"
            response = { 'return': [ { 'token': .... } ] }

        In the above example the structure will validate against any response
        that contains a key named "return" in the root of the response
        dictionary and its value is an array, and the first element of the
        array is a dictionary that contains the key 'token'.

    Example 5:
        Validator args:
            structure = "return[0][*] > key1"
            response = { 'return': [ [ { 'key1': ... } ], ...] }

        In the above example the structure will validate against any response
        that contains a key named "return" in the root of the response
        dictionary where its value is an array, and the first value of this
        array is also an array where all it's values must be a dictionary
        containing a key named "key1".

    Example 6:
        Validator args:
            structure = "return > (key1[*] & key2 & ?key3 > subkey)"
            response = { 'return': { 'key1': [...], 'key2: .... } ] }

        In the above example the structure will validate against any response
        that contains a key named "return" in the root of the response
        dictionary and its value is a dictionary that must contain a key named
        "key1" that is an array, a key named "key2", and optionaly a key named
        "key3" that is a dictionary that contains a key named "subkey".

    Example 7:
        Validator args:
            structure = "return >> roles[*]"
            response = { 'return': { 'key1': { 'roles': [...] }, 'key2': { 'roles': [...] } } }

        In the above example the structure will validate against any response
        that contains a key named "return" in the root of the response
        dictionary, and its value is a dictionary that for any key present in
        the dictionary their value is also a dictionary that must contain a key
        named 'roles' that is an array.  Please note that you can use any
        number of successive '>' to denote the level in the JSON tree that you
        want to match next step in the path.

    """

    @staticmethod
    def validate(structure, response):
        if structure is None:
            return

        _ResponseValidator._validate_level(structure, response)

    @staticmethod
    def _validate_level(level, resp):
        if not isinstance(resp, dict) and not isinstance(resp, list):
            raise BadResponseFormatException(
                "{} is neither a dict nor a list".format(resp))

        paths = _ResponseValidator._parse_level_paths(level)
        for path in paths:
            path_sep = path.find('>')
            if path_sep != -1:
                level_next = path[path_sep + 1:].strip()
            else:
                path_sep = len(path)
                level_next = None
            key = path[:path_sep].strip()

            if key == '*':
                continue
            elif key == '':  # check all keys
                for k in resp.keys():
                    _ResponseValidator._validate_key(k, level_next, resp)
            else:
                _ResponseValidator._validate_key(key, level_next, resp)

    @staticmethod
    def _validate_array(array_seq, level_next, resp):
        if array_seq:
            if not isinstance(resp, list):
                raise BadResponseFormatException(
                    "{} is not an array".format(resp))
            if array_seq[0].isdigit():
                idx = int(array_seq[0])
                if len(resp) <= idx:
                    raise BadResponseFormatException(
                        "length of array {} is lower than the index {}".format(
                            resp, idx))
                _ResponseValidator._validate_array(array_seq[1:], level_next,
                                                   resp[idx])
            elif array_seq[0] == '*':
                for r in resp:
                    _ResponseValidator._validate_array(array_seq[1:],
                                                       level_next, r)
            elif array_seq[0] == '+':
                if len(resp) < 1:
                    raise BadResponseFormatException(
                        "array should not be empty")
                for r in resp:
                    _ResponseValidator._validate_array(array_seq[1:],
                                                       level_next, r)
            else:
                raise Exception(
                    "Response structure is invalid: only <int> | '*' are "
                    "allowed as array index arguments")
        else:
            if level_next:
                _ResponseValidator._validate_level(level_next, resp)

    @staticmethod
    def _validate_key(key, level_next, resp):
        array_access = [a.strip() for a in key.split("[")]
        key = array_access[0]
        if key:
            optional = key[0] == '?'
            if optional:
                key = key[1:]
            if key not in resp:
                if optional:
                    return
                raise BadResponseFormatException(
                    "key {} is not in dict {}".format(key, resp))
            resp_next = resp[key]
        else:
            resp_next = resp
        if len(array_access) > 1:
            _ResponseValidator._validate_array(
                [a[0:-1] for a in array_access[1:]], level_next, resp_next)
        else:
            if level_next:
                _ResponseValidator._validate_level(level_next, resp_next)

    @staticmethod
    def _parse_level_paths(level):
        level = level.strip()
        if level[0] == '(':
            level = level[1:]
            if level[-1] == ')':
                level = level[:-1]

        paths = []
        lp = 0
        nested = 0
        for i, c in enumerate(level):
            if c == '&' and nested == 0:
                paths.append(level[lp:i].strip())
                lp = i + 1
            elif c == '(':
                nested += 1
            elif c == ')':
                nested -= 1
        paths.append(level[lp:].strip())
        return paths


class _Request(object):
    def __init__(self, method, path, path_params, rest_client, resp_structure):
        self.method = method
        self.path = path
        self.path_params = path_params
        self.rest_client = rest_client
        self.resp_structure = resp_structure

    def _gen_path(self):
        new_path = self.path
        matches = re.finditer(r'\{(\w+?)\}', self.path)
        for match in matches:
            if match:
                param_key = match.group(1)
                if param_key in self.path_params:
                    new_path = new_path.replace(
                        match.group(0), self.path_params[param_key])
                else:
                    raise RequestException(
                        'Invalid path. Param "{}" was not specified'
                        .format(param_key), None)
        return new_path

    def __call__(self,
                 req_data=None,
                 method=None,
                 params=None,
                 data=None,
                 raw_content=False):
        method = method if method else self.method
        if not method:
            raise Exception('No HTTP request method specified')
        if req_data:
            if method == 'get':
                if params:
                    raise Exception('Ambiguous source of GET params')
                params = req_data
            else:
                if data:
                    raise Exception('Ambiguous source of {} data'.format(
                        method.upper()))
                data = req_data
        resp = self.rest_client.do_request(method, self._gen_path(), params,
                                           data, raw_content)
        if raw_content and self.resp_structure:
            raise Exception("Cannot validate reponse in raw format")
        _ResponseValidator.validate(self.resp_structure, resp)
        return resp


class RestClient(object):
    def __init__(self, host, port, client_name=None, ssl=False, auth=None):
        super(RestClient, self).__init__()
        self.client_name = client_name if client_name else ''
        self.host = host
        self.port = port
        self.base_url = build_url(
            scheme='https' if ssl else 'http', host=host, port=port)
        logger.debug("REST service base URL: %s", self.base_url)
        self.headers = {'Accept': 'application/json'}
        self.auth = auth
        self.session = requests.Session()

    def _login(self, request=None):
        pass

    def _is_logged_in(self):
        pass

    def _reset_login(self):
        pass

    def is_service_online(self, request=None):
        pass

    @staticmethod
    def requires_login(func):
        def func_wrapper(self, *args, **kwargs):
            retries = 2
            while True:
                try:
                    if not self._is_logged_in():
                        self._login()
                    resp = func(self, *args, **kwargs)
                    return resp
                except RequestException as e:
                    if isinstance(e, BadResponseFormatException):
                        raise e
                    retries -= 1
                    if e.status_code not in [401, 403] or retries == 0:
                        raise e
                    self._reset_login()

        return func_wrapper

    def do_request(self,
                   method,
                   path,
                   params=None,
                   data=None,
                   raw_content=False):
        url = '{}{}'.format(self.base_url, path)
        logger.debug('%s REST API %s req: %s data: %s', self.client_name,
                     method.upper(), path, data)
        try:
            if method.lower() == 'get':
                resp = self.session.get(
                    url, headers=self.headers, params=params, auth=self.auth)
            elif method.lower() == 'post':
                resp = self.session.post(
                    url,
                    headers=self.headers,
                    params=params,
                    data=data,
                    auth=self.auth)
            elif method.lower() == 'put':
                resp = self.session.put(
                    url,
                    headers=self.headers,
                    params=params,
                    data=data,
                    auth=self.auth)
            elif method.lower() == 'delete':
                resp = self.session.delete(
                    url,
                    headers=self.headers,
                    params=params,
                    data=data,
                    auth=self.auth)
            else:
                raise RequestException('Method "{}" not supported'.format(
                    method.upper()), None)
            if resp.ok:
                logger.debug("%s REST API %s res status: %s content: %s",
                             self.client_name, method.upper(),
                             resp.status_code, resp.text)
                if raw_content:
                    return resp.content
                try:
                    return resp.json() if resp.text else None
                except ValueError:
                    logger.error(
                        "%s REST API failed %s req while decoding JSON "
                        "response : %s",
                        self.client_name, method.upper(), resp.text)
                    raise RequestException(
                        "{} REST API failed request while decoding JSON "
                        "response: {}".format(self.client_name, resp.text),
                        resp.status_code, resp.text)
            else:
                logger.error(
                    "%s REST API failed %s req status: %s", self.client_name,
                    method.upper(), resp.status_code)
                from pprint import pprint as pp
                from pprint import pformat as pf

                raise RequestException(
                    "{} REST API failed request with status code {}\n"
                    "{}"  # TODO remove
                    .format(self.client_name, resp.status_code, pf(
                        resp.content)),
                    resp.status_code,
                    resp.content)
        except ConnectionError as ex:
            if ex.args:
                if isinstance(ex.args[0], SSLError):
                    errno = "n/a"
                    strerror = "SSL error. Probably trying to access a non " \
                               "SSL connection."
                    logger.error("%s REST API failed %s, SSL error.",
                                 self.client_name, method.upper())
                else:
                    match = re.match(r'.*: \[Errno (-?\d+)\] (.+)',
                                     ex.args[0].reason.args[0])
                    if match:
                        errno = match.group(1)
                        strerror = match.group(2)
                        logger.error(
                            "%s REST API failed %s, connection error: "
                            "[errno: %s] %s",
                            self.client_name, method.upper(), errno, strerror)
                    else:
                        errno = "n/a"
                        strerror = "n/a"
                        logger.error(
                            "%s REST API failed %s, connection error.",
                            self.client_name, method.upper())
            else:
                errno = "n/a"
                strerror = "n/a"
                logger.error("%s REST API failed %s, connection error.",
                             self.client_name, method.upper())

            if errno != "n/a":
                ex_msg = (
                    "{} REST API cannot be reached: {} [errno {}]. "
                    "Please check your configuration and that the API endpoint"
                    " is accessible"
                    .format(self.client_name, strerror, errno))
            else:
                ex_msg = (
                    "{} REST API cannot be reached. Please check "
                    "your configuration and that the API endpoint is"
                    " accessible"
                    .format(self.client_name))
            raise RequestException(
                ex_msg, conn_errno=errno, conn_strerror=strerror)
        except InvalidURL as ex:
            logger.exception("%s REST API failed %s: %s", self.client_name,
                             method.upper(), str(ex))
            raise RequestException(str(ex))

    @staticmethod
    def api(path, **api_kwargs):
        def call_decorator(func):
            def func_wrapper(self, *args, **kwargs):
                method = api_kwargs.get('method', None)
                resp_structure = api_kwargs.get('resp_structure', None)
                args_name = inspect.getargspec(func).args
                args_dict = dict(zip(args_name[1:], args))
                for key, val in kwargs:
                    args_dict[key] = val
                return func(
                    self,
                    *args,
                    request=_Request(method, path, args_dict, self,
                                     resp_structure),
                    **kwargs)

            return func_wrapper

        return call_decorator

    @staticmethod
    def api_get(path, resp_structure=None):
        return RestClient.api(
            path, method='get', resp_structure=resp_structure)

    @staticmethod
    def api_post(path, resp_structure=None):
        return RestClient.api(
            path, method='post', resp_structure=resp_structure)

    @staticmethod
    def api_put(path, resp_structure=None):
        return RestClient.api(
            path, method='put', resp_structure=resp_structure)

    @staticmethod
    def api_delete(path, resp_structure=None):
        return RestClient.api(
            path, method='delete', resp_structure=resp_structure)
