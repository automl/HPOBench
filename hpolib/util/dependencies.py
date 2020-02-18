import re

import pkg_resources
from distutils.version import LooseVersion

RE_PATTERN = re.compile(
    r'^(?P<name>[\w\-]+)((?P<operation>==|>=|>)'
    r'(?P<version>(\d+)?(\.[a-zA-Z0-9]+)?(\.\d+)?))?$')

"""Generic code to verify package versions (a.k.a. dependencies).

Written by Anatolii Domashnev (@ayaro) for auto-sklearn. Licensed under a BSD
3-clause license.

See the following link for the original PR:
https://github.com/Ayaro/auto-sklearn/commit/
f59c6e9751061ec0e68a402507c83f6c10ae5bbd
"""


def verify_packages(packages):
    if not packages:
        return
    if isinstance(packages, str):
        packages = packages.splitlines()

    for package in packages:
        if not package:
            continue

        match = RE_PATTERN.match(package)
        if match:
            name = match.group('name')
            operation = match.group('operation')
            version = match.group('version')
            _verify_package(name, operation, version)
        else:
            raise ValueError('Unable to read requirement: %s' % package)


def _verify_package(name, operation, version):
    try:
        module = pkg_resources.get_distribution(name)
    except pkg_resources.DistributionNotFound:
        raise MissingPackageError(name)

    if not operation:
        return

    required_version = LooseVersion(version)
    installed_version = LooseVersion(module.version)

    if operation == '==':
        check = required_version == installed_version
    elif operation == '>':
        check = installed_version > required_version
    elif operation == '>=':
        check = installed_version > required_version or \
                installed_version == required_version
    else:
        raise NotImplementedError(
            'operation \'%s\' is not supported' % operation)
    if not check:
        raise IncorrectPackageVersionError(name, installed_version, operation,
                                           required_version)


class MissingPackageError(Exception):
    error_message = 'mandatory package \'{name}\' not found'

    def __init__(self, package_name):
        self.package_name = package_name
        super(MissingPackageError, self).__init__(
            self.error_message.format(name=package_name))


class IncorrectPackageVersionError(Exception):
    error_message = '\'{name} {installed_version}\' version mismatch ' \
                    '({operation}{required_version})'

    def __init__(self, package_name, installed_version, operation,
                 required_version):
        self.package_name = package_name
        self.installed_version = installed_version
        self.operation = operation
        self.required_version = required_version
        message = self.error_message.format(
            name=package_name,
            installed_version=installed_version,
            operation=operation,
            required_version=required_version)
        super(IncorrectPackageVersionError, self).__init__(message)
