from hpobench import config_file

import shutil
import logging
logger = logging.getLogger('Clean-up')
logger.setLevel(logging.INFO)


def _ask_for_del(directory, name):
    logger.info(f'Going to remove the {name} directory {directory}')
    inp = input('Do you want to proceed? [N|y] ')
    if inp in ['y', 'j', 'Y']:
        shutil.rmtree(directory)
    logger.info(f'Successfully removed the {name} directory.')


def delete_container():
    _ask_for_del(config_file.container_dir, 'container')


def clear_socket_dir():
    _ask_for_del(config_file.socket_dir, 'socket')


def clear_cache():
    _ask_for_del(config_file.cache_dir, 'cache')


def clear_data_dir():
    _ask_for_del(config_file.data_dir, 'data')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear_all", help="Remove containers, clear socket, data, and cache directory",
                        action="store_true")
    parser.add_argument("--clear_container", help="Delete the HPOBench container", action="store_true")
    parser.add_argument("--clear_cache", help="Delete the HPOBench cache", action="store_true")
    parser.add_argument("--clear_data", help="Delete the HPOBench data", action="store_true")
    parser.add_argument("--clear_socket", help="Delete the HPOBench socket", action="store_true")
    args = parser.parse_args()

    if args.clear_all or args.clear_container:
        delete_container()
    if args.clear_all or args.clear_cache:
        clear_cache()
    if args.clear_all or args.clear_data:
        clear_data_dir()
    if args.clear_all or args.clear_socket:
        clear_socket_dir()
