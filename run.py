#!/usr/bin/env python

import os
import runpy
import sys

rabbitUri = ''

PS_JOB_NAME = "ps"
WORKER_JOB_NAME = "worker"


# FLAGS and unparsed declared below configure_parse_arguments()


def configure_parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--ps_hosts", type=str, default="",
                        help="Comma-separated list of hostname:port pairs")
    parser.add_argument("--worker_hosts", type=str, default="",
                        help="Comma-separated list of hostname:port pairs")
    parser.add_argument("--job_name", type=str, default="", help="One of 'ps', 'worker'")
    parser.add_argument("--task_index", type=int, default=0, help="Index of task within the job")
    parser.add_argument("--data_set_path", type=str, default="", help="Path of the dataset to load")
    parser.add_argument("--checkpoint_path", type=str, default="",
                        help="Path for saving checkpoints")
    parser.add_argument("--restore_file_path", type=str, default="",
                        help="Path to file containing parameters for model restore")
    parser.add_argument("--experiment_id", type=str, default="", help="ID of experiment")
    parser.add_argument("--rabbit_uri", type=str, default="",
                        help="URI of RabbitMQ server, experiment id must be defined")
    return parser.parse_known_args()


FLAGS, unparsed = configure_parse_arguments()


def read_master():
    if FLAGS.ps_hosts:
        master_node = FLAGS.ps_hosts.split(',')[0]
    else:
        master_node = FLAGS.worker_hosts.split(',')[0]

    master_addrs, master_port = master_node.split(':')
    return master_addrs, master_port


def main(argv):
    master_addrs, master_port = read_master()

    nproc_per_node = 4
    nnodes = 2
    node_rank = FLAGS.task_index

    job_name = FLAGS.job_name
    if job_name == PS_JOB_NAME:
        print('no idea')
    elif job_name == WORKER_JOB_NAME:
        sys.argv[1:] = [
            f'--nproc_per_node={nproc_per_node}',
            f'--nnodes={nnodes}',
            f'--node_rank={node_rank}',
            f'--master_addr="{master_addrs}"',
            f'--master_port={master_port}',
            'train.py',
            'configs/webface_r18.py']
        print(sys)
        runpy.run_module('torch.distributed.launch', run_name='__main__')


def entry(datasets=None, checkpoint_path=None, restore_file_path=None):
    print(f"entry: datasets {datasets} , checkpoint_path {checkpoint_path}")

    FLAGS.ps_hosts = os.environ['PS_HOSTS']
    FLAGS.worker_hosts = os.environ['WORKER_HOSTS']
    FLAGS.job_name = os.environ['JOB_NAME']
    FLAGS.task_index = int(os.environ['TASK_INDEX'])
    if datasets is not None:
        FLAGS.data_set_path = datasets
    if checkpoint_path is not None:
        FLAGS.checkpoint_path = checkpoint_path
    if restore_file_path is not None:
        FLAGS.restore_file_path = restore_file_path

    print(FLAGS)
    main(argv=[sys.argv[0]] + unparsed)


if __name__ == "__main__":
    entry()
