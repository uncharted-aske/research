# Usage: python put_files.py <path_of_source_directory> <path_of_target_directory>

import os 
import sys
from argparse import ArgumentParser

from minio import Minio
from tqdm import tqdm

if __name__ == '__main__':

    # Arguments
    parser = ArgumentParser()
    parser.add_argument('source', type = str, help = 'relative path of the source directory')
    parser.add_argument('target', type = str, help = 'absolute path of the target directory')
    parser.add_argument('-s', '--secure', action = "store_true", help = 'set the Minio server connection to be secure')
    args = parser.parse_args()

    print(f"\n")

    # File paths
    source_path = args.source
    bucket_name, target_path = args.target.split('/', 1)
    __, __, source_filenames = next(os.walk(source_path))

    with open('.local', 'rt') as f:

        hostname, access_key, secret_key = [line.strip().split(" = ", 1)[1] for line in f.readlines()]
        minio_client = Minio(hostname, access_key = access_key, secret_key = secret_key, secure = args.secure)

        # Check if given bucket exists
        if not minio_client.bucket_exists(bucket_name):
            print(f"Could not find a bucket called '{bucket_name}'.")
            sys.exit(1)

        # Get list of files at target
        target_files = [obj for obj in minio_client.list_objects(bucket_name, prefix = target_path, recursive = True)]
        target_filenames = [os.path.basename(f.object_name) for f in target_files]

        # Upload files
        for filename in tqdm(source_filenames):

            # if filename in target_filenames:

            source = os.path.join(source_path, filename)
            target = os.path.join(target_path, filename)

            minio_client.fput_object(bucket_name = bucket_name, object_name = target, file_path = source)

