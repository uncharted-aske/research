import os
import sys
import shutil
from argparse import ArgumentParser

from minio import Minio
from tqdm import tqdm


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--secure', '-s', action="store_true",
                        help='the Minio server connection should be secure')
    parser.add_argument('--prefix', '-p', default="",
                        help='the folder prefix to download (e.g. "research/EPI")')
    parser.add_argument('--dir', '-d', default="aske_files",
                        help='the folder in which the downloaded files should be placed')
    args = parser.parse_args()

    root_dir = args.dir
    target_dir = os.path.join(root_dir, args.prefix)
    # delete the local target directory if it exists, to ensure that the
    #   local contents are in sync with Minio after fetching
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    with open(".local", "rt") as f:

        hostname, access_key, secret_key = [line.strip().split("=", 1)[1] for line in f.readlines()]

        minio_client = Minio(hostname, access_key=access_key, secret_key=secret_key, secure=args.secure)

        if not minio_client.bucket_exists("aske"):
            print("Could not find a bucket called 'aske'")
            sys.exit(1)

        objects = [o for o in minio_client.list_objects("aske", prefix=args.prefix, recursive=True)]

        if not objects:
            print("No objects with prefix '%s' found in 'aske' bucket" % args.prefix)
            sys.exit(1)

        for obj in tqdm(objects):
            directory, filename = obj.object_name.rsplit('/', 1)
            dir_path = os.path.join(root_dir, directory)
            os.makedirs(dir_path, exist_ok=True)
            minio_client.fget_object(bucket_name="aske", object_name=obj.object_name,
                                     file_path=os.path.join(dir_path, filename))
