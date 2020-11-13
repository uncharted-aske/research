import os
import sys
from argparse import ArgumentParser

from minio import Minio
from tqdm import tqdm

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--secure', '-s', action="store_true",
                        help='the Minio server connection should be secure')
    args = parser.parse_args()

    with open(".local", "rt") as f:

        hostname, access_key, secret_key = [line.strip().split("=")[1] for line in f.readlines()]

        minio_client = Minio(hostname, access_key=access_key, secret_key=secret_key, secure=args.secure)

        if not minio_client.bucket_exists("aske"):
            print("Could not find a bucket called 'aske'")
            sys.exit(1)

        objects = [o for o in minio_client.list_objects("aske", recursive=True)]

        for obj in tqdm(objects):
            directory, filename = obj.object_name.rsplit('/', 1)
            dir_path = os.path.join("aske_files", directory)
            os.makedirs(dir_path, exist_ok=True)
            minio_client.fget_object(bucket_name="aske", object_name=obj.object_name,
                                     file_path=os.path.join(dir_path, filename))
