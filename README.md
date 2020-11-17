# Research
Repository that includes results from various research threads (prototypes, etc).

## Development

Create a Python environment using either the `environment.yml` file (for Conda) or `requirements.txt`.

Create a file called `.local` in the root of the project (i.e. next to `get_files.py`). The file should contain the 
Minio server hostname and access and secret keys, in the following format:
```
hostname=<your_minio_hostname>
access_key=<your_access_key>
secret_key=<your_secret_key>
```

Fetch the ASKE files from the Minio server. From the root of the project:
```
$ python get_files.py
```
This will create a folder, `aske_files`, in the root of the project. The folder is not under version control.


