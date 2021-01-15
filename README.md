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

For example, a file might look like:
```
hostname=8.8.8.8:9000
access_key=foo
secret_key=bar
```

Fetch the ASKE files from the Minio server. From the root of the project:
```
$ python get_files.py
```
This will create a folder, `aske_files`, in the root of the project. The folder is not under version control.
The local target directory name can be changed by supplying the `--dir` argument. Also, by providing the 
`--prefix` argument, only certain content will be downloaded (e.g. `--prefix research/EPI`). 
NOTE: All the content in the local target directory will be deleted before fetching the files from the storage.
