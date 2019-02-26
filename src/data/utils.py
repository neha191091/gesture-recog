import urllib.request as urlreq
import zipfile
from pathlib import Path
import rarfile
import os
import src.config as config


def download_dataset():
    if not os.path.exists(config.DataDir):
        os.mkdir(config.DataDir)

        zip_path, _ = urlreq.urlretrieve(config.Url, config.DataZip)

        zip_ref = zipfile.ZipFile(zip_path, 'r')
        zip_ref.extractall(config.DataDir)
        zip_ref.close()

        path_list = Path(config.DataDir).glob('**/*.rar')
        for path in path_list:
            path = str(path)
            print("Extracting: " + path)
            opened_rar = rarfile.RarFile(path)
            rar_dir = path[:-len('.rar')]
            os.mkdir(rar_dir)
            opened_rar.extractall(rar_dir)
            os.remove(path)

        os.remove(zip_path)

    else:
        print('Data already downloaded')