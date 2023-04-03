import time
import zipfile
import os
from tqdm import tqdm
from collections import defaultdict


def RestoreTimestampsOfZipContents(zipname, extract_dir):
    # https://stackoverflow.com/questions/9813243/extract-files-from-zip-file-and-retain-mod-date
    # Restores the timestamps of zipfile contents.

    for f in zipfile.ZipFile(zipname, 'r').infolist():
        # path to this extracted f-item
        fullpath = os.path.join(extract_dir, f.filename)
        # still need to adjust the dt o/w item will have the current dt
        date_time = time.mktime(f.date_time + (0, 0, -1))
        # update dt
        os.utime(fullpath, (date_time, date_time))


if __name__ == "__main__":
    """
    Unzip latest files
    """
    # Change per run: 
    root_ann_folder = r"D:\61 Challenges\2022_CASE_\WebAnno\reviewing_annotations\Subtask2\15. Round13"
    
    # Do not touch the remaining:
    folder_names = ["annotation","curation"]
    for d in folder_names:

        folder = os.path.join(root_ann_folder, d)
        subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]

        pbar = tqdm(subfolders)
        for sub in pbar:
            pbar.set_description("Processing %s" % sub)

            dir_name = os.path.join(folder, sub)
            dict_of_files = defaultdict(list)

            # Find latest ZIP files
            for file in os.listdir(dir_name):
                if file.endswith(".zip"):
                    file_name = os.path.join(dir_name, file)
                    timestamp = os.path.getctime(file_name)
                    timestamp = int(timestamp//60 * 60)
                    dict_of_files[timestamp].append(file_name)
            if len(dict_of_files)==0:
                continue
        
            dict_of_files = dict(dict_of_files)
            most_recent_timestamp = max(list(dict_of_files.keys()))
            list_of_files = dict_of_files[most_recent_timestamp]

            # Unzip files
            for file in list_of_files:
                with zipfile.ZipFile(file, 'r') as zip_ref:
                    zip_ref.extractall(dir_name)
                RestoreTimestampsOfZipContents(file, dir_name)
