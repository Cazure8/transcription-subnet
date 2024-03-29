import requests
import tarfile
import os
from tqdm import tqdm

def download_file(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    local_filename = url.split('/')[-1]
    full_local_path = os.path.join(dest_folder, local_filename)

    # Check if the file already exists
    if os.path.exists(full_local_path):
        print(f"File {local_filename} already exists, skipping download.")
        return full_local_path

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_length = int(r.headers.get('content-length'))
        with open(full_local_path, 'wb') as f, tqdm(
            desc=local_filename,
            total=total_length,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024):
                size = f.write(chunk)
                bar.update(size)
    return full_local_path

def extract_tarfile(file_path, dest_folder):
    if not file_path.endswith((".tar.gz", ".tar")):
        print(f"File {file_path} is not a tar archive.")
        return

    # The name of the folder we expect to see after extraction
    expected_extracted_folder = os.path.join(dest_folder, 'LibriSpeech', os.path.basename(file_path).replace(".tar.gz", ""))

    if os.path.exists(expected_extracted_folder):
        print(f"Archive {file_path} already extracted in {expected_extracted_folder}.")
        return
    
    with tarfile.open(file_path, "r:*") as tar:
        tar.extractall(path=dest_folder)
        print(f"Extracted {file_path} to {dest_folder}")


def download_librispeech_subset(subset, base_url, dest_folder):
    subset_url = base_url + f"{subset}.tar.gz"
    tar_file = download_file(subset_url, dest_folder)
    extract_tarfile(tar_file, dest_folder)
    print(f"Downloaded and extracted {subset}")

def download_entire_librispeech():
    subsets = [
        'train-clean-100', 'train-clean-360', 'train-other-500',
        'dev-clean', 'dev-other', 'test-clean', 'test-other'
    ]
    base_url = "http://www.openslr.org/resources/12/"
    dest_folder = 'librispeech_dataset'

    for subset in subsets:
        download_librispeech_subset(subset, base_url, dest_folder)

if __name__ == "__main__":
    download_entire_librispeech()
