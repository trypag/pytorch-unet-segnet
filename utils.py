import requests

from tqdm import tqdm


def download_file(url, filename):
    chunkSize = 1024
    r = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        pbar = tqdm(unit="B", total=int(r.headers['Content-Length']))
        for chunk in r.iter_content(chunk_size=chunkSize):
            if chunk:
                pbar.update(len(chunk))
                f.write(chunk)
    return filename
