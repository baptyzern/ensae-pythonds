import requests


def download_file(url, filename):
    print(f"Téléchargement de {filename}")
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)
