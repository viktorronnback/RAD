from pexels_api import API
import urllib.request
from PIL import Image 
import requests
from requests.auth import HTTPBasicAuth
import shutil
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv('PEXELS_API_KEY')

GET_PATH = "https://api.pexels.com/v1/photos/"

if __name__ == "__main__":
    # Create API object
    api = API(API_KEY)

    api.search('people laughing', page=3, results_per_page=80)
    photos = api.get_entries()

    for photo in photos:
        print('Photographer: ', photo.photographer)
        print('Photo url: ', photo.url)
        photo_name = photo.url.split("/")[-2]

        print('Photo original size: ', photo.original)

        save_path = f"../input/pexels3/{photo.photographer}-{photo_name}.jpg"
        img_path = GET_PATH + str(photo.id)

        response = requests.get(url=photo.original)

        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
        else:
            print("Status code:", response.status_code)