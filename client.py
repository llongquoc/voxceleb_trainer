import requests
import argparse
import sys

URL = 'https://speaker-recognition-server-hahp6a25fq-as.a.run.app/predict'

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("file_path", type=str)
        args = parser.parse_args()
    except Exception as e:
        print("Invalid arguments. Error: " + str(e))
        parser.print_usage()
        sys.exit(1)
    
    file_path = args.file_path
    
    with open(file_path, 'rb') as file:
        values = {'file': (file_path, file, 'audio/wav')}
        response = requests.post(URL, files=values)
        data = response.json()

        print('Predicted result: {}'.format(data['speaker']))
