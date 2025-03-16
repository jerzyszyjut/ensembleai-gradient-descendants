import pandas as pd
import requests

TOKEN = "dJL9uGkRYeY3vlJ0UV4XnpIghehTr3"                        # Your token here
URL = "http://149.156.182.9:6060/task-1/submit"

if __name__ == '__main__':

    result = requests.post(
        URL,
        headers={"token": TOKEN},
        files={
            "csv_file": ("submission.csv", open("./results/submission.csv", "rb"))
        }
    )

    print(result.status_code, result.text)