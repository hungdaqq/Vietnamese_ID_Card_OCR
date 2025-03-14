import requests

url = "https://maintechapi.ahava.com.vn/api/upload"

file_1 = "tmp/030201003452_mattruoc.jpg"
file_2 = "tmp/030201003452_matsau.jpg"

# Open the files in binary mode
with open(file_1, "rb") as f1, open(file_2, "rb") as f2:
    # Create a list of tuples for the files
    files = [
        ("files", (file_1, f1, "image/jpeg")),
        ("files", (file_2, f2, "image/jpeg")),
    ]

    # Send the POST request
    response = requests.post(url, files=files)

# Check the response
print(response.status_code)
print(response.text)
