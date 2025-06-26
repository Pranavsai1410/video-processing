# decode_image.py
import json
import base64
import os

# Create search_results folder
output_dir = "search_results"
os.makedirs(output_dir, exist_ok=True)

# Check if response.json is empty
if not os.path.exists('response.json') or os.path.getsize('response.json') == 0:
    print("Error: response.json is empty or missing")
    exit(1)

with open('response.json', 'r') as f:
    data = json.load(f)

for i, result in enumerate(data['results']):
    base64_string = result['image_data'].split(',')[1]  # Remove "data:image/jpeg;base64,"
    img_data = base64.b64decode(base64_string)
    output_path = os.path.join(output_dir, f'result_{i}.jpg')
    with open(output_path, 'wb') as f:
        f.write(img_data)
    print(f"Saved {output_path}")