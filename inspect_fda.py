import requests

headers = {"User-Agent": "Nurudeen Aminu nurudeen.aminu41@gmail.com"}

# Test the correct FDA guidance search page
url = "https://www.fda.gov/api/guidance-documents/search"
params = {"q": "drug safety", "pageSize": 5}
r = requests.get(url, params=params, headers=headers, timeout=30)
print("Status:", r.status_code)
print("URL hit:", r.url)
if r.status_code == 200:
    try:
        print(r.json())
    except Exception:
        print(r.text[:2000])