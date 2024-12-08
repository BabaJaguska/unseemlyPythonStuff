import requests

proxies = {
    "http": "http://13.233.147.46:8080",
    "https": "http://13.233.147.46:8080"
}

try:
    response = requests.get("https://httpbin.org/ip", proxies=proxies, timeout=5)
    print("Proxy response:", response.text)
except Exception as e:
    print("Proxy error:", e)
