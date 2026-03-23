import urllib.request
try:
    req = urllib.request.Request(
        'https://jarvis-ai-api-backend.onrender.com/api/v4/stream', 
        data=b'{"message":"hello"}', 
        headers={'Content-Type':'application/json'}, 
        method='POST'
    )
    urllib.request.urlopen(req)
except Exception as e:
    print(e.read().decode())
