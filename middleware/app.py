# aggregate /v1/metrics and /v1/models, simplify /models
# validate stud / staff / guest / admin / legacy tokens and replace by vllm token
# select upstreams based on model
# perform load balancing based on model and upstream metrics

from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# List of backend servers
backend_servers = ["http://bigcuda4.informatik.uni-bonn.de:8000", "http://bigcuda4.informatik.uni-bonn.de:8001"]

def get_least_utilized_server():
    min_utilization = 100
    best_server = None

    for server in backend_servers:
        try:
            response = requests.get(f"{server}/gpu_utilization", timeout=1)
            utilization = response.json().get("gpu_utilization", 100)
            if utilization < min_utilization:
                min_utilization = utilization
                best_server = server
        except Exception:
            continue

    return best_server

@app.route('/v1/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy_request(path):
    # Determine the best server
    best_server = get_least_utilized_server()
    if not best_server:
        return jsonify({"error": "No available backend servers"}), 503

    # Forward the request
    response = requests.request(
        method=request.method,
        url=f"{best_server}/v1/{path}",
        headers={key: value for key, value in request.headers if key != 'Host'},
        json=request.get_json(),
        params=request.args
    )

    return (response.content, response.status_code, response.headers.items())

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9000)
