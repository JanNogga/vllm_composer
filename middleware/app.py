import re
import httpx
from collections import defaultdict
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from vllm_composer import vllmComposer


# FastAPI app instantiation
app = FastAPI()
composer = vllmComposer("config.yml", "secrets.yml")

@app.get("/health")
async def health_check():
    """Return health and cache status for all servers."""
    health_data = []
    for server in composer.servers:
        server_url = server["url"]
        health_data.append({
            "url": server_url,
            "healthy": composer.server_health.get(server_url, {}).get("healthy", True),
            "metrics_cached": composer.metrics_cache.get(server_url),
            "model_cached": composer.model_cache.get(server_url)
        })
    return JSONResponse(content={"servers": health_data})


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_request(path: str, request: Request):
    # Step 1: Extract the user's token from headers
    user_token = request.headers.get("Authorization")
    if not user_token or not user_token.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized: Missing or invalid token")

    user_token = user_token[len("Bearer "):]

    # Step 2: Determine the user's group
    user_group = composer.get_group_for_token(user_token)
    if not user_group:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid token or unauthorized group")
    
    # Restrict to specific routes
    if path not in ["chat/completions", "completions", "models", "embeddings"]:
        raise HTTPException(status_code=404, detail=f"Not Found: The route '{path}' is not supported.")
    
    # Handle /v1/models differently
    if path == "models":
        return await composer.handle_models_request(user_group)
    
    # Step 3: Extract the target model name from the JSON payload
    try:
        payload = await request.json()
        target_model_name = payload.get("model")
        if not target_model_name:
            raise HTTPException(status_code=400, detail="Bad Request: Missing 'model' in payload")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Bad Request: Invalid JSON payload ({str(exc)})")
    composer.logger.info(f"Received request for model '{target_model_name}' from group '{user_group}'.")

    # Step 4: Find compatible servers
    compatible_servers = await composer.get_compatible_servers(target_model_name, user_group)
    if not compatible_servers:
        raise HTTPException(status_code=503, detail="Service Unavailable: No compatible servers found")
    
    # Step 5: Find the least utilized server
    least_loaded_server = await composer.get_least_utilized_server(compatible_servers)
    if not least_loaded_server:
        raise HTTPException(status_code=503, detail="Service Unavailable: No available servers with sufficient capacity")
    composer.logger.info(f"Least loaded server for model '{target_model_name}': {least_loaded_server}. Forwarding request...")
    
    # Step 6: Replace user's token with self.vllm_token and forward the request
    url = f"{least_loaded_server}/v1/{path}"
    headers = {key: value for key, value in request.headers.items() if key.lower() != "authorization"}
    headers["Authorization"] = f"Bearer {composer.vllm_token}"

    stream_mode = payload.get("stream", False)

    async with httpx.AsyncClient() as client:
        try:
            if stream_mode:
                # Handle streaming response
                response = await client.stream(
                    method=request.method,
                    url=url,
                    headers=headers,
                    json=payload if request.method in ["POST", "PUT"] else None,
                    params=request.query_params
                )

                async def stream_response():
                    async for chunk in response.aiter_bytes():
                        yield chunk

                return StreamingResponse(
                    stream_response(),
                    status_code=response.status_code,
                    headers=dict(response.headers)
                )
            else:
                # Handle non-streaming response
                response = await client.request(
                    method=request.method,
                    url=url,
                    headers=headers,
                    json=payload if request.method in ["POST", "PUT"] else None,
                    params=request.query_params
                )
                return JSONResponse(
                    content=response.json(),
                    status_code=response.status_code,
                    headers=dict(response.headers)
                )
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Error communicating with backend server: {exc}")


def parse_metrics(raw_metrics: str, aggregated_metrics: dict):
    """
    Parse raw Prometheus-style metrics and aggregate them.

    Args:
        raw_metrics (str): Raw metrics data from a server.
        aggregated_metrics (dict): Dictionary to store aggregated metrics.
    """
    metrics_pattern = re.compile(r'(\w+:\w+)\{(.*?)\}\s+([\d\.\-eE]+)')
    label_pattern = re.compile(r'(\w+)="(.*?)"')
    metrics = defaultdict(list)

    # Match Prometheus-style metrics with labels
    for match in metrics_pattern.finditer(raw_metrics):
        metric_name = match.group(1)  # Metric name (e.g., "vllm:lora_requests_info")
        labels_raw = match.group(2)  # Raw labels string (e.g., 'max_lora="0",running_lora_adapters=""')
        metric_value = float(match.group(3))  # Metric value

        # Parse the labels into a dictionary
        labels = {m.group(1): m.group(2) for m in label_pattern.finditer(labels_raw)}

        # Use a tuple of the metric name and sorted labels as the aggregation key
        aggregation_key = (metric_name, tuple(sorted(labels.items())))

        metrics[aggregation_key].append(metric_value)

    # Aggregate metrics (e.g., sum for counters, average for gauges)
    for (metric_name, labels), values in metrics.items():
        if "counter" in metric_name:  # For counters, take the sum
            aggregated_value = sum(values)
        else:  # For gauges, take the average
            aggregated_value = sum(values) / len(values) if values else 0

        # Store the aggregated metric in the result, including its labels
        if metric_name not in aggregated_metrics:
            aggregated_metrics[metric_name] = []
        
        aggregated_metrics[metric_name].append({
            "labels": dict(labels),
            "value": aggregated_value
        })

@app.get("/metrics")
async def get_aggregated_metrics():
    """
    Aggregate metrics from all known servers and return the results.
    """

    metrics_data = {}

    async with httpx.AsyncClient() as client:
        for server in composer.servers:
            server_url = f"{server['url']}/metrics"
            try:
                response = await client.get(server_url)
                response.raise_for_status()
                # Parse the metrics data
                raw_metrics = response.text
                parse_metrics(raw_metrics, metrics_data)
            except httpx.RequestError as exc:
                composer.logger.error(f"Error fetching metrics from {server_url}: {exc}")
                continue

    return JSONResponse(content=metrics_data)