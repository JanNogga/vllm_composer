import httpx
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, Response

from vllmComposer import vllmComposer

# app instantiation to provide routes, implementation is mostly in vllmComposer
def create_app(config_path="config.yml", secrets_path="secrets.yml"):
    app = FastAPI()
    composer = vllmComposer(config_path, secrets_path)
    app.state.composer = composer

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # task for periodic refresh of models and metrics
        models_task = asyncio.create_task(run_refresh_models())
        metrics_task = asyncio.create_task(run_refresh_metrics())
        try:
            yield
        finally:
            # on shutdown
            models_task.cancel()
            metrics_task.cancel()
            await asyncio.gather(models_task, metrics_task, return_exceptions=True)

    async def run_refresh_models():
        while True:
            await composer.refresh_models()
            await asyncio.sleep(1)

    async def run_refresh_metrics():
        while True:
            await composer.refresh_metrics()
            await asyncio.sleep(0.1)

    app = FastAPI(lifespan=lifespan)

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
    
    # helper to fetch metrics asynchronously
    async def fetch_metrics(client, server_url):
        try:
            response = await asyncio.wait_for(client.get(server_url), timeout=2)
            response.raise_for_status()
            return response.text
        except httpx.RequestError as exc:
            return exc
    
    @app.get("/metrics")
    async def get_aggregated_metrics():
        """
        Fetch metrics from all known servers. Return a dictionary with server URLs as keys and raw metrics texts as values.
        """
        metrics_data = {}

        async with httpx.AsyncClient() as client:
            # Create a list of coroutine tasks for fetching metrics from each server
            tasks = []
            for server in composer.servers:
                server_url = f"{server['url']}/metrics"
                tasks.append(fetch_metrics(client, server_url))

            # Run all the tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect results into metrics_data dictionary
            for server, result in zip(composer.servers, results):
                if isinstance(result, Exception):
                    composer.logger.error(f"Error fetching metrics from {server['url']}: {result}")
                    metrics_data[server['url']] = f"Error: {result}"
                else:
                    metrics_data[server['url']] = result

        return JSONResponse(content=metrics_data)
        
    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def proxy_request(path: str, request: Request):
        """ Middleware to forward requests to the least loaded server. """

        # Extract the user's token from headers
        user_token = request.headers.get("Authorization")
        if not user_token or not user_token.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Unauthorized: Missing or invalid token")

        user_token = user_token[len("Bearer "):]

        # Determine the user's group based on the token
        user_group = composer.get_group_for_token(user_token)
        if not user_group:
            raise HTTPException(status_code=403, detail="Forbidden: Invalid token or unauthorized group")
        
        # Restrict to specific routes
        if path not in ["chat/completions", "completions", "models", "embeddings"]:
            raise HTTPException(status_code=404, detail=f"Not Found: The route '{path}' is not supported.")
        
        # Handle /v1/models by aggregating models from all servers
        if path == "models":
            return await composer.handle_models_request(user_group)
        
        # Extract the target model name from the JSON payload
        try:
            payload = await request.json()
            target_model_name = payload.get("model")
            if not target_model_name:
                raise HTTPException(status_code=400, detail="Bad Request: Missing 'model' in payload")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Bad Request: Invalid JSON payload ({str(exc)})")
        composer.logger.info(f"Received request for model '{target_model_name}' from group '{user_group}'.")

        # Find compatible servers for the target model and user group
        compatible_servers = await composer.get_compatible_servers(target_model_name, user_group)
        if not compatible_servers:
            raise HTTPException(status_code=503, detail="Service Unavailable: No compatible servers found")
        
        # Identify the least loaded server among the compatible servers based on vllm metrics (tiebreaker: round robin)
        least_loaded_server = await composer.get_least_utilized_server(compatible_servers)
        if not least_loaded_server:
            raise HTTPException(status_code=503, detail="Service Unavailable: No available servers with sufficient capacity")
        composer.logger.info(f"Least loaded server for model '{target_model_name}': {least_loaded_server}. Forwarding request...")
        
        # Replace user's token with the internal vllm token and forward the request
        url = f"{least_loaded_server}/v1/{path}"
        headers = {key: value for key, value in request.headers.items() if key.lower() != "authorization"}
        headers["Authorization"] = f"Bearer {composer.vllm_token}"

        # Register the time of utilization in composer.servers
        server_idx = [i for i, server in enumerate(composer.servers) if server["url"] == least_loaded_server][0]
        composer.servers[server_idx]["last_utilization"] = datetime.utcnow()

        # Forward the request either in streaming mode or non-streaming mode
        stream_mode = payload.get("stream", False)
        if stream_mode:
            # Handle streaming response
            client = httpx.AsyncClient()
            try:
                response = await client.request(
                    method=request.method,
                    url=url,
                    headers=headers,
                    json=payload if request.method in ["POST", "PUT"] else None,
                    params=request.query_params
                )

                # Check for non-success status codes before streaming
                if response.status_code >= 400:
                    content = await response.aread()
                    await response.aclose()
                    await client.aclose()
                    return Response(
                        content=content,
                        status_code=response.status_code,
                        headers=dict(response.headers)
                    )

                # Stream the response to the client
                async def response_generator():
                    try:
                        async for chunk in response.aiter_bytes():
                            yield chunk
                    finally:
                        await response.aclose()
                        await client.aclose()

                return StreamingResponse(
                    response_generator(),
                    status_code=response.status_code,
                    headers=dict(response.headers)
                )
            except Exception as exc:
                await client.aclose()
                raise HTTPException(status_code=502, detail=f"Error during streaming: {exc}")
        else:
            # Handle non-streaming response
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.request(
                        method=request.method,
                        url=url,
                        headers=headers,
                        json=payload if request.method in ["POST", "PUT"] else None,
                        params=request.query_params
                    )
                    return Response(
                        content=response.content,
                        status_code=response.status_code,
                        headers=dict(response.headers)
                    )
                except httpx.RequestError as exc:
                    raise HTTPException(status_code=502, detail=f"Error communicating with backend server: {exc}")

    return app