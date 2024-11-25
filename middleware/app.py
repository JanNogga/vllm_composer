import httpx
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from vllmComposer import vllmComposer

# app instantiation to provide routes, implementation is mostly in vllmComposer
def create_app(config_path="config.yml", secrets_path="secrets.yml"):
    app = FastAPI()
    composer = vllmComposer(config_path, secrets_path)
    app.state.composer = composer

    # @asynccontextmanager
    # async def lifespan(app: FastAPI):
    #     # task for periodic refresh of models and metrics
    #     models_task = asyncio.create_task(run_refresh_models())
    #     metrics_task = asyncio.create_task(run_refresh_metrics())
    #     try:
    #         yield
    #     finally:
    #         # on shutdown
    #         models_task.cancel()
    #         metrics_task.cancel()
    #         await asyncio.gather(models_task, metrics_task, return_exceptions=True)

    # async def run_refresh_models():
    #     while True:
    #         await composer.refresh_models()
    #         await asyncio.sleep(1)

    # async def run_refresh_metrics():
    #     while True:
    #         await composer.refresh_metrics()
    #         await asyncio.sleep(0.1)

    executor = ThreadPoolExecutor(max_workers=1)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_event_loop()
        # Schedule tasks in dedicated threads
        models_task = loop.run_in_executor(executor, run_refresh_models_sync)
        metrics_task = loop.run_in_executor(executor, run_refresh_metrics_sync)
        try:
            yield
        finally:
            # Cancel tasks on shutdown
            executor.shutdown(wait=True)

    def run_refresh_models_sync():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_refresh_models())

    def run_refresh_metrics_sync():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_refresh_metrics())

    async def run_refresh_models():
        while True:
            await composer.refresh_models()
            await asyncio.sleep(1)

    async def run_refresh_metrics():
        while True:
            await composer.refresh_metrics()
            await asyncio.sleep(0.1)

    app.router.lifespan_context = lifespan

    @app.get("/health")
    async def health_check():
        """Return health and cache status for all servers."""
        composer = app.state.composer
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
        composer = app.state.composer
        metrics_data = {}
        composer.logger.info("Fetching metrics from all servers...")
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
    
    @app.api_route("/reload", methods=["POST"])
    async def reload_configuration(request: Request):
        """ Reload configuration and secrets. """
        composer = app.state.composer
        # Extract the user's token from headers
        user_token = request.headers.get("Authorization")
        if not user_token or not user_token.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Unauthorized: Missing or invalid token")

        user_token = user_token[len("Bearer "):]

        # Determine the user's group based on the token
        user_group = composer.get_group_for_token(user_token)
        if not user_group:
            raise HTTPException(status_code=403, detail="Forbidden: Invalid token or unauthorized group")
        
        # Check if the user group has admin privileges
        if user_group not in composer.admin_groups:
            raise HTTPException(status_code=403, detail="Forbidden: Insufficient permissions")
        
        composer.logger.info(f"User '{user_group}' authorized to reload configuration.")

        # Reload configuration and secrets
        try:
            composer.load_config()
            composer.load_secrets()
            composer.logger.info("Configuration and secrets reloaded successfully.")
            return JSONResponse(content={"message": "Configuration and secrets reloaded successfully."})
        except Exception as exc:
            composer.logger.error(f"Error reloading configuration or secrets: {exc}")
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {exc}")

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def proxy_request(path: str, request: Request):
        """ Middleware to forward requests to the least loaded server. """
        composer = app.state.composer
        # Extract the user's token from headers
        user_token = request.headers.get("Authorization")
        if not user_token or not user_token.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Unauthorized: Missing or invalid token")

        user_token = user_token[len("Bearer "):]

        # Determine the user's group based on the token
        user_group = composer.get_group_for_token(user_token)
        if not user_group:
            raise HTTPException(status_code=403, detail="Forbidden: Invalid token or unauthorized group")
        composer.logger.info(f"User '{user_group}' authenticated.")
        
        # Restrict to specific routes
        if path not in ["chat/completions", "completions", "models", "embeddings"]:
            raise HTTPException(status_code=404, detail=f"Not Found: The route '{path}' is not supported.")
        
        # Handle /v1/models by aggregating models from all servers
        if path == "models":
            composer.logger.info("Received request for aggregated models from all servers.")
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
        composer.logger.info(f"Least loaded server for model '{target_model_name}': {least_loaded_server}.")
        composer.logger.debug(f" Forwarding request with payload: {payload}")
        
        # Replace user's token with the internal vllm token and sanitize headers
        url = f"{least_loaded_server}/v1/{path}"
        target_accept_encoding = 'gzip'
        if not request.headers.get("accept-encoding") in ["gzip", "gzip, deflate"]:
            if request.headers.get("accept-encoding") is not None:
                composer.logger.warning(f"'accept-encoding' changed to 'gzip' for backend request but was '{request.headers.get('accept-encoding')}'")
            else:
                composer.logger.warning("'accept-encoding' unspecified, setting to 'gzip' for backend request")
        else:
            target_accept_encoding = request.headers.get("accept-encoding")
        headers = {key: value for key, value in request.headers.items() if key.lower() not in ["content-length", "authorization", "api-key", "accept-encoding"]}
        headers["accept-encoding"] = target_accept_encoding
        composer.logger.info(f"Forwarding request to {url} with headers: {headers}")
        headers["Authorization"] = f"Bearer {composer.vllm_token}"

        # Register the time of utilization in composer.servers
        server_idx = [i for i, server in enumerate(composer.servers) if server["url"] == least_loaded_server][0]
        composer.servers[server_idx]["last_utilization"] = datetime.utcnow()

        # Forward the request either in streaming mode or non-streaming mode
        stream_mode = payload.get("stream", False)
        composer.logger.info(f"Streaming mode: {stream_mode}")
        if stream_mode:
            # Handle streaming response - see 'manual streaming mode' in https://www.python-httpx.org/async/
            timeout = httpx.Timeout(
                connect=10.0,
                read=10.0,
                write=5.0,
                pool=5.0
            )
            client = httpx.AsyncClient(timeout=timeout)
            try:
                req = client.build_request(
                            method=request.method,
                            url=url,
                            headers=headers,
                            json=payload if request.method in ["POST", "PUT"] else None,
                            params=request.query_params
                        )
                response = await client.send(req, stream=True)

                # Check for non-success status codes before streaming
                if response.status_code >= 400:
                    content = await response.aread()
                    await response.aclose()
                    return Response(
                        content=content,
                        status_code=response.status_code,
                        headers=dict(response.headers)
                    )

                # Augment aiter_bytes to handle exceptions
                async def safe_stream_generator():
                    try:
                        async for chunk in response.aiter_bytes():
                            yield chunk
                    except Exception as exc:
                        composer.logger.error(f"Error during streaming: {exc}")
                        yield b'event: error\ndata: {"error": "Streaming interrupted"}\n\n'
                    finally:
                        await response.aclose()

                # Stream the response to the client
                return StreamingResponse(
                    safe_stream_generator(),
                    status_code=response.status_code,
                    headers={key: value for key, value in dict(response.headers).items() if key.lower() != "content-length"},
                    background=BackgroundTask(response.aclose)
                )
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"Error during streaming: {exc}")
        else:
            # Handle non-streaming response
            try:
                timeout = httpx.Timeout(
                    connect=5.0,
                    read=60.0,
                    write=5.0,
                    pool=5.0
                )
                client = httpx.AsyncClient(timeout=timeout)
                req = client.build_request(
                    method=request.method,
                    url=url,
                    headers=headers,
                    json=payload if request.method in ["POST", "PUT"] else None,
                    params=request.query_params
                )
                response = await client.send(req)
                composer.logger.debug(f"Backend response headers: {dict(response.headers)}")
                composer.logger.debug(f"Backend response content: {response.content}")
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers)
                )
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"Error communicating with backend server: {exc}")
            finally:
                await client.aclose()

    return app