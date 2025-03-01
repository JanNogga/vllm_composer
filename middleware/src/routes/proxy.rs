// External crates
use actix_web::{web, HttpMessage, HttpRequest, HttpResponse, Responder};
use futures_util::{Stream, StreamExt};
use log::info;
use reqwest;
use serde_json::Value;
use bytes::Bytes;
use tokio::time::timeout;
use async_stream::try_stream;


// Standard library
use std::sync::Arc;
use std::io::{Error as IoError, ErrorKind};
use std::time::Duration;

// Internal modules
use crate::auth::AuthInfo;
use crate::state::{AppState, Endpoint};


// Helpers
fn stream_with_read_timeout<S>(
    upstream: S,
) -> impl Stream<Item = Result<Bytes, IoError>>
where
    S: Stream<Item = reqwest::Result<Bytes>> + Unpin,
{
    try_stream! {
        let mut resp_stream = upstream;

        // Loop over each chunk, applying a 30s timeout per chunk
        loop {
            // Wait up to 30s for the next chunk
            let next_chunk = match timeout(Duration::from_secs(30), resp_stream.next()).await {
                Ok(res) => res,                 // We got either Some(...) or None from the stream
                Err(_) => {
                    // Timed out waiting for the chunk
                    Err(IoError::new(ErrorKind::TimedOut, "Read timed out"))?
                }
            };

            match next_chunk {
                Some(Ok(chunk)) => {
                    // Successfully got one chunk
                    yield chunk;
                }
                Some(Err(e)) => {
                    // Convert reqwest error into IoError
                    Err(IoError::new(ErrorKind::Other, e))?;
                }
                None => {
                    // Stream ended
                    break;
                }
            }
        }
    }
}

// -- Handler: /v1/chat/completions (for generate) ----------------------------
pub async fn chat_completions_handler(
    req: HttpRequest,
    state: web::Data<Arc<AppState>>,
    body: web::Json<Value>,
) -> impl Responder {
    // 1. Check auth
    let auth_info = match req.extensions().get::<AuthInfo>() {
        Some(info) => info.clone(),
        None => return HttpResponse::Unauthorized().finish(),
    };
    let user_groups = &auth_info.groups;

    // 2. Extract model
    let model_id = match body.get("model").and_then(Value::as_str) {
        Some(m) => m,
        None => return HttpResponse::NotFound().body("The model `` does not exist."),
    };

    // 3. Check whether user wants streaming
    let stream_requested = body.get("stream").and_then(Value::as_bool).unwrap_or(false);

    // 4. Look in generate's model->endpoints map
    let model_to_endpoints_generate = state.model_to_endpoints_generate.lock().unwrap();
    let endpoints_for_model = match model_to_endpoints_generate.get(model_id) {
        Some(eps) => eps.clone(),
        None => {
            return HttpResponse::NotFound()
                .body(format!("The model `{}` does not exist.", model_id));
        }
    };
    drop(model_to_endpoints_generate);

    // 5. Filter endpoints by group
    let endpoints = state.endpoints_generate.lock().unwrap();
    let endpoints_list = endpoints_for_model
        .iter()
        .filter_map(|url| endpoints.iter().find(|e| &e.url == url))
        .filter(|ep| ep.groups.iter().any(|g| user_groups.contains(g)))
        .cloned()
        .collect::<Vec<Endpoint>>();

    // 6. If no authorized endpoints remain, 404
    if endpoints_list.is_empty() {
        return HttpResponse::NotFound()
            .body(format!("The model `{}` does not exist.", model_id));
    }

    // 7. Pick the first one and rotate
    let target_endpoint = &endpoints_list[0];
    {
        let mut map_lock = state.model_to_endpoints_generate.lock().unwrap();
        if let Some(urls) = map_lock.get_mut(model_id) {
            if let Some(pos) = urls.iter().position(|url| url == &target_endpoint.url) {
                let url = urls.remove(pos);
                urls.push(url);
            }
        }
    }

    // Log the forwarded request details
    if stream_requested {
        info!(
            "forwarded streaming request for model {} to endpoint {}",
            model_id, target_endpoint.url
        );
    } else {
        info!(
            "forwarded request for model {} to endpoint {}",
            model_id, target_endpoint.url
        );
    }

    // 8. Forward the entire request body
    let forward_url = format!("{}/v1/chat/completions", target_endpoint.url);

    // Set up the client
    let client_builder = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5));
    let client = if stream_requested {
        client_builder
            .build()
            .unwrap()
    } else {
        client_builder
            // For non-streaming block for a maximum of 90 seconds.
            .timeout(Duration::from_secs(90))
            .build()    
            .unwrap()
    };
    let forward_resp = client
        .post(forward_url)
        .bearer_auth(&target_endpoint.access_token)
        .json(&*body)
        .send()
        .await;

    // 9. Handle streaming vs non-streaming response
    match forward_resp {
        Ok(resp) => {
            let status = resp.status();
            if stream_requested {
                let content_type = resp
                    .headers()
                    .get(reqwest::header::CONTENT_TYPE)
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("application/octet-stream")
                    .to_string();
                let byte_stream = resp.bytes_stream();
                // Wrap the original stream per-chunk timeout logic
                let timed_stream = stream_with_read_timeout(byte_stream);
                HttpResponse::build(status)
                    .content_type(content_type)
                    // Pass the *new* timed_stream to Actix
                    .streaming(timed_stream)
            } else {
                let text = resp.text().await.unwrap_or_default();
                HttpResponse::build(status)
                    .content_type("application/json")
                    .body(text)
            }
        }
        Err(e) => HttpResponse::InternalServerError().body(format!("Forward request failed: {}", e)),
    }
}


// -- Handler: /v1/embeddings (for embed) -------------------------------              
pub async fn embeddings_handler(
    req: HttpRequest,
    state: web::Data<Arc<AppState>>,
    body: web::Json<Value>,
) -> impl Responder {
    // 1. Check auth
    let auth_info = match req.extensions().get::<AuthInfo>() {
        Some(info) => info.clone(),
        None => return HttpResponse::Unauthorized().finish(),
    };
    let user_groups = &auth_info.groups;

    // 2. Extract model
    let model_id = match body.get("model").and_then(Value::as_str) {
        Some(m) => m,
        None => return HttpResponse::NotFound().body("The model `` does not exist."),
    };

    // 3. Look in embed's model->endpoints map
    let model_to_endpoints_embed = state.model_to_endpoints_embed.lock().unwrap();
    let endpoints_for_model = match model_to_endpoints_embed.get(model_id) {
        Some(eps) => eps.clone(),
        None => {
            return HttpResponse::NotFound()
                .body(format!("The model `{}` does not exist.", model_id));
        }
    };
    drop(model_to_endpoints_embed);

    // 4. Filter endpoints by group
    let endpoints = state.endpoints_embed.lock().unwrap();
    let endpoints_list = endpoints_for_model
        .iter()
        .filter_map(|url| endpoints.iter().find(|e| &e.url == url))
        .filter(|ep| ep.groups.iter().any(|g| user_groups.contains(g)))
        .cloned()
        .collect::<Vec<Endpoint>>();

    // 5. If no authorized endpoints remain, 404
    if endpoints_list.is_empty() {
        return HttpResponse::NotFound()
            .body(format!("The model `{}` does not exist.", model_id));
    }

    // 6. Pick the first one and rotate
    let target_endpoint = &endpoints_list[0];
    {
        let mut map_lock = state.model_to_endpoints_embed.lock().unwrap();
        if let Some(urls) = map_lock.get_mut(model_id) {
            if let Some(pos) = urls.iter().position(|url| url == &target_endpoint.url) {
                let url = urls.remove(pos);
                urls.push(url);
            }
        }
    }

    // 7. Forward the entire request body
    info!(
        "forwarded embed request for model {} to endpoint {}",
        model_id, target_endpoint.url
    );
    let forward_url = format!("{}/v1/embeddings", target_endpoint.url);
    let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(90))
            .build()
            .unwrap();
    let forward_resp = client
        .post(forward_url)
        .bearer_auth(&target_endpoint.access_token)
        .json(&*body)
        .send()
        .await;

    match forward_resp {
        Ok(resp) => {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            HttpResponse::build(status)
                .content_type("application/json")
                .body(text)
        }
        Err(e) => HttpResponse::InternalServerError().body(format!("Forward request failed: {}", e)),
    }
}

// -- Handler: /v1/completions (legacy) ----------------------------
pub async fn chat_completions_handler_legacy(
    req: HttpRequest,
    state: web::Data<Arc<AppState>>,
    body: web::Json<Value>,
) -> impl Responder {
    // 1. Check auth
    let auth_info = match req.extensions().get::<AuthInfo>() {
        Some(info) => info.clone(),
        None => return HttpResponse::Unauthorized().finish(),
    };
    let user_groups = &auth_info.groups;

    // 2. Extract model
    let model_id = match body.get("model").and_then(Value::as_str) {
        Some(m) => m,
        None => return HttpResponse::NotFound().body("The model `` does not exist."),
    };

    // 3. Check whether user wants streaming
    let stream_requested = body.get("stream").and_then(Value::as_bool).unwrap_or(false);

    // 4. Look in generate's model->endpoints map
    let model_to_endpoints_generate = state.model_to_endpoints_generate.lock().unwrap();
    let endpoints_for_model = match model_to_endpoints_generate.get(model_id) {
        Some(eps) => eps.clone(),
        None => {
            return HttpResponse::NotFound()
                .body(format!("The model `{}` does not exist.", model_id));
        }
    };
    drop(model_to_endpoints_generate);

    // 5. Filter endpoints by group
    let endpoints = state.endpoints_generate.lock().unwrap();
    let endpoints_list = endpoints_for_model
        .iter()
        .filter_map(|url| endpoints.iter().find(|e| &e.url == url))
        .filter(|ep| ep.groups.iter().any(|g| user_groups.contains(g)))
        .cloned()
        .collect::<Vec<Endpoint>>();

    // 6. If no authorized endpoints remain, 404
    if endpoints_list.is_empty() {
        return HttpResponse::NotFound()
            .body(format!("The model `{}` does not exist.", model_id));
    }

    // 7. Pick the first one and rotate
    let target_endpoint = &endpoints_list[0];
    {
        let mut map_lock = state.model_to_endpoints_generate.lock().unwrap();
        if let Some(urls) = map_lock.get_mut(model_id) {
            if let Some(pos) = urls.iter().position(|url| url == &target_endpoint.url) {
                let url = urls.remove(pos);
                urls.push(url);
            }
        }
    }

    // Log the forwarded request details
    if stream_requested {
        info!(
            "forwarded streaming request for model {} to endpoint {}",
            model_id, target_endpoint.url
        );
    } else {
        info!(
            "forwarded request for model {} to endpoint {}",
            model_id, target_endpoint.url
        );
    }

    // 8. Forward the entire request body
    let forward_url = format!("{}/v1/completions", target_endpoint.url);

    // Set up the client
    let client_builder = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5));
    let client = if stream_requested {
        client_builder
            .build()
            .unwrap()
    } else {
        client_builder
            // For non-streaming block for a maximum of 90 seconds.
            .timeout(Duration::from_secs(90))
            .build()    
            .unwrap()
    };
    let forward_resp = client
        .post(forward_url)
        .bearer_auth(&target_endpoint.access_token)
        .json(&*body)
        .send()
        .await;

    // 9. Handle streaming vs non-streaming response
    match forward_resp {
        Ok(resp) => {
            let status = resp.status();
            if stream_requested {
                let content_type = resp
                    .headers()
                    .get(reqwest::header::CONTENT_TYPE)
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("application/octet-stream")
                    .to_string();
                let byte_stream = resp.bytes_stream();
                // Wrap the original stream per-chunk timeout logic
                let timed_stream = stream_with_read_timeout(byte_stream);
                HttpResponse::build(status)
                    .content_type(content_type)
                    // Pass the *new* timed_stream to Actix
                    .streaming(timed_stream)
            } else {
                let text = resp.text().await.unwrap_or_default();
                HttpResponse::build(status)
                    .content_type("application/json")
                    .body(text)
            }
        }
        Err(e) => HttpResponse::InternalServerError().body(format!("Forward request failed: {}", e)),
    }
}