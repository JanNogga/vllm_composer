// External crates
use actix_web::{web, HttpMessage, HttpRequest, HttpResponse, Responder};
use futures_util::TryStreamExt;
use log::info;
use reqwest;
use serde_json::Value;

// Standard library
use std::sync::Arc;

// Internal modules
use crate::auth::AuthInfo;
use crate::state::{AppState, Endpoint};

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
    let client = reqwest::Client::new();
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
                let byte_stream = resp.bytes_stream().map_err(|err| {
                    std::io::Error::new(std::io::ErrorKind::Other, err)
                });
                HttpResponse::build(status)
                    .content_type(content_type)
                    .streaming(byte_stream)
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
    let client = reqwest::Client::new();
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
    let client = reqwest::Client::new();
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
                let byte_stream = resp.bytes_stream().map_err(|err| {
                    std::io::Error::new(std::io::ErrorKind::Other, err)
                });
                HttpResponse::build(status)
                    .content_type(content_type)
                    .streaming(byte_stream)
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