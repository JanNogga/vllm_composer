// External crates
use actix_web::{HttpRequest, HttpResponse, Responder, web, HttpMessage};
use serde_json::{json, Value};

// Standard library
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

// Internal modules
use crate::auth::AuthInfo;
use crate::state::{AppState, Endpoint};

// -- Handler: /v1/models (combined list from both generate and embed) ----------------
pub async fn models_handler(req: HttpRequest, state: web::Data<Arc<AppState>>) -> impl Responder {
    // Retrieve AuthInfo
    let auth_info = match req.extensions().get::<AuthInfo>() {
        Some(info) => info.clone(),
        None => return HttpResponse::Unauthorized().finish(),
    };
    let user_groups = &auth_info.groups;

    // Lock endpoints for group checks
    let endpoints_generate = state.endpoints_generate.lock().unwrap().clone();
    let endpoints_embed = state.endpoints_embed.lock().unwrap().clone();

    let endpoint_models_generate = state.endpoint_models_generate.lock().unwrap();
    let endpoint_models_embed = state.endpoint_models_embed.lock().unwrap();

    let mut all_models = Vec::new();

    // Combine generate models
    for (endpoint_url, models) in endpoint_models_generate.iter() {
        if let Some(endpoint) = endpoints_generate.iter().find(|ep| ep.url == *endpoint_url) {
            if endpoint.groups.iter().any(|g| user_groups.contains(g)) {
                for model in models {
                    let mut model_with_url = model.clone();
                    if let Value::Object(ref mut map) = model_with_url {
                        map.insert("endpoint_url".to_string(), Value::String(endpoint_url.clone()));
                        map.insert("task".to_string(), Value::String(endpoint.task.clone()));
                    }
                    all_models.push(model_with_url);
                }
            }
        }
    }

    // Combine embed models
    for (endpoint_url, models) in endpoint_models_embed.iter() {
        if let Some(endpoint) = endpoints_embed.iter().find(|ep| ep.url == *endpoint_url) {
            if endpoint.groups.iter().any(|g| user_groups.contains(g)) {
                for model in models {
                    let mut model_with_url = model.clone();
                    if let Value::Object(ref mut map) = model_with_url {
                        map.insert("endpoint_url".to_string(), Value::String(endpoint_url.clone()));
                        map.insert("task".to_string(), Value::String(endpoint.task.clone()));
                    }
                    all_models.push(model_with_url);
                }
            }
        }
    }

    let output = json!({
        "object": "list",
        "data": all_models
    });
    HttpResponse::Ok().json(output)
}

// -- Handler: /model-to-endpoints (combines generate and embed) --------------------
pub async fn model_to_endpoints_handler(
    req: HttpRequest,
    state: web::Data<Arc<AppState>>,
) -> impl Responder {
    let auth_info = match req.extensions().get::<AuthInfo>() {
        Some(info) => info.clone(),
        None => return HttpResponse::Unauthorized().finish(),
    };
    let user_groups = &auth_info.groups;

    // Build separate endpoint maps for generate and embed
    let endpoints_generate = state.endpoints_generate.lock().unwrap().clone();
    let endpoints_embed = state.endpoints_embed.lock().unwrap().clone();
    let gen_map: HashMap<String, Endpoint> = endpoints_generate
        .into_iter()
        .map(|ep| (ep.url.clone(), ep))
        .collect();
    let emb_map: HashMap<String, Endpoint> = endpoints_embed
        .into_iter()
        .map(|ep| (ep.url.clone(), ep))
        .collect();

    let model_to_endpoints_generate = state.model_to_endpoints_generate.lock().unwrap();
    let model_to_endpoints_embed = state.model_to_endpoints_embed.lock().unwrap();

    // We'll combine them into a single HashMap for the final result
    let mut combined: HashMap<String, HashSet<String>> = HashMap::new();

    // Fill from generate side
    for (model_id, endpoint_list) in model_to_endpoints_generate.iter() {
        for url in endpoint_list {
            if let Some(ep) = gen_map.get(url) {
                if ep.groups.iter().any(|g| user_groups.contains(g)) {
                    combined
                        .entry(model_id.clone())
                        .or_insert_with(HashSet::new)
                        .insert(url.clone());
                }
            }
        }
    }

    // Fill from embed side
    for (model_id, endpoint_list) in model_to_endpoints_embed.iter() {
        for url in endpoint_list {
            if let Some(ep) = emb_map.get(url) {
                if ep.groups.iter().any(|g| user_groups.contains(g)) {
                    combined
                        .entry(model_id.clone())
                        .or_insert_with(HashSet::new)
                        .insert(url.clone());
                }
            }
        }
    }

    // Convert HashSet<String> back to Vec<String>
    let final_map: HashMap<String, Vec<String>> = combined
        .into_iter()
        .map(|(k, v)| (k, v.into_iter().collect()))
        .collect();

    HttpResponse::Ok().json(final_map)
}