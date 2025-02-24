// External crates
use actix_web::{web, HttpMessage, HttpRequest, HttpResponse, Responder};

// Standard library
use std::collections::HashMap;
use std::sync::Arc;

// Internal modules
use crate::auth::AuthInfo;
use crate::state::{
    AppState,
    load_endpoints_from_yaml,
    load_auth_tokens_from_yaml,
    partition_endpoints,
};
use crate::monitoring::monitor_endpoint;

// -----------------------------------------------------------------------------
// Handlers
// -----------------------------------------------------------------------------

// -- Handler: /endpoints (returns both generate and embed) --------------------
pub async fn endpoints_handler(req: HttpRequest, state: web::Data<Arc<AppState>>) -> impl Responder {
    let auth_info = match req.extensions().get::<AuthInfo>() {
        Some(info) => info.clone(),
        None => return HttpResponse::Unauthorized().finish(),
    };
    let user_groups = &auth_info.groups;

    let endpoints_generate = state.endpoints_generate.lock().unwrap().clone();
    let endpoints_embed = state.endpoints_embed.lock().unwrap().clone();

    let filtered_endpoints: Vec<serde_json::Value> = endpoints_generate
        .into_iter()
        .chain(endpoints_embed.into_iter())
        .filter(|ep| ep.groups.iter().any(|g| user_groups.contains(g)))
        .map(|ep| {
            // Convert to JSON, remove the "access_tokens" field, and return the modified JSON.
            let mut value = serde_json::to_value(ep).unwrap();
            if let serde_json::Value::Object(ref mut map) = value {
                map.remove("access_token");
            }
            value
        })
        .collect();

    HttpResponse::Ok().json(filtered_endpoints)
}

// -- Handler: /health-status (returns combined health) ------------------------
pub async fn health_status_handler(req: HttpRequest, state: web::Data<Arc<AppState>>) -> impl Responder {
    let auth_info = match req.extensions().get::<AuthInfo>() {
        Some(info) => info.clone(),
        None => return HttpResponse::Unauthorized().finish(),
    };
    let user_groups = &auth_info.groups;

    // Lock endpoints
    let endpoints_generate = state.endpoints_generate.lock().unwrap().clone();
    let endpoints_embed = state.endpoints_embed.lock().unwrap().clone();

    let health_status_generate = state.health_status_generate.lock().unwrap();
    let health_status_embed = state.health_status_embed.lock().unwrap();

    let mut combined_status = HashMap::new();

    // Process generate-based endpoints
    for endpoint in endpoints_generate {
        if endpoint.groups.iter().any(|g| user_groups.contains(g)) {
            if let Some(hs) = health_status_generate.get(&endpoint.url) {
                combined_status.insert(endpoint.url.clone(), hs);
            }
        }
    }

    // Process embed-based endpoints
    for endpoint in endpoints_embed {
        if endpoint.groups.iter().any(|g| user_groups.contains(g)) {
            if let Some(hs) = health_status_embed.get(&endpoint.url) {
                combined_status.insert(endpoint.url.clone(), hs);
            }
        }
    }

    HttpResponse::Ok().json(combined_status)
}

// -- Handler: /reload (resets and reapplies both sets) ------------------------
pub async fn reload_handler(req: HttpRequest, state: web::Data<Arc<AppState>>) -> impl Responder {
    // Auth check
    let auth_info = match req.extensions().get::<AuthInfo>() {
        Some(info) => info.clone(),
        None => return HttpResponse::Unauthorized().finish(),
    };
    if !auth_info.groups.contains(&"admin".to_string())
        && !auth_info.groups.contains(&"staff".to_string())
    {
        return HttpResponse::Forbidden().finish();
    }

    match load_endpoints_from_yaml() {
        Ok(new_endpoints) => {
            let (new_generate, new_embed) = partition_endpoints(new_endpoints.clone());
            {
                let mut gen_lock = state.endpoints_generate.lock().unwrap();
                *gen_lock = new_generate;
                let mut emb_lock = state.endpoints_embed.lock().unwrap();
                *emb_lock = new_embed;
            }
            {
                state.health_status_generate.lock().unwrap().clear();
                state.health_status_embed.lock().unwrap().clear();
            }
            {
                state.endpoint_models_generate.lock().unwrap().clear();
                state.endpoint_models_embed.lock().unwrap().clear();
            }
            {
                state.model_to_endpoints_generate.lock().unwrap().clear();
                state.model_to_endpoints_embed.lock().unwrap().clear();
            }

            // Reload auth tokens
            match load_auth_tokens_from_yaml() {
                Ok(new_auth_tokens) => {
                    let mut auth_tokens = state.auth_tokens.lock().unwrap();
                    *auth_tokens = new_auth_tokens;
                }
                Err(e) => {
                    return HttpResponse::InternalServerError()
                        .body(format!("Failed to load auth tokens YAML: {}", e));
                }
            }

            // Spin up monitors again
            for endpoint in new_endpoints {
                let state_clone = state.get_ref().clone();
                tokio::spawn(async move {
                    monitor_endpoint(endpoint, state_clone).await;
                });
            }

            HttpResponse::Ok().body("Reloaded endpoints and reset all statuses")
        }
        Err(e) => HttpResponse::InternalServerError().body(format!("Failed to load YAML: {}", e)),
    }
}

// -- Handler: /health ---------------------------------------------------------
pub async fn health_handler() -> impl Responder {
    HttpResponse::Ok().finish()
}