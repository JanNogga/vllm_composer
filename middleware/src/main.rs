// External crates
use actix_web::{web, App, HttpServer};
use log::debug;

// Standard library
use std::collections::HashMap;
use std::io;
use std::sync::{Arc, Mutex};
use std::env;

// Internal modules
mod auth;
use auth::AuthMiddleware;

mod routes;
use routes::{
    endpoints_handler,
    health_status_handler,
    reload_handler,
    health_handler,
    models_handler,
    model_to_endpoints_handler,
    chat_completions_handler,
    embeddings_handler,
};

mod state;
use state::{
    AppState,
    load_endpoints_from_yaml,
    load_auth_tokens_from_yaml,
    partition_endpoints,
};

mod monitoring;
use monitoring::monitor_endpoint;

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
#[actix_web::main]
async fn main() -> io::Result<()> {
    env_logger::init();
    debug!("Logger activated.");

    // Load initial endpoints
    let all_endpoints = load_endpoints_from_yaml().unwrap_or_else(|_| Vec::new());
    let (gen_initial, emb_initial) = partition_endpoints(all_endpoints);

    // Load auth tokens
    let auth_tokens = load_auth_tokens_from_yaml().unwrap_or_else(|_| HashMap::new());

    // Construct state
    let state = Arc::new(AppState {
        endpoints_generate: Mutex::new(gen_initial.clone()),
        health_status_generate: Mutex::new(HashMap::new()),
        endpoint_models_generate: Mutex::new(HashMap::new()),
        model_to_endpoints_generate: Mutex::new(HashMap::new()),

        endpoints_embed: Mutex::new(emb_initial.clone()),
        health_status_embed: Mutex::new(HashMap::new()),
        endpoint_models_embed: Mutex::new(HashMap::new()),
        model_to_endpoints_embed: Mutex::new(HashMap::new()),

        auth_tokens: Mutex::new(auth_tokens),
    });

    // Spawn monitors for both sets
    for endpoint in gen_initial {
        let state_clone = Arc::clone(&state);
        tokio::spawn(async move {
            monitor_endpoint(endpoint, state_clone).await;
        });
    }
    for endpoint in emb_initial {
        let state_clone = Arc::clone(&state);
        tokio::spawn(async move {
            monitor_endpoint(endpoint, state_clone).await;
        });
    }

    // Get port from command line arguments or default to 8080
    let port: u16 = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(8080);
    let bind_address = format!("0.0.0.0:{}", port);

    HttpServer::new(move || {
        App::new()
            .wrap(AuthMiddleware)
            .app_data(web::Data::new(state.clone()))
            .route("/endpoints", web::get().to(endpoints_handler))
            .route("/reload", web::get().to(reload_handler))
            .route("/health-status", web::get().to(health_status_handler))
            .route("/v1/models", web::get().to(models_handler))
            .route("/model-to-endpoints", web::get().to(model_to_endpoints_handler))
            .route("/health", web::get().to(health_handler))
            .route("/v1/chat/completions", web::post().to(chat_completions_handler))
            .route("/v1/embeddings", web::post().to(embeddings_handler))
    })
    .bind(bind_address)?
    .run()
    .await
}