// External crates
use reqwest;
use serde_json::Value;
use tokio::time::sleep;

// Standard library
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

// Internal modules
use crate::state::{AppState, Endpoint, EndpointHealth};

// -----------------------------------------------------------------------------
// Monitoring
// -----------------------------------------------------------------------------

pub async fn perform_health_check(url: &str) -> bool {
    match reqwest::get(url).await {
        Ok(resp) => resp.status().is_success(),
        Err(_) => false,
    }
}

pub async fn fetch_models(endpoint: &Endpoint) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{}/v1/models", endpoint.url))
        .bearer_auth(&endpoint.access_token)
        .send()
        .await?
        .error_for_status()?;
    let json: Value = resp.json().await?;
    if let Some(data) = json.get("data").and_then(|d| d.as_array()) {
        Ok(data.clone())
    } else {
        Ok(vec![])
    }
}

// Single monitor function, picks generate vs embed data structures
pub async fn monitor_endpoint(endpoint: Endpoint, state: Arc<AppState>) {
    let mut interval = Duration::from_millis(500);

    loop {
        // If endpoint is no longer in its relevant vector, exit the loop
        {
            let found = if endpoint.task == "generate" {
                let endpoints = state.endpoints_generate.lock().unwrap();
                endpoints.iter().any(|e| e.url == endpoint.url)
            } else {
                let endpoints = state.endpoints_embed.lock().unwrap();
                endpoints.iter().any(|e| e.url == endpoint.url)
            };
            if !found {
                break;
            }
        }

        let health_url = format!("{}/health", endpoint.url);
        let is_healthy = perform_health_check(&health_url).await;

        // Update the correct health map
        let (health_map, endpoint_models, model_to_endpoints) = if endpoint.task == "generate" {
            (
                &state.health_status_generate,
                &state.endpoint_models_generate,
                &state.model_to_endpoints_generate,
            )
        } else {
            (
                &state.health_status_embed,
                &state.endpoint_models_embed,
                &state.model_to_endpoints_embed,
            )
        };

        {
            let mut health_map_lock = health_map.lock().unwrap();
            let entry = health_map_lock.entry(endpoint.url.clone()).or_insert(EndpointHealth {
                current_status: is_healthy,
                consecutive_checks: 0,
                check_interval: interval.as_millis() as u64,
            });
            if entry.current_status == is_healthy {
                entry.consecutive_checks += 1;
                entry.check_interval = std::cmp::min(entry.check_interval + 500, 30_000);
            } else {
                entry.current_status = is_healthy;
                entry.consecutive_checks = 1;
                entry.check_interval = 500;
            }
            interval = Duration::from_millis(entry.check_interval);
        }

        if is_healthy {
            if let Ok(models) = fetch_models(&endpoint).await {
                // Two-way sync
                let mut models_map = endpoint_models.lock().unwrap();
                let mut model_to_endpoints_map = model_to_endpoints.lock().unwrap();

                // Current known models
                let current_models = models_map.get(&endpoint.url).cloned().unwrap_or_default();
                let current_ids: HashSet<String> = current_models
                    .iter()
                    .filter_map(|m| m.get("id").and_then(|v| v.as_str()))
                    .map(String::from)
                    .collect();

                // Freshly fetched models
                let new_ids: HashSet<String> = models
                    .iter()
                    .filter_map(|m| m.get("id").and_then(|v| v.as_str()))
                    .map(String::from)
                    .collect();

                // Identify add/remove
                let to_add = new_ids.difference(&current_ids).cloned().collect::<HashSet<_>>();
                let to_remove = current_ids.difference(&new_ids).cloned().collect::<HashSet<_>>();

                // Update endpoint_models
                models_map.insert(endpoint.url.clone(), models.clone());

                // Add new associations
                for model_id in to_add {
                    let entry = model_to_endpoints_map.entry(model_id).or_insert_with(Vec::new);
                    if !entry.contains(&endpoint.url) {
                        entry.push(endpoint.url.clone());
                    }
                }
                // Remove stale associations
                for model_id in to_remove {
                    if let Some(urls) = model_to_endpoints_map.get_mut(&model_id) {
                        urls.retain(|u| u != &endpoint.url);
                        if urls.is_empty() {
                            model_to_endpoints_map.remove(&model_id);
                        }
                    }
                }
            }
        } else {
            // Remove the endpoint's URL from the model_to_endpoints map
            {
                let mut map_lock = model_to_endpoints.lock().unwrap();
                for urls in map_lock.values_mut() {
                    urls.retain(|u| u != &endpoint.url);
                }
                map_lock.retain(|_, v| !v.is_empty());
            }
            {
                // Remove from endpoint_models
                let mut models_map = endpoint_models.lock().unwrap();
                models_map.remove(&endpoint.url);
            }
        }

        sleep(interval).await;
    }
}