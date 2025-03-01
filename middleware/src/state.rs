// External crates
use serde::{Deserialize, Serialize};
use serde_json::Value;
use log::info;

// Standard library
use std::collections::HashMap;
use std::fs;
use std::io;
use std::sync::Mutex;
use std::path::Path;

// -----------------------------------------------------------------------------
// Structures
// -----------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Endpoint {
    pub url: String,
    pub access_token: String,
    pub groups: Vec<String>,
    // "generate" or "embed"
    pub task: String,
}

#[derive(Debug, Serialize)]
pub struct EndpointHealth {
    pub current_status: bool,
    pub consecutive_checks: u32,
    pub check_interval: u64,
}

#[derive(Debug, Deserialize)]
pub struct Secrets {
    pub groups: Vec<HashMap<String, Vec<String>>>,
}

// -----------------------------------------------------------------------------
// YAML Loading Functions
// -----------------------------------------------------------------------------
pub fn load_auth_tokens_from_yaml() -> Result<HashMap<String, Vec<String>>, Box<dyn std::error::Error>> {
    let path = Path::new("secrets.yaml").canonicalize()?;
    info!("Load secrets from: {}", path.display());
    let contents = fs::read_to_string(&path)?;
    let secrets: Secrets = serde_yaml::from_str(&contents)?;
    let mut tokens = HashMap::new();
    for group_map in secrets.groups {
        for (group, tokens_list) in group_map {
            tokens.insert(group, tokens_list);
        }
    }
    Ok(tokens)
}

pub fn load_endpoints_from_yaml() -> io::Result<Vec<Endpoint>> {
    let path = Path::new("endpoints.yaml").canonicalize()?;
    info!("Load endpoints from: {}", path.display());
    let contents = fs::read_to_string(&path)?;
    let raw_endpoints: Vec<serde_yaml::Value> = serde_yaml::from_str(&contents).map_err(|e| {
        io::Error::new(io::ErrorKind::InvalidData, format!("YAML parse error: {}", e))
    })?;

    let mut endpoints = Vec::new();
    for mut raw in raw_endpoints {
        // If "task" is missing, default to "generate".
        if let Some(mapping) = raw.as_mapping_mut() {
            let key = serde_yaml::Value::String("task".to_string());
            if !mapping.contains_key(&key) {
                mapping.insert(key, serde_yaml::Value::String("generate".to_string()));
            }
        }
        let endpoint: Endpoint = serde_yaml::from_value(raw).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("YAML parse error: {}", e))
        })?;
        if endpoint.task != "generate" && endpoint.task != "embed" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid task value: {}", endpoint.task),
            ));
        }
        endpoints.push(endpoint);
    }
    Ok(endpoints)
}

// -----------------------------------------------------------------------------
// Misc Helper Functions
// -----------------------------------------------------------------------------
// Helper to split endpoints into two sets (generate vs. embed).
pub fn partition_endpoints(all: Vec<Endpoint>) -> (Vec<Endpoint>, Vec<Endpoint>) {
    let mut generate_endpoints = Vec::new();
    let mut embed_endpoints = Vec::new();
    for ep in all {
        if ep.task == "generate" {
            generate_endpoints.push(ep);
        } else {
            embed_endpoints.push(ep);
        }
    }
    (generate_endpoints, embed_endpoints)
}

// -----------------------------------------------------------------------------
// App State
// -----------------------------------------------------------------------------
pub struct AppState {
    // Generate-task data
    pub endpoints_generate: Mutex<Vec<Endpoint>>,
    pub health_status_generate: Mutex<HashMap<String, EndpointHealth>>,
    pub endpoint_models_generate: Mutex<HashMap<String, Vec<Value>>>,
    pub model_to_endpoints_generate: Mutex<HashMap<String, Vec<String>>>,

    // Embed-task data
    pub endpoints_embed: Mutex<Vec<Endpoint>>,
    pub health_status_embed: Mutex<HashMap<String, EndpointHealth>>,
    pub endpoint_models_embed: Mutex<HashMap<String, Vec<Value>>>,
    pub model_to_endpoints_embed: Mutex<HashMap<String, Vec<String>>>,

    // Auth tokens -> access groups
    pub auth_tokens: Mutex<HashMap<String, Vec<String>>>,
}