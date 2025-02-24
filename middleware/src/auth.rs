// External crates
use actix_web::{
    dev::{Service, ServiceRequest, ServiceResponse, Transform},
    HttpMessage,
    HttpResponse,
    web,
    Error,
};
use actix_web::body::{BoxBody, MessageBody};
use futures::future::{ok, LocalBoxFuture, Ready};

// Standard library
use std::rc::Rc;
use std::sync::Arc;

// Internal modules
use crate::state::AppState;

#[derive(Debug, Clone)]
pub struct AuthInfo {
    pub groups: Vec<String>,
}

pub struct AuthMiddleware;

impl<S, B> Transform<S, ServiceRequest> for AuthMiddleware
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    B: MessageBody + 'static,
{
    // Force the response body type to BoxBody so both branches are consistent.
    type Response = ServiceResponse<BoxBody>;
    type Error = Error;
    type InitError = ();
    type Transform = AuthMiddlewareService<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ok(AuthMiddlewareService {
            service: Rc::new(service),
        })
    }
}

pub struct AuthMiddlewareService<S> {
    service: Rc<S>,
}

impl<S, B> Service<ServiceRequest> for AuthMiddlewareService<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    B: MessageBody + 'static,
{
    // Return BoxBody for consistency.
    type Response = ServiceResponse<BoxBody>;
    type Error = Error;
    type Future = LocalBoxFuture<'static, Result<Self::Response, Self::Error>>;

    actix_web::dev::forward_ready!(service);

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let svc = self.service.clone();

        Box::pin(async move {
            // Skip auth check if path is /health
            if req.path() == "/health" {
                return Ok(svc.call(req).await?.map_into_boxed_body());
            }
            
            // Otherwise determine the user groups based on the token
            if let Some(auth_header) = req.headers().get("Authorization").and_then(|h| h.to_str().ok()) {
                if auth_header.starts_with("Bearer ") {
                    let token = auth_header.trim_start_matches("Bearer ").trim();
                    if let Some(state) = req.app_data::<web::Data<Arc<AppState>>>() {
                        let groups: Vec<String> = {
                            let auth_tokens = state.auth_tokens.lock().unwrap();
                            auth_tokens
                                .iter()
                                .filter_map(|(group, tokens)| {
                                    if tokens.contains(&token.to_string()) {
                                        Some(group.clone())
                                    } else {
                                        None
                                    }
                                })
                                .collect()
                        };
                        if !groups.is_empty() {
                            req.extensions_mut().insert(AuthInfo { groups });
                            // Now that all borrows are dropped, we can move `req`.
                            let res = svc.call(req).await?;
                            return Ok(res.map_into_boxed_body());
                        }
                    }
                }
            }

            // If no valid token is found, return an unauthorized response
            let response = req.into_response(
                HttpResponse::Unauthorized().finish().map_into_boxed_body()
            );
            Ok(response)
        })
    }
}
