# vllm_composer

Functionality for setting up a reverse proxy to compose multiple [vLLM](https://github.com/vllm-project/vllm) serving instances. Includes basic load balancing per hosted model. Much heavy lifting is done by [Caddy](https://github.com/caddyserver/caddy).

## Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/JanNogga/vllm_composer.git
   ```

2. **Navigate to the Project Directory**
   ```bash
   cd vllm_composer/
   ```

3. **Copy the Environment Template**
   ```bash
   cp .env.template .env
   ```

4. **Configure the Environment Variables**

   Open the `.env` file in a text editor (e.g., `vim .env`) and set the path to the SSL certificate and key. If you like you can also configure [Open WebUI](https://github.com/open-webui/open-webui) there.

5. **Navigate to the Middleware Directory**
   ```bash
   cd middleware/
   ```

6. **Copy the Configuration Template**
   ```bash
   cp endpoints.yaml.template endpoints.yml
   ```

7. **Configure the Middleware Settings**

   Open the `endpoints.yaml` file in a text editor (e.g., `vim endpoints.yaml`) and configure the vLLM server settings.

8. **Copy the Secrets Template**
   ```bash
   cp secrets.yaml.template secrets.yaml
   ```

9. **Configure the Secrets**

   Open the `secrets.yaml` file in a text editor (e.g., `vim secrets.yaml`) and configure the tokens and user access groups.

10. **Navigate to the Caddy Directory**
    ```bash
    cd ../caddy/
    ```

11. **Copy the Caddyfile Template**
    ```bash
    cp Caddyfile.template Caddyfile
    ```

12. **Configure the Caddyfile**

    Open the `Caddyfile` in a text editor (e.g., `vim Caddyfile`) and configure the server URL.

13. **Build the middleware Docker image**

    ```bash
    cd .. && docker compose build
    ```

14. **Start caddy, open-webui and middleware**

    ```bash
    docker compose up -d
    ```

Note that it is also possible to use only the middleware and caddy without open-webui or only the middleware without anything else. Just adjust `docker-compose.yml` accordingly.