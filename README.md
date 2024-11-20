# vllm_composer

Functionality for setting up a reverse proxy to compose multiple [vLLM](https://github.com/vllm-project/vllm) serving instances. Includes basic load balancing per hosted model. Much heavy lifting is done by [Caddy](https://github.com/caddyserver/caddy)

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
   Open the `.env` file in a text editor (e.g., `vim .env`) and set the path to the SSL certificate and key.

5. **Navigate to the Middleware Directory**
   ```bash
   cd middleware/
   ```

6. **Copy the Configuration Template**
   ```bash
   cp config.yml.template config.yml
   ```

7. **Configure the Middleware Settings**
   Open the `config.yml` file in a text editor (e.g., `vim config.yml`) and configure the servers and app settings.

8. **Copy the Secrets Template**
   ```bash
   cp secrets.yml.template secrets.yml
   ```

9. **Configure the Secrets**
   Open the `secrets.yml` file in a text editor (e.g., `vim secrets.yml`) and configure the tokens and user groups.

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

13. **Build the Docker Compose Services**
    ```bash
    docker compose build
    ```

14. **Start the Docker Compose Services**
    ```bash
    docker compose up -d
