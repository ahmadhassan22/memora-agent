# Deployment Notes — Alibaba Cloud ECS

Memora's FastAPI backend is containerized with Docker and deployed to
Alibaba Cloud ECS. The Streamlit UI runs locally and points at the public
ECS address — this keeps the deployed surface small and directly satisfies
the requirement that the *backend* runs on Alibaba Cloud, without the added
complexity of also containerizing and exposing the UI.

## One-time server setup (Ubuntu 22.04 instance)

\`\`\`bash
# Install Docker + the Compose plugin
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2
sudo systemctl enable --now docker

# Get the code onto the instance
git clone https://github.com/ahmadhassan22/memora-agent.git
cd memora-agent

# Create .env with real credentials (gitignored — never committed)
nano .env
\`\`\`

Paste into `.env`:
\`\`\`
QWEN_API_KEY=your-key-here
QWEN_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
\`\`\`

## Build and run

\`\`\`bash
cd deploy
sudo docker compose up -d --build
\`\`\`

## Verify it's running

\`\`\`bash
curl http://localhost:8000/health
# expect: {"status":"ok","service":"memora"}
\`\`\`

From your own machine (not the server), confirm it's reachable publicly:
\`\`\`
http://<ECS_PUBLIC_IP>:8000/health
\`\`\`
Requires port 8000 to be open in the instance's security group (inbound rule).

## Common operations

\`\`\`bash
sudo docker compose logs -f         # follow live logs
sudo docker compose down            # stop the container
sudo docker compose up -d --build   # rebuild after a code change
\`\`\`