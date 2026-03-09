# EC2 Deployment Guide (Dash Dashboard)

This deploys the dashboard to Amazon EC2 so stakeholders can access it via a public URL.

## 1. EC2 prerequisites

- Ubuntu 22.04 instance
- Security Group inbound:
  - `22` (SSH) from your IP
  - `80` (HTTP) from `0.0.0.0/0`
- Optional later: `443` (HTTPS)

## 2. SSH and install dependencies

```bash
ssh -i /path/to/key.pem ubuntu@<EC2_PUBLIC_IP>
sudo apt update
sudo apt install -y python3-venv python3-pip nginx git
```

## 3. Pull repo and create virtual environment

```bash
cd /opt
sudo git clone <YOUR_REPO_URL> organ_donors_analytics
sudo chown -R ubuntu:ubuntu /opt/organ_donors_analytics
cd /opt/organ_donors_analytics
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Generate or copy latest data

If you need fresh data on EC2:

```bash
source .venv/bin/activate
python scripts/fabric_run_pipeline.py --no-plot
```

This writes to `sample_data/l1_bronze`, `sample_data/l2_silver`, `sample_data/quarantine`, and `sample_data/l3_gold`.

## 5. Configure systemd service

```bash
sudo cp deploy/dashboard.service /etc/systemd/system/dashboard.service
sudo systemctl daemon-reload
sudo systemctl enable dashboard
sudo systemctl start dashboard
sudo systemctl status dashboard --no-pager
```

## 6. Configure Nginx reverse proxy

```bash
sudo cp deploy/nginx_dashboard.conf /etc/nginx/sites-available/dashboard
sudo ln -sf /etc/nginx/sites-available/dashboard /etc/nginx/sites-enabled/dashboard
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

Open:

`http://<EC2_PUBLIC_IP>/`

## 7. Optional HTTPS (recommended for stakeholders)

If you have a domain (example `dashboard.example.com`):

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d dashboard.example.com
```

## 8. Updating after code changes

```bash
cd /opt/organ_donors_analytics
git pull
source .venv/bin/activate
pip install -r requirements.txt
python scripts/fabric_run_pipeline.py --no-plot
sudo systemctl restart dashboard
```

## 9. Optional GitHub Actions auto-deploy

This repo can auto-deploy to EC2 on every push to `master` using:

- [.github/workflows/deploy-ec2.yml](/c:/Users/tcham/Wokspace/organ_donors_analytics/.github/workflows/deploy-ec2.yml)

Add these GitHub repository secrets:

- `EC2_HOST`: public IP or DNS name
- `EC2_USER`: SSH user, usually `ubuntu`
- `EC2_SSH_KEY`: private key contents for that instance
- `EC2_PORT`: optional, defaults to `22`

The workflow:

- connects to `/opt/organ_donors_analytics`
- pulls latest `master`
- rebuilds `.venv`
- installs `requirements.txt`
- reruns `python scripts/fabric_run_pipeline.py --no-plot`
- restarts `dashboard` and `nginx`

Before enabling it, make sure the EC2 host already has:

- the repo cloned at `/opt/organ_donors_analytics`
- `dashboard.service` installed
- `nginx` configured
- the deploy user allowed to run `sudo systemctl restart dashboard` and `sudo systemctl restart nginx`

## 10. Health checks / logs

```bash
sudo systemctl status dashboard --no-pager
sudo journalctl -u dashboard -n 100 --no-pager
curl -I http://127.0.0.1:8050/
curl -I http://127.0.0.1/
```
