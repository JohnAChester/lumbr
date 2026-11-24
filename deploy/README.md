# Demo search server

The demo site (Vercel) calls the B0 search service at **https://api.johnchester.dev**. The service runs on one
AWS Lightsail server, because the full corpus and indexes (~7 GB) don't fit a Vercel function.

| | |
|---|---|
| Server | Lightsail `lumbr-search`, plan `medium_3_0` (4 GB RAM, 2 vCPU, 80 GB), Ubuntu 24.04, us-east-1a, $24/month |
| Address | static IP `lumbr-search-ip`, DNS A record `api` on Cloudflare (DNS only, not proxied) |
| Firewall | 22 from John's IP only; 80 and 443 open |
| Backups | Lightsail automatic daily snapshots, last 7 kept |
| Monitoring | external uptime check on `/health`, email to John |
| AWS access | IAM user `lumbr-dev` from [`aws/iam-cli-user.yaml`](aws/iam-cli-user.yaml): Lightsail in us-east-1 only, no root |

## How a request flows

```
Vercel site --HTTPS + Bearer key--> nginx :443 (Let's Encrypt, 10 req/s per IP, burst 20)
            --> uvicorn 127.0.0.1:8000 (systemd lumbr-search, user lumbr) --> B0: BM25 (Lucene) + R4 (FAISS fp16), RRF
```

`/retrieve` and `/passages` need `Authorization: Bearer <LUMBR_BACKEND_KEY>`. The key lives in
`/etc/lumbr/backend.env` (root-only) and in Vercel. `/health` is public.

## Files

| File | Installed as |
|---|---|
| [`lightsail/setup.sh`](lightsail/setup.sh) | run once on a fresh server |
| [`lightsail/lumbr-search.service`](lightsail/lumbr-search.service) | `/etc/systemd/system/lumbr-search.service` |
| [`lightsail/lumbr-pin.service`](lightsail/lumbr-pin.service) | `/etc/systemd/system/lumbr-pin.service` |
| [`lightsail/warmup.sh`](lightsail/warmup.sh) | `/usr/local/lib/lumbr/warmup.sh` |
| [`lightsail/nginx-lumbr.conf`](lightsail/nginx-lumbr.conf) | `/etc/nginx/sites-available/lumbr` (certbot then adds HTTPS) |
| [`lightsail/nginx-ratelimit.conf`](lightsail/nginx-ratelimit.conf) | `/etc/nginx/conf.d/lumbr-ratelimit.conf` |
| [`aws/iam-cli-user.yaml`](aws/iam-cli-user.yaml) | CloudFormation stack `lumbr-iam-cli-user` (us-east-1) |

## Why it's set up this way

- **4 GB plan.** The process holds ~1.5 GB, and the rest is page cache for the 1.1 GB FAISS index and the BM25
  postings. Swap (2 GB) is a safety net and normally unused.
- **CPU-only PyTorch.** `pip install -e ".[demo,demo-b0]"` pulls the CUDA build by default. On this server it
  loaded unused GPU libraries, used ~5 GB more disk, and pushed the index out of RAM, so some queries took
  8-60 s. `setup.sh` installs the CPU wheel first.
- **FAISS index locked in RAM (`lumbr-pin`).** Every query scans the whole 1.13 GB index. Even with CPU torch,
  only ~0.7 GB of file pages stayed in the active cache, so parts of the index were pushed out and re-read from
  disk. Batches stalled for 25-30 s, with 23-36% I/O pressure. `vmtouch -l` locks the file in RAM, and
  `LUMBR_JVM_HEAP=512m` hands ~0.5 GB back to the page cache (Lucene uses mmap, not the heap). After the fix,
  150 sequential k=1 calls at ~5 req/s all returned in 0.44 s or less.
- **Warm-up at start.** Right after a start, the first queries paid for random disk reads (up to 60 s).
  `warmup.sh` reads the indexes sequentially and runs 20 queries before systemd marks the service started.
  A cold start takes ~3.5 minutes; after that, queries take ~0.2 s on the server and ~0.3 s end to end.

## Common tasks

Deploy a new version of `main`:

```bash
ssh ubuntu@52.86.172.3
sudo -u lumbr bash -c 'cd /srv/lumbr/app && git pull -q && /srv/lumbr/venv/bin/pip install -q -e ".[demo,demo-b0]"'
sudo -u lumbr /srv/lumbr/venv/bin/python -c "import torch; print(torch.__version__)"   # must end in +cpu
sudo systemctl restart lumbr-search    # ~3.5 min, warm-up included
```

Check on it:

```bash
curl -s https://api.johnchester.dev/health
sudo systemctl status lumbr-search
sudo journalctl -u lumbr-search -n 50
```

New artifacts (needs the HF login on the server; delete the token afterwards):

```bash
sudo -u lumbr -H /srv/lumbr/venv/bin/hf download johnchester/lumbr-demo-b0 --repo-type dataset --local-dir /srv/lumbr/artifacts
sudo systemctl restart lumbr-search
```

SSH refused: John's IP changed. Update the port 22 rule (Lightsail console, or `PutInstancePublicPorts` as
`lumbr-dev`).

## Rebuild from scratch

1. Create the instance and static IP above, import the SSH key, and set the firewall.
2. `scp ~/.lumbr/backend_key ubuntu@<ip>:/tmp/backend_key`, then run `lightsail/setup.sh` on the server.
3. Follow the manual steps it prints: HF login and download, start the service, DNS record, certbot.
4. Faster alternative: restore the latest automatic snapshot as a new instance and move the static IP to it.
