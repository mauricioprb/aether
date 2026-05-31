# Deploy no Coolify

O Coolify atua como proxy reverso externo: o Traefik embutido termina o TLS
(Let's Encrypt, automático) e roteia o domínio público para o container `web`.
A API fica interna. Os ~28 MB de dataset + checkpoint **não** estão no git nem na
imagem — são baixados no primeiro boot a partir de `ARTIFACTS_URL` para um volume
persistente (o modelo ETR é re-treinado no primeiro uso, veja abaixo).

## 1. Subir o tarball de artefatos

Um tarball com o conjunto mínimo de runtime fica em `/tmp/aether-artifacts.tar.gz`
(~28 MB). Regenere quando quiser com:

```sh
make artifacts-tarball     # veja o target no Makefile, ou rode o tar à mão
```

Conteúdo (extrai relativo a `/app`):

```
data/metadata.sqlite
data/splits.json
data/mace_features/{train,val,test}_emb.npz
logs/checkpoints/mace_ft_stageA_v2_seed42/last.ckpt
```

O modelo ETR (`etr_emb.pkl`, ~193 MB) **não** vai no pacote: a API o re-treina a
partir dos arquivos `_emb.npz` e assina o cache com HMAC na primeira chamada a
`/screen` (~1-2 min, depois servido pelo volume `aether-data`). Isso mantém o
download pequeno e garante que um pickle adulterado nunca viaje dentro do tarball.

Suba para qualquer object storage acessível por HTTP e pegue uma URL de download:

- **Cloudflare R2** (free tier, sem taxa de egress) — recomendado
- **AWS S3 / Backblaze B2** — URL presigned ou objeto público
- Qualquer host de arquivos estáticos que devolva o `.tar.gz` cru

A URL precisa devolver os bytes gzip crus (não uma página HTML de download). Teste:

```sh
curl -fsSL "$ARTIFACTS_URL" | tar -tz | head
```

Calcule o sha256 (obrigatório — a API recusa extrair um tarball não verificado, e
a URL precisa ser `https://`):

```sh
curl -fsSL "$ARTIFACTS_URL" | sha256sum
```

## 2. Criar o recurso no Coolify

1. **+ New** → **Docker Compose** (baseado em Git) → selecione este repo + branch.
2. Em **Docker Compose Location**, aponte para `docker-compose.coolify.yml`.
3. **Environment Variables** → adicione:
   - `ARTIFACTS_URL` = a URL de download do tarball do passo 1 (precisa ser `https://`).
   - `ARTIFACTS_SHA256` = o sha256 do passo 1. O boot aborta se não bater.
   - `ETR_CACHE_KEY` = um segredo aleatório longo (`openssl rand -hex 32`). Assina o
     cache do modelo ETR (HMAC) para que um `etr_emb.pkl` adulterado nunca seja
     desserializado.
   - `API_KEY` *(opcional)* = defina para exigir `X-API-Key` no `/screen`. Deixe
     em branco para o demo público aberto. Se definir, encaminhe também o header
     no proxy (veja Notas).
   - `ENABLE_DOCS` *(opcional)* = `1` para expor `/api/docs`. Desligado por padrão.
4. **Domains** → no serviço `web`, defina seu domínio, porta interna `80`.
   O Coolify provisiona o certificado TLS automaticamente.
5. **Deploy.**

Primeiro deploy: o container da API baixa + extrai o tarball antes de passar no
healthcheck (`start_period: 300s` dá margem pra isso). O `web` espera o `api`
ficar saudável. Acompanhe os logs da API por `[entrypoint] artifacts ready.`

## 3. Redeploys

Os volumes `aether-data` e `aether-checkpoints` são persistentes, então o
download acontece **apenas uma vez**. Redeploys seguintes reaproveitam e sobem
rápido. Para forçar um novo download, apague esses volumes no Coolify e redeploy.

## Endpoints quando estiver no ar

- `https://<domínio>/` — UI web
- `https://<domínio>/api/stats` — API via proxy (rate-limit 10 r/s)
- `https://<domínio>/api/docs` — Swagger do FastAPI (só se `ENABLE_DOCS=1`)

## Notas / limites

- A auth de aplicação é **opt-in**: `API_KEY` em branco = demo aberto (o rate-limit
  do nginx é o único controle); `API_KEY` definido = `/screen` exige um `X-API-Key`
  correspondente. Quando ativado, injete o header no proxy para o SPA continuar
  funcionando, ex.: adicione `proxy_set_header X-API-Key "<chave>";` ao bloco
  `/api/` em `services/web/nginx.conf`, ou use o Basic Auth do Coolify no domínio.
- A API escuta apenas na rede interna do compose (`expose`, nunca publicada),
  então não é alcançável diretamente pela internet.
- `results/` (dados de comparação dos modelos) é embutido na imagem da API, então
  `/api/comparison` funciona sem o tarball.
- Se o proxy rodar em um host separado dos containers, tudo bem — o Coolify conecta
  o proxy à rede dos containers de qualquer forma.
