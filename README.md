# Fine-grained vehicle identification

Take a photo of a car on your phone and get back its make and model: *Toyota
Wish*, *Honda Civic*, and so on. The photo goes from a web app in the browser,
through the cloud, to a **hand-written CUDA inference engine** running on a
local GPU. The engine compares the photo against a database of reference
images and returns the closest match.

> **Current version: prototype v0.** The CUDA engine works end to end from the
> command line: it embeds images, stores them, and identifies new images by
> nearest-neighbour search on the GPU. The network service, the Postgres
> database and the mobile web app are not built yet. See [Status](#status).

The engine is the core of the project, and it is built from scratch: our own
CUDA kernels and our own ONNX graph executor, with no ONNX Runtime, cuBLAS or
TensorRT in the path that serves a request. Details are in
**[CUDA/README.md](CUDA/README.md)**.

## How it fits together

```mermaid
flowchart LR
    subgraph phone["📱 Any phone or desktop browser"]
        UI["Web app<br/>(Expo, exported for web)<br/>camera or gallery"]
    end

    subgraph cloud["☁️ Cloud"]
        APP["App service<br/>hosts the web app,<br/>relays requests"]
    end

    subgraph home["🖥️ Local machine with the GPU"]
        TUN["Tunnel endpoint<br/>(outbound connection,<br/>no open router port)"]
        SVC["CUDA service<br/>C++ HTTP server"]
        ENG["CUDA inference engine<br/>preprocess → CLIP ViT-B/32<br/>→ 512-dim embedding"]
        KNN["GPU kNN search<br/>+ label vote"]
        DB[("PostgreSQL<br/>labels + embeddings")]
        GPU{{"RTX 4060"}}
    end

    UI -- "HTTPS: upload photo" --> APP
    APP -- "HTTPS" --> TUN
    TUN --> SVC
    SVC --> ENG
    ENG --> KNN
    KNN <--> DB
    ENG -. runs on .- GPU
    KNN -. runs on .- GPU
    SVC -- "best match + candidates" --> TUN
    TUN --> APP
    APP --> UI
```

**Why the GPU stays at home.** Renting a cloud GPU around the clock to serve a
demo costs far more than the local card. The GPU work happens on a local
machine; only the lightweight web app runs in the cloud.

**Why a tunnel rather than port forwarding.** The phone loads the app over
HTTPS, and browsers block an HTTPS page from calling a plain-HTTP address
("mixed content"). So the GPU machine needs an HTTPS address that the cloud can
reach. Forwarding a router port works, but it exposes the home IP and needs a
DNS name and a TLS certificate. It also fails outright if the ISP uses
carrier-grade NAT. An outbound tunnel (Cloudflare Tunnel, Tailscale Funnel,
ngrok) gives the local service an HTTPS address without opening any inbound
port. The final choice is still open.

## What happens to one photo

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant W as Web app (browser)
    participant A as App service (cloud)
    participant S as CUDA service (local)
    participant G as GPU
    participant D as PostgreSQL (local)

    U->>W: take a photo or pick one from the gallery
    W->>A: POST image (HTTPS)
    A->>S: forward through the tunnel (HTTPS)
    S->>G: decode → resize/normalise → run CLIP graph
    G-->>S: 512-dim embedding
    Note over S,D: the stored embeddings are loaded onto<br/>the GPU once, at start-up
    S->>G: similarity vs every stored embedding (one GEMM)
    G-->>S: k nearest neighbours
    S->>S: similarity-weighted vote per label
    S-->>A: best label (e.g. honda_civic), vote share, distance, candidates
    A-->>W: result
    W-->>U: "Honda Civic", with the runner-up matches
```

## Identifying by similarity, not by fixed classes

A conventional classifier has one output per known car model. Teaching it a new
model means retraining. This project instead turns every image into a
**fingerprint** (an embedding: 512 numbers) and identifies a photo by finding
the most similar stored fingerprints.

```mermaid
flowchart TB
    subgraph build["Building the catalog (offline, any time)"]
        direction LR
        R["Reference photos<br/>one folder per model<br/>toyota_wish/ honda_civic/ ..."] --> E1["CUDA engine"] --> F1["fingerprints"] --> DB[("database<br/>label + fingerprint")]
    end
    subgraph query["Identifying a photo"]
        direction LR
        Q["New photo"] --> E2["CUDA engine"] --> F2["fingerprint"] --> NN["find the nearest<br/>stored fingerprints"] --> L["vote → label"]
    end
    DB -. compared against .-> NN
```

- **Adding a model is a data change, not a retrain.** Drop a folder of
  photos into the catalog and that model can be recognised. This matters for
  models that public datasets miss. Stanford Cars, the standard benchmark, covers
  US-market cars up to 2012, so a regional model like the Toyota Wish has to come
  from your own photos.
- **The domain is configuration.** The engine only sees a `[N, 3, 224, 224]`
  image tensor. Cars are just the first catalog; flowers or birds would be a
  different set of folders.

## Components

| Component | Where | What it is | State |
|---|---|---|---|
| **CUDA inference engine** | [`CUDA/`](CUDA/README.md) | Hand-written CUDA kernels and ONNX graph executor running CLIP ViT-B/32's vision encoder; preprocessing on the GPU; embedding store; GPU brute-force kNN | **v0 working**, as a command-line tool |
| CUDA service | `CUDA/src/server/` (planned) | C++ HTTP server around the engine: request handling, dynamic batching, backpressure | planned |
| Database | local PostgreSQL | Catalog of labels and embeddings. The engine already talks to storage through a proxy interface; a local file backend stands in for Postgres today | file backend only |
| Web app | [`app/`](app/) | Expo (React Native) app, exported for the web so any phone browser can use it; camera or gallery upload | starter template |
| App service | cloud | Hosts the web app and relays requests to the local CUDA service | planned |
| Offline Python | [`CUDA/python/`](CUDA/python/) | Model export to ONNX, reference outputs for correctness checks, later fine-tuning | export working |

### The CUDA engine in one picture

```mermaid
flowchart LR
    J["JPEG / PNG"] --> D["decode<br/>(OpenCV, CPU)"]
    D --> P["bicubic resize + crop<br/>+ normalise<br/><b>our CUDA kernels</b>"]
    P --> X["ONNX graph executor<br/>~480 nodes, 14 op types<br/><b>our C++</b>"]
    X --> K["Conv · MatMul · LayerNorm<br/>Softmax · Gather · ...<br/><b>our CUDA kernels</b>"]
    K --> V["512-dim embedding"]
    V --> S["kNN on our GEMM kernel<br/>+ weighted label vote"]
```

Correctness is checked node by node against ONNX Runtime, which is used only as
a test oracle and is never linked into the serving path. Commands, storage
format, kernels and benchmarks: **[CUDA/README.md](CUDA/README.md)**.

### Is Expo a good fit for the web app?

Yes. Expo Router builds for the web through `react-native-web`, and this repo is
already set up for it: `app.json` has a `web` target with static output. A
static web export can be hosted on any static host or cloud bucket, and phone
browsers can open it with nothing installed. `expo-image-picker` works in the
browser too: on a phone it offers the camera and the photo library through the
browser's own file picker. The same code can later become native iOS and
Android apps. If the relay also needs to live in the app, Expo Router can serve
API routes from a server build. Otherwise a small separate service does that
job.

## Status

**Prototype v0.** What works today, run from the command line on the GPU
machine:

```bash
# Build a catalog: one folder per model, the folder name is the label
cuda add   --storage data --images "photos/catalog/"

# Identify photos against it
cuda infer --storage data --images "photos/query/"

# Maintain it
cuda read   --storage data
cuda update --storage data --images "photos/catalog/honda_civic/"
cuda delete --storage data --label toyota_wish
```

```mermaid
flowchart LR
    v0["<b>v0 — now</b><br/>engine end to end<br/>file-backed store<br/>GPU kNN identify<br/>CLI"]
    v1["matmul optimisation<br/>vs cuBLAS, profiled"]
    v2["fine-tune on<br/>Stanford Cars<br/>+ accuracy check"]
    v3["PostgreSQL backend<br/>+ indexed search"]
    v4["C++ HTTP service<br/>batching, backpressure,<br/>load test"]
    v5["web app + cloud relay<br/>+ tunnel"]
    v0 --> v1 --> v2 --> v3 --> v4 --> v5
```

## Repository layout

```
.
├── README.md            this file: the whole system
├── app/, assets/        Expo web app (starter template for now)
├── app.json, package.json
├── cuda.sln             Visual Studio solution for the engine
└── CUDA/                the CUDA inference engine; see CUDA/README.md
    ├── src/             kernels, graph executor, storage, search, tests
    ├── python/          offline: ONNX export, reference outputs
    ├── tools/           GPU clock locking, profiling scripts
    └── BENCHMARKS.md    append-only measurement log
```

## Getting started

**CUDA engine** (Windows, Visual Studio, CUDA, RTX-class GPU). Build
instructions, model export and every command are in
[CUDA/README.md](CUDA/README.md). The short version:

```bash
msbuild cuda.sln /p:Configuration=Release /p:Platform=x64
```

**Web app** (Node.js):

```bash
npm install
npx expo start --web
```
