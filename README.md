# Qwen Image Edit - RunPod Serverless

Image editing API powered by Qwen2.5-VL, deployed on RunPod serverless infrastructure.

## Docker Image

Ready-to-use Docker image available on Docker Hub:

```bash
docker pull lawlieties/qwen-image-edit
```

## RunPod Deployment

1. Create a new Serverless Endpoint on [RunPod](https://runpod.io)
2. Use the Docker image: `lawlieties/qwen-image-edit`
3. Configure GPU (recommended: RTX 3090 or higher)

## Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

`RUNPOD_ENDPOINT_ID` Your RunPod endpoint ID
`RUNPOD_API_KEY` Your RunPod API key

### API Usage

### Request

```json
{
  "input": {
    "prompt": "Change the t-shirt to red color",
    "image": "<base64_encoded_image> or <url>",
    "num_inference_steps": 30,
    "guidance_scale": 1.0
  }
}
```

### Response

```json
{
  "output": {
    "image": "<base64_encoded_result>"
  }
}
```