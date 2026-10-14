# face-analyzer

A face detection and emotion recognition service built to practise **MLOps**:
versioned data, tracked experiments, a model registry with an evaluation gate,
human-reviewed feedback, automated retraining, CI/CD and monitoring.

STATUS: Early development, not yet in working state


## What it does

Send an image → the service finds faces and estimates the expressed emotion for each one
(angry, disgust, fear, happy, sad, surprise, neutral).
Later versions add continuous valence/arousal and video input.

## Architecture

```mermaid
flowchart LR
    U[Client] -->|image| API[FastAPI service]
    API --> D[Face detector<br/>YuNet]
    D --> C[Emotion classifier]
    C --> API
    U -->|label correction| FB[Feedback pool]
    FB --> R[Human review<br/>Label Studio]
    R --> DS[(Versioned dataset<br/>DVC)]
    DS --> T[Training pipeline]
    T --> MR[(Model registry<br/>MLflow)]
    MR -->|evaluation gate| P{Better than<br/>champion?}
    P -->|yes| API
```
