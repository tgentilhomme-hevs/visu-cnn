# visu-cnn

Follow the instructions [here](./Lab.md)


## Quick start (Docker)

```bash
docker compose up -d
docker exec -it pydev bash
pip install -e . -r requirements.txt
pytest -q
```

## Quick start (Local)

```bash
python3 -m venv venv    
source venv/bin/activate
pip install -e . -r requirements.txt
pytest -q
```

## Run tests

```bash
pytest -q
``` 
