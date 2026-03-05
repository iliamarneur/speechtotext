# Transcription Audio — Whisper

Application web de transcription audio vers texte, basee sur **Faster-Whisper** (open source, gratuit).

## Utilisation rapide (Windows, pour debutants)

### Prerequis

1. **Installer Python 3.10+** : https://www.python.org/downloads/
   - IMPORTANT : cochez **"Add Python to PATH"** pendant l'installation
2. **Installer ffmpeg** : ouvrez un terminal et tapez `winget install ffmpeg`

### Installation

1. Telechargez ce dossier (ou decompressez le ZIP)
2. Double-cliquez sur **`install-windows.bat`**
3. Attendez que l'installation se termine

### Lancement

1. Double-cliquez sur **`demarrer.bat`**
2. Attendez que le message "MODELE PRET" apparaisse (~5 min au 1er lancement)
3. Ouvrez **http://localhost:8000/** dans votre navigateur
4. Deposez un fichier audio, cliquez "Transcrire"
5. Pour arreter : fermez la fenetre noire

---

## Deploiement sur un serveur (VPS Ubuntu)

### Option 1 : Docker (recommande)

Le plus simple pour deployer sur un serveur.

```bash
# 1. Installer Docker
curl -fsSL https://get.docker.com | sh

# 2. Cloner le projet
git clone <url-du-repo> /opt/whisper-stt
cd /opt/whisper-stt

# 3. Lancer (CPU)
docker compose up -d

# 4. Verifier les logs
docker compose logs -f
```

Le modele se telecharge au 1er lancement (~1.5 Go pour medium).
Ensuite le service demarre automatiquement au boot du serveur.

**Si le serveur a un GPU NVIDIA :**
```bash
# Installer nvidia-container-toolkit d'abord
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

docker compose -f docker-compose.gpu.yml up -d
```

### Option 2 : Installation directe

```bash
# 1. Installer les dependances
sudo apt update && sudo apt install -y python3 python3-venv ffmpeg

# 2. Cloner et installer
git clone <url-du-repo> /opt/whisper-stt
cd /opt/whisper-stt
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt requests

# 3. Tester le lancement
WHISPER_MODEL=medium uvicorn main:app --host 0.0.0.0 --port 8000
```

### Service systemd (demarrage automatique)

```bash
sudo tee /etc/systemd/system/whisper-stt.service << 'EOF'
[Unit]
Description=Whisper Speech-to-Text
After=network.target

[Service]
WorkingDirectory=/opt/whisper-stt
ExecStart=/opt/whisper-stt/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
Environment=WHISPER_MODEL=medium
Environment=WHISPER_MODEL_DIR=/opt/whisper-stt/models

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now whisper-stt
sudo systemctl status whisper-stt
```

### Reverse proxy (acces via un nom de domaine)

**Caddy** (le plus simple, HTTPS automatique) :
```bash
sudo apt install -y caddy
echo 'transcription.mondomaine.fr {
    reverse_proxy localhost:8000
}' | sudo tee /etc/caddy/Caddyfile
sudo systemctl restart caddy
```

**Nginx** :
```bash
sudo apt install -y nginx
sudo tee /etc/nginx/sites-available/whisper << 'EOF'
server {
    listen 80;
    server_name transcription.mondomaine.fr;
    client_max_body_size 500M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 600s;
        proxy_buffering off;
    }
}
EOF
sudo ln -s /etc/nginx/sites-available/whisper /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

> `proxy_buffering off` est important pour que la barre de progression fonctionne en streaming.
> `client_max_body_size 500M` permet d'envoyer des gros fichiers audio.
> `proxy_read_timeout 600s` evite le timeout sur les longues transcriptions.

---

## Donner le projet a quelqu'un (ZIP)

Pour partager avec quelqu'un qui ne s'y connait pas :

1. Copiez tout le dossier `speechtotext` dans un ZIP
2. Envoyez le ZIP
3. La personne doit :
   - Installer Python (https://python.org) en cochant "Add to PATH"
   - Decompresser le ZIP
   - Double-cliquer `install-windows.bat`
   - Double-cliquer `demarrer.bat`
   - Ouvrir http://localhost:8000/

Le dossier `models/` peut etre inclus dans le ZIP (~1.5 Go) pour eviter le re-telechargement.

---

## Configuration

| Variable              | Defaut      | Description                                    |
|-----------------------|-------------|------------------------------------------------|
| `WHISPER_MODEL`       | `large-v3`  | Modele Whisper                                 |
| `WHISPER_DEVICE`      | `auto`      | `auto`, `cpu` ou `cuda`                        |
| `WHISPER_COMPUTE_TYPE`| `auto`      | `auto`, `float16`, `int8`, `float32`           |
| `WHISPER_MODEL_DIR`   | `./models`  | Repertoire des modeles                         |

### Choix du modele

| Modele     | Taille  | Qualite FR | Vitesse CPU        |
|------------|---------|------------|---------------------|
| `tiny`     | ~75 Mo  | Basique    | Tres rapide         |
| `base`     | ~150 Mo | Correcte   | Rapide              |
| `small`    | ~500 Mo | Bonne      | Moyen               |
| `medium`   | ~1.5 Go | Tres bonne | Lent (~5x temps reel) |
| `large-v3` | ~3 Go   | Excellente | Tres lent (GPU recommande) |

## Test en ligne de commande

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@mon-audio.mp3" \
  -F "language=fr"
```

## Structure du projet

```
speechtotext/
├── main.py               # Backend FastAPI
├── static/index.html     # Interface web
├── requirements.txt      # Dependances Python
├── install-windows.bat   # Installation one-click Windows
├── demarrer.bat          # Lancement one-click Windows
├── Dockerfile            # Image Docker GPU
├── Dockerfile.cpu        # Image Docker CPU
├── docker-compose.yml    # Compose CPU
├── docker-compose.gpu.yml # Compose GPU
└── README.md
```

## Licence

Composants open source : Faster-Whisper (MIT), FastAPI (MIT), Whisper (MIT).
