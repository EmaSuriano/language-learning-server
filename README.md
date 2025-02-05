# language-learning-server


## How to setup

This project has been setup using `uv` package manager, so please install it in your local computer.

```bash
> uv venv
> source .venv/bin/activate
> uv sync
> uv run src/main.py
```

This will start the server locally running at `http://0.0.0.0:8000`, you can check the documentation under `http://0.0.0.0:8000/docs` for each API


## Running this project on Windows

In case you have a GPU installed in your computer and you are using Windows as your OS, then you have to open and run the project entirely using WSL, which is a feature inside Windows that allows to run a Linux distro, in most of the cases Ubuntu, and thanks to that you can access CUDA. Otherwise you are going to be running the project using the CPU only.


Expose WSL route to the network

```bash
> netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=172.27.122.158
```

## Expose local server with 

