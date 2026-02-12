# llama

## start-llama.sh
- Installs build tools with apt.
- Installs CUDA 12.6 if `nvcc` is missing, and updates `~/.bashrc` PATH/LD_LIBRARY_PATH.
- Cleans and configures a Ninja build for `llama.cpp` with CUDA and server enabled.
- Builds `llama.cpp` with Ninja.
- Ensures the configured GGUF model exists in `~/llama-models`.
- Launches `llama-server` with a GPU/CPU split, fixed threads/context/temp, and binds to `0.0.0.0:8080`.
- current models:
  - [Qwen3-Coder-Q8](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF?show_file_info=Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf&local-app=llama.cpp)

## run-llama.service
- Systemd unit that runs `start-llama.sh` as root.
- Uses `/home/jaime/Work/ai/llama` as the working directory.
- Restarts the service on failure with a 10-second delay.
- Waits for the network to be online before starting.
- Sets `HOME=/home/jaime` so the script uses the expected paths.

### Install and manage the service
Copy the unit into root's systemd services, then reload and enable it:

```bash
sudo cp /home/jaime/Work/ai/llama/run-llama.service /etc/systemd/system/run-llama.service
sudo systemctl daemon-reload
sudo systemctl enable --now run-llama.service
```

Common operations:

```bash
sudo systemctl start run-llama.service
sudo systemctl stop run-llama.service
sudo systemctl restart run-llama.service
sudo systemctl status run-llama.service
```

View logs:

```bash
sudo journalctl -u run-llama.service -f
```

### Testing changes
When testing edits to `start-llama.sh` or `run-llama.service`:

```bash
sudo systemctl stop run-llama.service
sudo cp /home/jaime/Work/ai/llama/run-llama.service /etc/systemd/system/run-llama.service
sudo systemctl daemon-reload
sudo systemctl start run-llama.service
sudo journalctl -u run-llama.service -f
```
