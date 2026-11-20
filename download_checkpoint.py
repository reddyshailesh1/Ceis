import os
import tarfile
import paramiko
from pathlib import Path

local_ceis = Path("c:/Users/CSE_SDPL/nlp/ceis")
local_run = local_ceis / "runs" / "ceis_mednli_v1"
local_run.mkdir(parents=True, exist_ok=True)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("172.16.72.6", username="sreekanthreddy", password="mech@123", timeout=15)

print("Creating remote tarball of checkpoint...")
remote_tar = "/home/sreekanthreddy/ceis_checkpoint.tar.gz"
cmd = f"cd /home/sreekanthreddy/ceis/runs && tar -czf {remote_tar} ceis_mednli_v1"
stdin, stdout, stderr = ssh.exec_command(cmd)
stdout.channel.recv_exit_status()

local_tar = local_ceis / "ceis_checkpoint.tar.gz"
print(f"Downloading {remote_tar} to {local_tar}...")
sftp = ssh.open_sftp()
sftp.get(remote_tar, str(local_tar))
sftp.close()

print("Extracting checkpoint locally...")
with tarfile.open(local_tar, "r:gz") as tar:
    tar.extractall(path=local_ceis / "runs")

print("Local checkpoint files:")
for p in (local_ceis / "runs" / "ceis_mednli_v1").rglob("*"):
    if p.is_file():
        print(" ", p.relative_to(local_ceis / "runs" / "ceis_mednli_v1"), round(p.stat().st_size / (1024*1024), 2), "MB")

ssh.close()
print("Download complete!")
