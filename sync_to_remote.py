import os
import tarfile
import paramiko
from pathlib import Path

# Create a clean tar.gz of ceis excluding runs/ and caches
archive_path = Path("c:/Users/CSE_SDPL/nlp/ceis_remote_sync.tar.gz")
base_dir = Path("c:/Users/CSE_SDPL/nlp/ceis")

print("Creating tar archive of ceis...")
with tarfile.open(archive_path, "w:gz") as tar:
    for root, dirs, files in os.walk(base_dir):
        # Exclude runs, __pycache__, .git
        dirs[:] = [d for d in dirs if d not in ("runs", "__pycache__", ".git", ".pytest_cache")]
        for file in files:
            if file.endswith((".pyc", ".tar.gz", ".log")):
                continue
            full_path = Path(root) / file
            arc_name = full_path.relative_to(base_dir)
            tar.add(full_path, arcname=str(arc_name))

print("Tar archive created, size:", round(archive_path.stat().st_size / (1024*1024), 2), "MB")

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("172.16.72.6", username="sreekanthreddy", password="mech@123", timeout=15)

sftp = ssh.open_sftp()
remote_tar = "/home/sreekanthreddy/ceis_remote_sync.tar.gz"
print(f"Uploading {archive_path} to {remote_tar}...")
sftp.put(str(archive_path), remote_tar)
sftp.close()

print("Extracting on remote server...")
stdin, stdout, stderr = ssh.exec_command("mkdir -p /home/sreekanthreddy/ceis && tar -xzf /home/sreekanthreddy/ceis_remote_sync.tar.gz -C /home/sreekanthreddy/ceis && ls -la /home/sreekanthreddy/ceis")
print(stdout.read().decode('utf-8', errors='replace').strip())

ssh.close()
print("Sync complete!")
