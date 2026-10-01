# terraform/main.tf

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Fetch latest Ubuntu 22.04 LTS AMI
data "aws_ami" "ubuntu" {
  most_recent = true
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
  owners = ["099720109477"] # Canonical
}

# Security group allowing SSH (22) and the VERTO API (8000) from allowed_cidr only
resource "aws_security_group" "verto_sg" {
  name        = "verto-dashboard-sg"
  description = "Security group for VERTO Cloud Dashboard & API"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
  }

  ingress {
    description = "VERTO API Dashboard"
    from_port   = var.verto_port
    to_port     = var.verto_port
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "verto-security-group"
  }
}

# Launch Cloud EC2 Instance running VERTO Cloud Server
resource "aws_instance" "verto_server" {
  ami                    = var.ami_id != "" ? var.ami_id : data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = var.key_name != "" ? var.key_name : null
  vpc_security_group_ids = [aws_security_group.verto_sg.id]

  root_block_device {
    volume_size = var.root_volume_gb
    volume_type = "gp3"
  }

  user_data = <<-EOF
              #!/bin/bash
              set -e
              apt-get update
              apt-get install -y git sqlite3 curl

              cd /opt
              git clone https://github.com/pr4deepkumar/VERTO.git || true
              cd /opt/VERTO

              # Python 3.12 and the pinned requirements, through uv
              curl -LsSf https://astral.sh/uv/0.11.15/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
              uv venv /opt/VERTO/venv --python 3.12
              uv pip install --python /opt/VERTO/venv/bin/python -r requirements.txt
              uv pip install --python /opt/VERTO/venv/bin/python -e .

              # Spider databases, and the fine-tuned adapter if a URL was given
              /opt/VERTO/venv/bin/python scripts/setup_data.py
              if [ -n "${var.checkpoint_url}" ]; then
                curl -fL "${var.checkpoint_url}" -o checkpoints/best_model.pt
              fi

              # A systemd service, so the API comes back after a crash or a reboot
              cat > /etc/systemd/system/verto.service <<'UNIT'
              [Unit]
              Description=VERTO text-to-SQL API
              After=network-online.target
              Wants=network-online.target

              [Service]
              WorkingDirectory=/opt/VERTO
              ExecStart=/opt/VERTO/venv/bin/uvicorn verto.api.main:app --host 0.0.0.0 --port ${var.verto_port}
              Restart=always
              RestartSec=5

              [Install]
              WantedBy=multi-user.target
              UNIT
              systemctl daemon-reload
              systemctl enable --now verto
              EOF

  tags = {
    Name = "VERTO-Cloud-Dashboard"
  }
}
