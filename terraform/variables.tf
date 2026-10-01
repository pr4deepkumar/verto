# terraform/variables.tf

variable "aws_region" {
  description = "AWS region for cloud deployment"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type. T5-large's weights alone take about 3 GB of memory; a t3.xlarge has 16 GB. For a GPU (e.g. g4dn.xlarge), also set ami_id to an image with the NVIDIA driver."
  type        = string
  default     = "t3.xlarge"
}

variable "ami_id" {
  description = "AMI to boot. Empty uses the latest Ubuntu 22.04, which has no NVIDIA driver; GPU instance types need an AWS Deep Learning AMI or similar."
  type        = string
  default     = ""
}

variable "root_volume_gb" {
  description = "Root disk size in GB. The AMI default of 8 GB is too small for torch, T5-large and the Spider data."
  type        = number
  default     = 40
}

variable "allowed_cidr" {
  description = "The only address range that can reach SSH and the API, e.g. your IP as 203.0.113.4/32"
  type        = string
}

variable "verto_port" {
  description = "Port on which VERTO Dashboard API runs"
  type        = number
  default     = 8000
}

variable "key_name" {
  description = "Optional SSH key pair name for EC2 instance access"
  type        = string
  default     = ""
}

variable "checkpoint_url" {
  description = "Optional URL of a best_model.pt to download at boot, such as a GitHub release asset. Empty: copy it to /opt/VERTO/checkpoints/ yourself and restart the verto service."
  type        = string
  default     = ""
}
