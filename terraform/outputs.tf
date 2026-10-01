# terraform/outputs.tf

output "public_ip" {
  description = "Public IP address of the VERTO Cloud Instance"
  value       = aws_instance.verto_server.public_ip
}

output "dashboard_url" {
  description = "URL to access the VERTO Web Dashboard"
  value       = "http://${aws_instance.verto_server.public_ip}:${var.verto_port}"
}

output "cli_export_command" {
  description = "Command to run locally so the verto CLI routes to the cloud server"
  value       = "export VERTO_SERVER_URL=http://${aws_instance.verto_server.public_ip}:${var.verto_port}"
}
