output "gateway_url" {
  value = "http://${aws_instance.app.public_ip}:8000"
}

output "instance_id" {
  value = aws_instance.app.id
}

output "api_key" {
  value     = random_password.api_key.result
  sensitive = true # terraform output -raw api_key
}

output "dashboard_url" {
  value = "https://${var.region}.console.aws.amazon.com/cloudwatch/home?region=${var.region}#dashboards/dashboard/${aws_cloudwatch_dashboard.main.dashboard_name}"
}

output "admin_shell" {
  value = "aws ssm start-session --region ${var.region} --target ${aws_instance.app.id}"
}

output "grafana_tunnel" {
  description = "Grafana stays private; tunnel it to localhost:3000 through SSM"
  value       = "aws ssm start-session --region ${var.region} --target ${aws_instance.app.id} --document-name AWS-StartPortForwardingSession --parameters portNumber=3000,localPortNumber=3000"
}
