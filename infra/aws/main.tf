# One EC2 instance running the Week 4 compose stack (gateway + llama-server + Prometheus + Grafana).
# No SSH key and no port 22: admin access is SSM Session Manager. Only :8000 (the gateway) is open,
# and only to var.allowed_cidrs. The API key lives in SSM Parameter Store, not in user data.

data "aws_ssm_parameter" "al2023" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

resource "random_password" "api_key" {
  length  = 40
  special = false
}

resource "aws_ssm_parameter" "api_key" {
  name  = "/qwen-sql/gateway-api-key"
  type  = "SecureString"
  value = random_password.api_key.result
}

resource "aws_cloudwatch_log_group" "app" {
  name              = "/qwen-sql/app"
  retention_in_days = 7
}

# ---------- IAM: SSM access + read the key + write logs ----------

resource "aws_iam_role" "instance" {
  name = "qwen-sql-instance"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "ec2.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "app" {
  name = "qwen-sql-app"
  role = aws_iam_role.instance.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["ssm:GetParameter"], Resource = aws_ssm_parameter.api_key.arn },
      { Effect = "Allow", Action = ["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"],
      Resource = "${aws_cloudwatch_log_group.app.arn}:*" },
    ]
  })
}

resource "aws_iam_instance_profile" "instance" {
  name = "qwen-sql-instance"
  role = aws_iam_role.instance.name
}

# ---------- network ----------

resource "aws_security_group" "gateway" {
  name        = "qwen-sql-gateway"
  description = "Gateway :8000 from allowed CIDRs only; no SSH"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "gateway"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidrs
  }
  egress {
    description = "image pulls, HF model download, CloudWatch, SSM"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ---------- instance ----------

resource "aws_instance" "app" {
  ami                    = data.aws_ssm_parameter.al2023.value
  instance_type          = var.instance_type
  subnet_id              = data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.gateway.id]
  iam_instance_profile   = aws_iam_instance_profile.instance.name

  metadata_options {
    http_tokens = "required" # IMDSv2 only
  }

  root_block_device {
    volume_size = var.root_volume_gb
    volume_type = "gp3"
    encrypted   = true
  }

  user_data_replace_on_change = true
  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    region        = var.region
    repo_url      = var.repo_url
    git_ref       = var.git_ref
    hf_url        = "https://huggingface.co/${var.hf_model_repo}/resolve/${var.hf_revision}/${var.hf_model_file}"
    api_key_param = aws_ssm_parameter.api_key.name
    log_group     = aws_cloudwatch_log_group.app.name
    llama_threads = var.llama_threads
    queue_wait_s  = var.queue_wait_s
  })

  tags = { Name = "qwen-sql" }

  depends_on = [aws_iam_role_policy.app, aws_iam_role_policy_attachment.ssm_core]
}
