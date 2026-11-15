# CloudWatch without an agent: the containers' stdout goes to CloudWatch Logs (Docker awslogs
# driver), and metric filters turn the gateway's JSON access log into metrics (request count,
# latency with percentiles, 429s, 5xx). EC2 publishes CPU and status checks on its own.

locals {
  ns  = "QwenSQL"
  api = "$.msg = \"request\" && ($.path = \"/v1/sql\" || $.path = \"/v1/completions\" || $.path = \"/v1/chat/completions\")"
  filter = { for k, v in {
    Requests  = { pattern = "{ ${local.api} }", value = "1" }
    LatencyMs = { pattern = "{ ${local.api} }", value = "$.latency_ms" }
    Rejected  = { pattern = "{ ${local.api} && $.status = 429 }", value = "1" }
    ServerErr = { pattern = "{ ${local.api} && $.status >= 500 }", value = "1" }
  } : k => v }
}

resource "aws_cloudwatch_log_metric_filter" "gw" {
  for_each       = local.filter
  name           = "qwen-sql-${each.key}"
  log_group_name = aws_cloudwatch_log_group.app.name
  pattern        = each.value.pattern
  metric_transformation {
    name      = each.key
    namespace = local.ns
    value     = each.value.value
    unit      = each.key == "LatencyMs" ? "Milliseconds" : "Count"
  }
}

resource "aws_sns_topic" "alerts" {
  name = "qwen-sql-alerts"
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

resource "aws_cloudwatch_metric_alarm" "p95_latency" {
  alarm_name          = "qwen-sql-p95-latency-over-slo"
  alarm_description   = "p95 /v1 latency above the 10 s SLO for 3 consecutive 5-minute periods"
  namespace           = local.ns
  metric_name         = "LatencyMs"
  extended_statistic  = "p95"
  period              = 300
  evaluation_periods  = 3
  threshold           = 10000
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  depends_on          = [aws_cloudwatch_log_metric_filter.gw]
}

resource "aws_cloudwatch_metric_alarm" "server_errors" {
  alarm_name          = "qwen-sql-5xx"
  alarm_description   = "3+ gateway 5xx responses in 5 minutes (backend down, timeouts)"
  namespace           = local.ns
  metric_name         = "ServerErr"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 3
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  depends_on          = [aws_cloudwatch_log_metric_filter.gw]
}

resource "aws_cloudwatch_metric_alarm" "instance_status" {
  alarm_name          = "qwen-sql-instance-status"
  alarm_description   = "EC2 status check failing"
  namespace           = "AWS/EC2"
  metric_name         = "StatusCheckFailed"
  dimensions          = { InstanceId = aws_instance.app.id }
  statistic           = "Maximum"
  period              = 60
  evaluation_periods  = 3
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

resource "aws_budgets_budget" "monthly" {
  name         = "qwen-sql-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
  }
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.alert_email]
  }
}

resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "qwen-sql"
  dashboard_body = jsonencode({
    widgets = [
      { type = "metric", x = 0, y = 0, width = 12, height = 6, properties = {
        title = "Requests and refusals (per minute)", region = var.region, period = 60, stat = "Sum", view = "timeSeries",
      metrics = [[local.ns, "Requests"], [local.ns, "Rejected"], [local.ns, "ServerErr"]] } },
      { type = "metric", x = 12, y = 0, width = 12, height = 6, properties = {
        title   = "Latency p50 / p95 (ms) vs 10 s SLO", region = var.region, period = 60, view = "timeSeries",
        metrics = [[local.ns, "LatencyMs", { stat = "p50" }], [local.ns, "LatencyMs", { stat = "p95" }]],
      annotations = { horizontal = [{ label = "SLO", value = 10000 }] } } },
      { type = "metric", x = 0, y = 6, width = 12, height = 6, properties = {
        title = "CPU utilization (%)", region = var.region, period = 60, stat = "Average", view = "timeSeries",
      metrics = [["AWS/EC2", "CPUUtilization", "InstanceId", aws_instance.app.id]] } },
      { type = "log", x = 12, y = 6, width = 12, height = 6, properties = {
        title = "Latest gateway requests", region = var.region, view = "table",
      query = "SOURCE '${aws_cloudwatch_log_group.app.name}' | filter msg = 'request' and path like /v1/ | fields @timestamp, status, latency_ms, path, request_id | sort @timestamp desc | limit 20" } },
    ]
  })
}
