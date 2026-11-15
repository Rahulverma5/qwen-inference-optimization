variable "region" {
  type    = string
  default = "ap-south-1" # Mumbai: closest to Bengaluru
}

variable "instance_type" {
  # Free-plan eligible for accounts created on/after 2025-07-15 (verify in the EC2 console:
  # "Free tier eligible" filter). 2 vCPU / 8 GiB: Week 3 showed 2 vCPUs keep ~90% of a 4-vCPU
  # box's request throughput, and 8 GiB fits the 1 GiB model + 1 GiB prompt cache easily.
  type    = string
  default = "m7i-flex.large"
}

variable "allowed_cidrs" {
  description = "Who may reach the gateway on :8000 (your IP as x.x.x.x/32). The API key is still required."
  type        = list(string)
}

variable "alert_email" {
  description = "Receives alarm and budget emails (confirm the SNS subscription email AWS sends)."
  type        = string
}

variable "monthly_budget_usd" {
  type    = number
  default = 5
}

variable "repo_url" {
  type    = string
  default = "https://github.com/Rahulverma5/qwen-inference-optimization.git"
}

variable "git_ref" {
  description = "Commit or branch the instance checks out (pin a commit for reproducible deploys)."
  type        = string
  default     = "main"
}

variable "hf_model_repo" {
  description = "Hugging Face repo holding the GGUF, e.g. <user>/qwen3-1.7b-spider-sql-GGUF"
  type        = string
}

variable "hf_model_file" {
  type    = string
  default = "qwen3-1.7b-spider-Q4_K_M.gguf"
}

variable "hf_revision" {
  description = "HF commit hash (pin it) or branch"
  type        = string
  default     = "main"
}

variable "llama_threads" {
  type    = number
  default = 2 # = vCPUs of m7i-flex.large
}

variable "queue_wait_s" {
  description = "GATEWAY_MAX_QUEUE_WAIT_S; Week 4 suggests ~SLO (10 s) - measured model p50"
  type        = number
  default     = 5
}

variable "root_volume_gb" {
  type    = number
  default = 20 # free plan includes 30 GB of EBS
}
