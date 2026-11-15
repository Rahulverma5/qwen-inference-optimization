"""Prometheus metrics. One registry per app, so tests (and several apps in one process) don't
collide. Buckets are in seconds and sized for CPU serving: a request takes ~2-10 s."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 3, 4, 6, 8, 10, 15, 20, 30, 60)


class Metrics:
    def __init__(self) -> None:
        r = self.registry = CollectorRegistry()
        self.requests = Counter("gateway_requests", "Requests by route and HTTP status",
                                ["route", "status"], registry=r)
        self.latency = Histogram("gateway_request_seconds", "End-to-end latency at the gateway",
                                 ["route"], buckets=LATENCY_BUCKETS, registry=r)
        self.ttft = Histogram("gateway_ttft_seconds", "Time to first streamed byte from the backend",
                              ["route"], buckets=LATENCY_BUCKETS, registry=r)
        self.queue_wait = Histogram("gateway_queue_wait_seconds", "Time spent waiting for a backend slot",
                                    buckets=LATENCY_BUCKETS, registry=r)
        self.rejections = Counter("gateway_rejections", "Requests refused before reaching the backend",
                                  ["reason"], registry=r)
        self.backend_errors = Counter("gateway_backend_errors", "Backend failures by kind",
                                      ["kind"], registry=r)
        self.retries = Counter("gateway_backend_retries", "Retries after a connection failed before any byte",
                               registry=r)
        self.queue_depth = Gauge("gateway_queue_depth", "Requests waiting for a backend slot", registry=r)
        self.in_flight = Gauge("gateway_in_flight", "Requests currently at the backend", registry=r)
        self.ready = Gauge("gateway_ready", "1 if the last readiness check passed", registry=r)
