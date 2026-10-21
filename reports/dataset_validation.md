# Dataset validation

- real_sample_count: 101851
- feature_count: 36
- missing_values: {'trace_id': 0, 'timestamp_s': 0, 'source': 0, 'throughput_mbps': 0, 'latency_ms': 101851, 'loss_pct': 101851, 'jitter_ms': 101851, 'is_synthetic': 0, 'split': 0}
- duplicates: 0
- time_range_s: [0.0, 4207.0]
- unique_traces_by_source: {'real_world_traces': 587}
- split_trace_counts: {'train': 410, 'validation': 88, 'test': 89}
- zero_trace_overlap: True
- outliers_over_1000_mbps: 0
- field_availability: {'throughput_mbps': 101851, 'latency_ms': 0, 'loss_pct': 0, 'jitter_ms': 0}
- class_distribution: {'Stable': 46776, 'Improving': 26860, 'Degrading': 16618, 'Severely Degrading': 8662}