import requests
import json

KEY  = "sk-free-124ce0ee9e41ee22a30d3057"
BASE = "http://localhost:8000"
HDR  = {"X-API-Key": KEY}
SEP  = "=" * 62


def detect(path, conf=None):
    with open(path, "rb") as f:
        data = {"conf": str(conf)} if conf else {}
        r = requests.post(
            f"{BASE}/api/v1/vision/detect",
            headers=HDR,
            files={"file": (path.split("/")[-1], f, "image/jpeg")},
            data=data,
        )
    return r.json()


def show(d, title):
    print()
    print(SEP)
    print(f"  {title}")
    print(SEP)
    print(f"  Model  : {d['model']}")
    print(f"  Size   : {d['image_width']} x {d['image_height']} px")
    print(f"  Speed  : {d['execution_time_ms']} ms")
    print(f"  Found  : {d['object_count']} objects")
    print()
    print(f"  {'#':<4}{'Label':<16}{'Conf':<9} Bounding Box")
    print("  " + "-" * 54)
    for i, det in enumerate(sorted(d["detections"], key=lambda x: -x["confidence"]), 1):
        b = det["box"]
        print(
            f"  {i:<4}{det['label']:<16}{det['confidence']:.4f}  "
            f"({b['x_min']:.0f},{b['y_min']:.0f}) -> ({b['x_max']:.0f},{b['y_max']:.0f})"
        )


# ── Run detections ────────────────────────────────────────────
r1 = detect("/tmp/bus.jpg")
r2 = detect("/tmp/zidane.jpg")
r3 = detect("/tmp/bus.jpg", conf=0.10)

show(r1, "TEST 1 -- bus.jpg  [street scene: people + vehicles]")
show(r2, "TEST 2 -- zidane.jpg  [person close-up + tie]")

# ── Confidence comparison ─────────────────────────────────────
print()
print(SEP)
print("  TEST 3 -- Confidence Threshold Comparison  (bus.jpg)")
print(SEP)
print(f"  conf >= 0.25  -->  {r1['object_count']} objects  (default)")
print(f"  conf >= 0.10  -->  {r3['object_count']} objects  (+{r3['object_count'] - r1['object_count']} extra low-confidence hits)")
extra = [d for d in r3["detections"] if d["confidence"] < 0.25]
if extra:
    print("  Low-conf extras:")
    for d in extra:
        b = d["box"]
        print(f"    - {d['label']:<12} conf={d['confidence']:.4f}  ({b['x_min']:.0f},{b['y_min']:.0f})->({b['x_max']:.0f},{b['y_max']:.0f})")

# ── Auth tests ────────────────────────────────────────────────
print()
print(SEP)
print("  TEST 4 -- Auth checks")
print(SEP)
r_no_key  = requests.post(f"{BASE}/api/v1/vision/detect")
r_bad_key = requests.post(f"{BASE}/api/v1/vision/detect", headers={"X-API-Key": "sk-fake-000"})
print(f"  No API key   -> HTTP {r_no_key.status_code}  (expected 401)")
print(f"  Bad API key  -> HTTP {r_bad_key.status_code}  (expected 403)")

# ── /healthz + /metrics ───────────────────────────────────────
h = requests.get(f"{BASE}/healthz").json()
m = requests.get(f"{BASE}/metrics").json()
print()
print(SEP)
print("  TEST 5 -- /healthz + /metrics  (live traffic counters)")
print(SEP)
print(f"  Status   : {h['status']}  |  model={h['model_path']}  |  uptime={h['uptime_seconds']}s")
print(f"  Requests : total={m['total_requests']}  success={m['successful_requests']}  errors={m['failed_requests']}")
print(f"  Latency  : avg={m['avg_latency_ms']} ms")
print(f"  Tiers    : {json.dumps(m['tier_breakdown'])}")
print()
print("ALL 5 TESTS COMPLETE")
