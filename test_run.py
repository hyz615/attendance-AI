"""Quick test script to diagnose pipeline on example image."""
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

from attendance_ai.main import process_attendance_sheet

result = process_attendance_sheet(
    "1774321122561.jpg",
    debug=True,
    debug_dir="debug_output/test_run2",
)

print("\n" + "=" * 60)
print("SUMMARY:", json.dumps(result["summary"], indent=2))
print("=" * 60)

for s in result.get("students", []):
    print(f"  {s['name']:30s}  {s['status']}  conf={s['confidence']}")

if "error" in result:
    print("ERROR:", result["error"])

print("\nTIMING:", json.dumps(result.get("timing", {}), indent=2))
