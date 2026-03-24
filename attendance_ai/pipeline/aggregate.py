"""Aggregation module: compile attendance results into structured output."""

import csv
import json
import io
import logging
from typing import Any

from attendance_ai.pipeline.classify_cell import CellLabel

logger = logging.getLogger(__name__)


class AttendanceAggregator:
    """Aggregates cell classification results into attendance summary."""

    def run(
        self,
        student_cells: list[dict],
        classifications: list[tuple[CellLabel, float]],
        latest_col_index: int,
        student_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Aggregate classification results for the latest column.

        Args:
            student_cells: list of student dicts from CellExtractor
            classifications: list of (label, confidence) for each student
            latest_col_index: which column was classified
            student_names: list of OCR-extracted names (optional)

        Returns:
            dict with:
                - students: list of {name, status, confidence}
                - summary: {should_attend, absent, present}
                - csv_string: CSV formatted output
                - json_summary: JSON formatted summary
        """
        logger.info("=== AGGREGATION ===")
        students = []
        absent_count = 0

        for i, (label, conf) in enumerate(classifications):
            if student_names and i < len(student_names):
                name = student_names[i]
            else:
                name = f"Student {i + 1}"

            status = "A" if label == CellLabel.A else "P"
            if label == CellLabel.A:
                absent_count += 1

            students.append({
                "name": name,
                "status": status,
                "confidence": round(conf, 3),
                "raw_label": label.value,
            })

        total = len(students)
        present = total - absent_count

        summary = {
            "should_attend": total,
            "absent": absent_count,
            "present": present,
        }

        csv_str = self._to_csv(students)
        json_str = json.dumps(summary, indent=2)

        logger.info(
            f"Results: {total} students, {absent_count} absent, {present} present"
        )

        return {
            "students": students,
            "summary": summary,
            "csv_string": csv_str,
            "json_summary": json_str,
            "latest_column_index": latest_col_index,
        }

    @staticmethod
    def _to_csv(students: list[dict]) -> str:
        """Generate CSV string from student results."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Student", "Status"])
        for s in students:
            writer.writerow([s["name"], s["status"]])
        return output.getvalue()
