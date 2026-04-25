#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import SETTINGS
from backend.pipeline.formatter import _estimate_tokens, validate_markdown_output
from backend.pipeline.template_manager import TEMPLATE_MANAGER


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact manually evaluated MoM candidate for the long transcript.")
    parser.add_argument("--job-dir", type=Path, default=SETTINGS.jobs_dir / "2026-04-12-12:36-super_long_fix")
    parser.add_argument("--template-id", default="xlab_MoM")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _title(job_dir: Path) -> str:
    runtime_path = job_dir / "job_runtime.json"
    if runtime_path.exists():
        title = str((_load_json(runtime_path) or {}).get("title") or "").strip()
        if title:
            return title
    return job_dir.name


def _render(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "None"


def main() -> int:
    args = _parse_args()
    job_dir = args.job_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else job_dir / "compact_agenda_mom"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    title = _title(job_dir)
    participants = [
        "Deputy Mayor Glenfield",
        "Mayor Terry",
        "Councillor Kaczynski/Kuzinski",
        "Councillor Miller",
        "Councillor Warren",
        "Mr. Draper",
        "Ms. Holly Irvin-Nott",
        "Mr. Wohlman/Wolman",
        "Jason Tucker",
        "Matthew Braun and Nicole Duma, Oakwood Dairy Farms Limited",
        "Applicants, municipal staff, provincial reviewers, and public delegations",
        "Auto-assigned speaker aliases in transcript: apestos guy, hungarian, main guy, new guy, radio guy, woman 1",
    ]
    overview = [
        "The agenda was approved with one amendment, and the Qualico development agreement was discussed as an unsigned final draft targeted for the February planning meeting.",
        "Council heard and approved conditional use 2601 for Winnipeg Dig and Demolition at 216 Jean-Marc, with operational/site-cleanup conditions discussed.",
        "Council approved conditional use 2602 for a second dwelling near aggregate operations at 25022 Oakwood Road; the later vote records approval subject to three conditions.",
        "Council reviewed conditional use 25-05 for Ridgeland Colony livestock expansion, including the TRC report, water rights/domestic water concerns, manure management, avian flu/biosecurity, Cooks Creek impacts, shelter belts, and final resolution language.",
        "Council approved subdivision application 4189257850 for industrial property at Springfield Road and Oxford Street subject to four conditions.",
        "Council approved subdivision application 4189-25-7853 for an Oakwood Dairy Farms farmstead yard site subject to three conditions.",
    ]
    actions = [
        "Radio guy to provide council with the development agreement drafts and final draft for review before the February planning meeting.",
        "Winnipeg Dig and Demolition/applicant to provide missing paperwork, remove derelict vehicles and parts, relocate the refuse bin out of public view, and obtain applicable permits before operation.",
        "Municipal Planning Office to receive signed written correspondence from current owners before the conditional use 2602 land transfer.",
        "Applicant for Ridgeland Colony to submit the required drainage/grading plan by a professional engineer, obtain the development and building permits, and maintain the required 100-meter setback from major water bodies.",
        "Water stewardship/ECC to clarify farm and domestic water use assumptions, assess whether total water use exceeds licensing thresholds, and resolve the 33 vs. higher per-person domestic water-use values.",
        "Groundwater branch/producers to confirm the status of unknown or abandoned wells and complete required decommissioning where applicable.",
        "Council/RM to add shelter-belt enhancement as a condition and explore tree support through Cook's Creek Conservation District.",
        "Public Works to review and approve drainage/grading plans for the subdivision files and require digital geo-referenced subdivision copies in the required AutoCAD format.",
        "Developer for subdivision 4189-25-7853 to construct any required drainage improvements before building permits are issued.",
    ]
    conclusions = [
        "The agenda was approved unanimously.",
        "The Qualico development agreement was not signed; it was treated as a final draft pending presentation/review, with Qualico indicating no concerns but needing more time for internal executive review.",
        "Conditional use 2601 for Winnipeg Dig and Demolition was approved unanimously after discussion of the site, equipment storage, cleanup, refuse-bin location, and pending Chicago Auto Repair-related paperwork.",
        "Conditional use 2602 was granted for the property at 25022 Oakwood Road; the later vote records the final approval as subject to three conditions, superseding the earlier broader condition discussion where the exact condition count was inconsistent.",
        "For Ridgeland Colony conditional use 25-05, the TRC report found the proposed operation met provincial requirements, while council and public discussion focused on water use, wells, manure management, avian flu/biosecurity, Cooks Creek, and shelter-belt mitigation.",
        "The Ridgeland public hearing was closed and the resolution passed with shelter-belt enhancement added as a condition; earlier requests to defer were discussion points, not the final meeting outcome.",
        "The Ridgeland approval framework included requirements around provincial due diligence, drainage/grading, permits, setbacks from major water bodies, and shelter-belt enhancement.",
        "Subdivision application 4189257850 was approved subject to four conditions; the hearing was non-public and the applicant was expected to provide zoning variance documentation.",
        "Subdivision application 4189-25-7853 was approved unanimously subject to payment of fees, a professionally prepared drainage/grading plan approved by Public Works, and a digital AutoCAD subdivision copy.",
    ]
    open_points = [
        "Exact timing for Qualico final signing and whether the agreement will be re-signed or signed as the same agreement.",
        "Whether Chicago Auto Repair’s related application and missing paperwork will be completed for the February review.",
        "The exact final wording/details of the three conditions attached to conditional use 2602.",
        "Whether the Ridgeland water-use assessment correctly accounts for domestic use, livestock use, real meter data, and water-rights licensing thresholds.",
        "Current status of unknown or abandoned wells and how former homestead wells are accounted for near manure application or storage areas.",
        "Whether additional Cooks Creek water-quality testing or biosecurity/manure-management oversight will be required.",
        "How shelter-belt enhancement will be implemented and maintained.",
        "Whether subdivision 4189257850 requires environmental licensing or a zoning variance for the utility shed.",
    ]
    risks = [
        "Qualico signing delays could push the development agreement into the February planning meeting.",
        "Incomplete paperwork, derelict vehicles, refuse-bin placement, or code/fire inspection issues could delay or complicate Winnipeg Dig and Demolition operations.",
        "Disclosure and private-well liability conditions near aggregate operations could affect future land transfers or disputes.",
        "Incorrect domestic water-use assumptions could understate water-rights requirements or aquifer impacts for Ridgeland Colony.",
        "Unknown or improperly sealed wells, manure application, open lagoons, and Cooks Creek runoff could create groundwater or surface-water contamination risks.",
        "Expanded flock sizes and wildlife attraction around manure or food scraps could increase avian flu and biosecurity concerns.",
        "Shelter-belt implementation may be incomplete without clear ownership, funding, or conservation-district support.",
        "Subdivision approvals depend on drainage plans, water-rights requirements, possible environmental licensing, access constraints, and required digital submissions.",
    ]

    mom = "\n\n".join(
        [
            f"### Title: {title}",
            f"#### Participants:\n{_render(participants)}",
            f"#### Concise Overview:\n{_render(overview)}",
            f"#### TODO's:\n{_render(actions)}",
            f"#### CONCLUSIONS:\n{_render(conclusions)}",
            f"#### DECISION/OPEN POINTS:\n{_render(open_points)}",
            f"#### RISKS:\n{_render(risks)}",
        ]
    )
    template = TEMPLATE_MANAGER.load(args.template_id)
    evaluation = {
        "validation": validate_markdown_output(mom, template.sections),
        "estimated_tokens": _estimate_tokens(mom),
        "basis": "Manual compact pass over map notes, range reconciliation output, and transcript spot checks.",
    }
    (output_dir / "mom_compact.md").write_text(mom + "\n", encoding="utf-8")
    (output_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"output_dir={output_dir}")
    print(json.dumps(evaluation, indent=2))
    return 0 if bool(evaluation["validation"]["valid"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
