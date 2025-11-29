"""Streamlit component for rate analysis explorer and builder.

The widget organises the information that was shared as scanned schedules into a
structured dataset so the user can browse, sanity-check, and reuse the values
while also supporting quick what-if studies (change overhead %, duplicate an
item, download JSON, etc.).

The data entered here is intentionally explicit rather than generated on the
fly so it is easy for the user to audit/extend.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

import json


import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Data model helpers
# ---------------------------------------------------------------------------


@dataclass
class LineItem:
    """Single entry inside a rate analysis table."""

    description: str
    unit: str
    quantity: Optional[float]
    rate: Optional[float]
    amount: Optional[float]
    notes: str = ""

    def to_row(self) -> Dict[str, Any]:
        return {
            "Description": self.description,
            "Unit": self.unit,
            "Quantity": self.quantity,
            "Rate (Rs)": self.rate,
            "Amount (Rs)": self.amount,
            "Notes": self.notes or "",
        }


@dataclass
class Section:
    name: str
    rows: List[LineItem] = field(default_factory=list)
    total_label: str = "Total"
    explicit_total: Optional[float] = None

    def total(self) -> Optional[float]:
        if self.explicit_total is not None:
            return self.explicit_total
        amounts = [row.amount for row in self.rows if row.amount is not None]
        if not amounts:
            return None
        return float(sum(amounts))

    def dataframe(self) -> pd.DataFrame:
        data = [row.to_row() for row in self.rows]
        df = pd.DataFrame(data)
        return df


@dataclass
class RateAnalysis:
    code: str
    title: str
    unit_label: str
    unit_quantity: float
    rate_per_unit: Optional[float]
    description: str
    sections: List[Section]
    overhead_percent: float = 13.615
    notes: List[str] = field(default_factory=list)

    def base_cost(self) -> Optional[float]:
        totals = [sec.total() for sec in self.sections if sec.total() is not None]
        if len(totals) != len(self.sections):
            return None
        return float(sum(totals))

    def total_with_overhead(self) -> Optional[float]:
        base = self.base_cost()
        if base is None:
            return None
        return base * (1 + self.overhead_percent / 100.0)

    def rate_with_overhead(self) -> Optional[float]:
        total = self.total_with_overhead()
        if total is None or self.unit_quantity == 0:
            return None
        return total / self.unit_quantity

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["sections"] = [
            {
                "name": sec.name,
                "total_label": sec.total_label,
                "explicit_total": sec.explicit_total,
                "rows": [row.to_row() for row in sec.rows],
            }
            for sec in self.sections
        ]
        return payload


# ---------------------------------------------------------------------------
# Raw data from the supplied schedules (transcribed manually)
# ---------------------------------------------------------------------------


RATE_ANALYSES: Dict[str, RateAnalysis] = {
    "IRR-CAW1-2": RateAnalysis(
        code="IRR-CAW1-2",
        title="Excavation in all kinds of soil including boulders up to 0.30 m dia",
        unit_label="cu.m",
        unit_quantity=440.0,
        rate_per_unit=440.00,
        description=(
            "Excavation for field channels and embankments in soils with boulders"
            " up to 0.30 m dia, including dressing to profile, mechanical mixing for"
            " embankment and placing excavated stuff for service roads within 10 m"
            " initial lead and 3 m lift."
        ),
        sections=[
            Section(
                name="Materials",
                rows=[
                    LineItem("Excavated soil (reused)", "cu.m", 440.00, 0.00, 0.00),
                ],
                total_label="Total cost of Materials",
                explicit_total=0.0,
            ),
            Section(
                name="Machinery",
                rows=[
                    LineItem("Shovel 0.50 cum capacity", "Hour", 32.00, 704.00, 22528.00),
                    LineItem("Crew for shovel", "Hour", 32.00, 183.00, 5856.00),
                    LineItem("Tipper 5 cum capacity", "Hour", 16.00, 847.00, 13552.00),
                    LineItem("Crew for tipper", "Hour", 16.00, 203.00, 3248.00),
                    LineItem("Diesel pump set 5.0 cum/hr", "Hour", 32.00, 84.00, 2688.00),
                    LineItem("Fuel / Energy charges", "Hour", 32.00, 337.00, 10784.00),
                ],
                total_label="Total hire charges of Machinery",
                explicit_total=58656.00,
            ),
            Section(
                name="Labour",
                rows=[
                    LineItem("Canal dressing crew", "Hour", 32.00, None, None,
                             notes="Fill in as per schedule — dressing, watering, compacting"),
                    LineItem("Embankment helpers", "Hour", 64.00, None, None),
                    LineItem("Mate / supervisor", "Hour", 16.00, None, None),
                    LineItem("Watchman / traffic control", "Hour", 16.00, None, None),
                ],
                total_label="Total cost of Labour",
            ),
        ],
        notes=[
            "Values transcribed from schedule IRR-CAW1-2; labour breakup to be keyed in",
            "Use the machinery hire reference table for verifying hour rates.",
        ],
    ),
    "RR-CAW1-6": RateAnalysis(
        code="RR-CAW1-6",
        title="Excavation in hard rock of all toughness by blasting",
        unit_label="cu.m",
        unit_quantity=68.0,
        rate_per_unit=761.35,
        description=(
            "Excavation in hard rock (boulders above 1.2 m dia) for canals,"
            " filter drains, catch water drains etc., including drilling,"
            " blasting, removal of rock projections by hammering/chiselling"
            " and depositing the excavated rock in an approved dump area with"
            " initial lead up to 10 m and all lifts."
        ),
        sections=[
            Section(
                name="Materials",
                rows=[
                    LineItem("Use rate of drill (1.5 m length)", "Rm", 11.30, 28.84, 326.87),
                    LineItem("Reconditioning charges", "Rm", 11.30, 11.26, 127.24),
                    LineItem("Detonators 25 mm dia @ 10 Nos", "Each", 2.80, 52.00, 145.60),
                    LineItem("Fuse 4 mm dia (2 m per hole)", "Rm", 13.50, 8.43, 113.81),
                    LineItem("Explosive small dia (Powder 20 g)", "Kg", 5.60, 315.00, 1764.00),
                    LineItem("Carbide for sharpening bits", "Kg", 1.80, 178.00, 320.40),
                    LineItem("Gelatine stick (75% strength)", "Kg", 5.60, 388.00, 2172.80),
                    LineItem("Energy charges", "Hour", 5.60, 337.00, 1887.20),
                ],
                total_label="Total cost of Materials",
                explicit_total=6857.92,
            ),
            Section(
                name="Machinery",
                rows=[
                    LineItem("Shovel 0.85 cum capacity", "Hour", 8.00, 784.00, 6272.00),
                    LineItem("Crew for shovel", "Hour", 8.00, 183.00, 1464.00),
                    LineItem("Dozer D50", "Hour", 2.00, 1426.00, 2852.00),
                    LineItem("Crew for dozer", "Hour", 2.00, 203.00, 406.00),
                    LineItem("Tipper 5 cum capacity (4 Nos)", "Hour", 4.00, 847.00, 3388.00),
                    LineItem("Crew for tipper (4 Nos)", "Hour", 4.00, 203.00, 812.00),
                    LineItem("Air compressor 450 cfm", "Hour", 4.00, 1284.00, 5136.00),
                    LineItem("Crew for compressor", "Hour", 4.00, 254.00, 1016.00),
                    LineItem("Pneumatic drill (2 Nos)", "Hour", 4.00, 220.00, 880.00),
                    LineItem("Crew for drill", "Hour", 4.00, 200.00, 800.00),
                    LineItem("Fuel / Energy charges", "Hour", 4.00, 337.00, 1348.00),
                ],
                total_label="Total hire charges of Machinery",
                explicit_total=24374.00,
            ),
            Section(
                name="Labour",
                rows=[
                    LineItem("Crew for shovel", "Hour", 8.00, 183.00, 1464.00),
                    LineItem("Crew for tipper", "Hour", 4.00, 203.00, 812.00),
                    LineItem("Helpers (unskilled)", "Hour", 12.00, 160.00, 1920.00,
                             notes="Includes mucking and channel cleaning"),
                    LineItem("Mate", "Hour", 4.00, 200.00, 800.00),
                    LineItem("Watchman", "Hour", 4.00, 200.00, 800.00),
                    LineItem("Sirdar", "Hour", 4.00, 220.00, 880.00),
                    LineItem("Blaster", "Hour", 4.00, 220.00, 880.00),
                    LineItem("Waterman", "Hour", 4.00, 160.00, 640.00),
                    LineItem("Mazdoor (unskilled)", "Hour", 8.00, 160.00, 1280.00),
                    LineItem("Foreman", "Hour", 4.00, 220.00, 880.00),
                ],
                total_label="Total cost of Labour",
                explicit_total=10356.00,
            ),
        ],
        notes=[
            "Notes from schedule:",
            "1) Adopt for canals carrying <15 cumecs or where average depth of hard rock"
            " excavation is <3 m.",
            "2) Includes levelling canal bed by hammering/chiselling rock projections.",
        ],
    ),
    "IRR-PMW1-1": RateAnalysis(
        code="IRR-PMW1-1",
        title="Clearing thin jungle growth (more than 50% open space)",
        unit_label="sqm",
        unit_quantity=1000.0,
        rate_per_unit=2.38,
        description=(
            "Removal of thin jungle growth including parthenium, bushes up to 0.3 m"
            " girth and disposal/burning as directed."
        ),
        sections=[
            Section(name="Materials", rows=[], total_label="Total cost of Materials", explicit_total=0.0),
            Section(name="Machinery", rows=[], total_label="Total cost of Machinery", explicit_total=0.0),
            Section(
                name="Labour",
                rows=[
                    LineItem("Work inspector", "Day", 1.00, 266.79, 266.79),
                    LineItem("Mazdoor", "Day", 12.00, 152.00, 1824.00),
                ],
                total_label="Total cost of Labour",
                explicit_total=2090.79,
            ),
        ],
    ),
}


# ---------------------------------------------------------------------------
# Reference tables (lead charges etc.)
# ---------------------------------------------------------------------------


HEAD_LOAD_LEAD = pd.DataFrame(
    [
        {
            "Total lead": "Lead upto 50 m (covered by initial lead)",
            "Earth/Sand/etc. (Rs per cum)": 15.85,
            "Cement/Reinf. steel (Rs per tonne)": 49.58,
            "PCC slab / CC block / Stone / Wood (Rs per cum)": 108.18,
        },
        {
            "Total lead": "Lead upto 100 m",
            "Earth/Sand/etc. (Rs per cum)": 31.65,
            "Cement/Reinf. steel (Rs per tonne)": 99.05,
            "PCC slab / CC block / Stone / Wood (Rs per cum)": 216.36,
        },
        {
            "Total lead": "Lead upto 150 m",
            "Earth/Sand/etc. (Rs per cum)": 47.50,
            "Cement/Reinf. steel (Rs per tonne)": 148.55,
            "PCC slab / CC block / Stone / Wood (Rs per cum)": 324.54,
        },
        {
            "Total lead": "Lead upto 200 m",
            "Earth/Sand/etc. (Rs per cum)": 63.35,
            "Cement/Reinf. steel (Rs per tonne)": 198.10,
            "PCC slab / CC block / Stone / Wood (Rs per cum)": 432.72,
        },
    ]
)


TRUCK_LEAD = pd.DataFrame(
    [
        {
            "Distance": "Lead upto 1 km",
            "Earth/Sand/etc. (Rs per cum)": 26.75,
            "Cement/Steel/Packed (Rs per tonne)": 84.00,
            "RCC/CC blocks etc. (Rs per cum)": 183.75,
            "Wood (Rs per cum)": 20.30,
        },
        {
            "Distance": "Lead upto 2 km",
            "Earth/Sand/etc. (Rs per cum)": 53.50,
            "Cement/Steel/Packed (Rs per tonne)": 168.00,
            "RCC/CC blocks etc. (Rs per cum)": 367.50,
            "Wood (Rs per cum)": 40.60,
        },
        {
            "Distance": "Lead upto 3 km",
            "Earth/Sand/etc. (Rs per cum)": 80.25,
            "Cement/Steel/Packed (Rs per tonne)": 252.00,
            "RCC/CC blocks etc. (Rs per cum)": 551.25,
            "Wood (Rs per cum)": 60.90,
        },
        {
            "Distance": "Lead upto 4 km",
            "Earth/Sand/etc. (Rs per cum)": 107.00,
            "Cement/Steel/Packed (Rs per tonne)": 336.00,
            "RCC/CC blocks etc. (Rs per cum)": 735.00,
            "Wood (Rs per cum)": 81.20,
        },
        {
            "Distance": "Lead upto 5 km",
            "Earth/Sand/etc. (Rs per cum)": 133.75,
            "Cement/Steel/Packed (Rs per tonne)": 420.00,
            "RCC/CC blocks etc. (Rs per cum)": 918.75,
            "Wood (Rs per cum)": 101.50,
        },
        {
            "Distance": "Lead upto 6 km",
            "Earth/Sand/etc. (Rs per cum)": 160.50,
            "Cement/Steel/Packed (Rs per tonne)": 504.00,
            "RCC/CC blocks etc. (Rs per cum)": 1102.50,
            "Wood (Rs per cum)": 121.80,
        },
        {
            "Distance": "Lead upto 7 km",
            "Earth/Sand/etc. (Rs per cum)": 187.25,
            "Cement/Steel/Packed (Rs per tonne)": 588.00,
            "RCC/CC blocks etc. (Rs per cum)": 1286.25,
            "Wood (Rs per cum)": 142.10,
        },
        {
            "Distance": "Lead upto 8 km",
            "Earth/Sand/etc. (Rs per cum)": 214.00,
            "Cement/Steel/Packed (Rs per tonne)": 672.00,
            "RCC/CC blocks etc. (Rs per cum)": 1470.00,
            "Wood (Rs per cum)": 162.40,
        },
        {
            "Distance": "Lead upto 9 km",
            "Earth/Sand/etc. (Rs per cum)": 240.75,
            "Cement/Steel/Packed (Rs per tonne)": 756.00,
            "RCC/CC blocks etc. (Rs per cum)": 1653.75,
            "Wood (Rs per cum)": 182.70,
        },
        {
            "Distance": "Lead upto 10 km",
            "Earth/Sand/etc. (Rs per cum)": 267.50,
            "Cement/Steel/Packed (Rs per tonne)": 840.00,
            "RCC/CC blocks etc. (Rs per cum)": 1837.50,
            "Wood (Rs per cum)": 203.00,
        },
        {
            "Distance": "For every km beyond 10 km",
            "Earth/Sand/etc. (Rs per cum)": 26.75,
            "Cement/Steel/Packed (Rs per tonne)": 84.00,
            "RCC/CC blocks etc. (Rs per cum)": 183.75,
            "Wood (Rs per cum)": 20.30,
        },
    ]
)


# ---------------------------------------------------------------------------
# UI logic
# ---------------------------------------------------------------------------


def render_rate(analysis: RateAnalysis) -> None:
    st.subheader(f"{analysis.code} · {analysis.title}")
    st.caption(analysis.description)

    default_overhead = analysis.overhead_percent
    overhead = st.slider(
        "Overhead / contractor's profit (%)",
        min_value=0.0,
        max_value=25.0,
        value=float(default_overhead),
        step=0.1,
        key=f"overhead_{analysis.code}",
    )

    base_cost = analysis.base_cost()
    total_cost = None if base_cost is None else base_cost * (1 + overhead / 100.0)
    rate = None if total_cost is None else total_cost / analysis.unit_quantity

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Quantity analysed", f"{analysis.unit_quantity:.2f} {analysis.unit_label}")
    with c2:
        if total_cost is not None:
            st.metric("Total (incl. overhead)", f"₹ {total_cost:,.2f}")
        else:
            st.metric("Total (incl. overhead)", "-")
    with c3:
        if rate is not None:
            st.metric("Rate per unit", f"₹ {rate:,.2f} / {analysis.unit_label}")
        elif analysis.rate_per_unit is not None:
            st.metric("Rate per unit (from schedule)", f"₹ {analysis.rate_per_unit:.2f}")
        else:
            st.metric("Rate per unit", "-")

    for section in analysis.sections:
        st.markdown(f"#### {section.name}")
        df = section.dataframe()
        st.dataframe(df, hide_index=True, use_container_width=True)
        total = section.total()
        if total is not None:
            st.write(f"**{section.total_label}: ₹ {total:,.2f}**")
        else:
            st.write(f"**{section.total_label}:** data entry required")

    if analysis.notes:
        st.markdown("**Notes**")
        for note in analysis.notes:
            st.markdown(f"- {note}")

    payload = analysis.to_dict()
    payload["overhead_percent"] = overhead
    st.download_button(
        label="⬇️ Download JSON snapshot",
        data=json.dumps(payload, indent=2),
        file_name=f"{analysis.code}_rate_analysis.json",
        mime="application/json",
        key=f"download_{analysis.code}",
    )


def render_reference_tables() -> None:
    st.markdown("### Conveyance reference (Head load)")
    st.dataframe(HEAD_LOAD_LEAD, hide_index=True, use_container_width=True)
    st.info(
        "No loading/unloading charges are admissible for the above head-load"
        " leads. Beyond 150 m, adopt mechanical means as per schedule notes."
    )

    st.markdown("### Conveyance reference (Trucks / Tippers)")
    st.dataframe(TRUCK_LEAD, hide_index=True, use_container_width=True)
    st.caption(
        "Charges exclude loading/unloading and idle hire of machinery."
        " For distances beyond 10 km add the incremental row per km."
    )


def run(state: Dict[str, Any]) -> None:
    st.title("📊 Rate Analysis Explorer")
    st.write(
        "Browse and interact with the rate analyses transcribed from the shared"
        " schedules. Adjust the contractor's overhead, export the data or use"
        " the reference conveyance tables while preparing new items."
    )

    codes = sorted(RATE_ANALYSES.keys())
    selected_code = st.selectbox("Select an analysis", codes)
    analysis = RATE_ANALYSES[selected_code]

    render_rate(analysis)

    st.markdown("---")
    with st.expander("Reference conveyance schedules"):
        render_reference_tables()


def register() -> Dict[str, Any]:
    return {
        "name": "Rate Analysis",
        "key": "rate_analysis",
        "icon": "🧮",
        "run": run,
    }
