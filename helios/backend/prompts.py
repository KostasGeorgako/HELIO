"""
HELIO prompt library.

All prompt text comes from the data-science team's handoff document.
Kept verbatim where possible so prompt tuning that happened on their side
survives intact.
"""

PROMPT_SYSTEM = """You are HELIO, an expert land investment analyst specialising in \
hyperspectral satellite remote sensing. You work for a Greek real estate \
intelligence firm. You are knowledgeable, warm, and direct.

You have access to EnMAP hyperspectral data: 190 wavelength bands from \
418nm to 2445nm per pixel, covering VNIR and SWIR. This lets you detect \
soil chemistry, clay minerals, vegetation stress, moisture, and hidden \
subsurface anomalies invisible to standard photography.

Constraints on land use: sites cannot be used for residential buildings, \
apartment blocks, or retail shops. All other uses are permitted.

When you need to extract structured data from the user, you will be given \
explicit instructions. In those cases return ONLY valid JSON with no \
surrounding text. In all other cases speak naturally and warmly.

Always refer to the sites by their display names (Arkadia, Arkadia 2, \
Magnisia, Veroia), never by internal keys."""


def prompt_welcome(sites_summary: str) -> str:
    """User message for PROMPT_WELCOME."""
    return f"""The user has just uploaded hyperspectral satellite data.
The following sites were detected:

{sites_summary}

Write a warm welcome message (3-5 sentences). Acknowledge the specific \
sites you can see. Then ask the user which sites they want to include in \
the investment comparison, and what the asking price is for each site. \
Make it feel like a knowledgeable colleague starting a consultation, \
not a form. Do not list technical band counts or file names."""


def prompt_parse_selection(user_message: str) -> str:
    return f"""Extract the site selection, prices, and (if mentioned) preferred \
acquisition dates from this message.

CRITICAL DISTINCTION — these are FOUR SEPARATE SITES, not three:
  • arkadia    → display name "Arkadia"     (one specific parcel)
  • arkadia2   → display name "Arkadia 2"   (a DIFFERENT parcel — same region, different location)
  • magnisia   → display name "Magnisia"
  • veroia     → display name "Veroia"

NEVER collapse "Arkadia" and "Arkadia 2" together. If the user says \
"arkadia AND arkadia 2" or "arkadia 2", that is the second site (arkadia2), \
distinct from the first. The number matters.

The user may mention specific acquisition dates in YYYYMMDD form \
(e.g. "20241024" or "for arkadia 2 it's 20240531"). Capture those per site.

User message: "{user_message}"

Return ONLY this JSON. Include EVERY site the user mentioned. If price is not \
stated, set it to null. If a date is not stated for a site, set it to null.

{{
  "sites_selected": ["arkadia", "arkadia2", "magnisia", "veroia"],
  "prices": {{ "arkadia": 1000000, "arkadia2": 1000000, "magnisia": 1000000, "veroia": 1000000 }},
  "preferred_dates": {{ "arkadia": "20241024", "arkadia2": "20240531", "magnisia": "20241024", "veroia": "20250821" }},
  "missing_prices": [],
  "needs_clarification": false,
  "clarification_needed": ""
}}"""


def prompt_selection_confirm(parsed_selection_json: str) -> str:
    return f"""The user selected these sites, prices, and preferred acquisition dates:
{parsed_selection_json}

Write a warm 2-3 sentence confirmation that:
  1. Lists EVERY site the user named (use display names: Arkadia, Arkadia 2, \
Magnisia, Veroia — Arkadia and Arkadia 2 are TWO DIFFERENT parcels, never merge them).
  2. States each site's price.
  3. If specific acquisition dates were mentioned, briefly acknowledge them \
(e.g. "noted your preference for the October 2024 scene on Arkadia").

Then ask what they intend to use the land for. Give 3-4 examples (e.g. olive \
groves, solar farm, agritourism, mineral extraction) to help them think, but \
accept any answer."""


def prompt_parse_usecase(user_message: str) -> str:
    return f"""A land investor stated their intended use as:
"{user_message}"

You must assign scoring weights for hyperspectral analysis.

Available axes (all normalised 0-1 within the compared sites):

W_SOIL:     soil quality — low clay, low bare soil, low salinity.
            Good for: agriculture, horticulture, agritourism.
W_CLAY:     clay richness — HIGH clay = better.
            Good for: clay excavation ONLY. Set to 0 for everything else.
W_MINERAL:  mineral interest — iron oxides + carbonates.
            Good for: mineral prospecting, quarrying.
W_CONSIST:  spatial uniformity of the terrain.
            Good for: solar farms, logistics, any large-scale uniform use.
W_VEG:      vegetation/biomass (seasonally adjusted).
            Good for: agriculture, agritourism, reforestation.
W_MOISTURE: canopy and soil moisture signal.
            Good for: intensive crops, horticulture, wetland restoration.
W_ANOMALY:  spectral anomaly burden weight.
            Always include at minimum 0.05.

anomaly_sign: -1 means anomalies are BAD (contamination risk).
              +1 means anomalies are GOOD (mineral deposits present).
              Use +1 ONLY for mineral extraction or prospecting.

date_discounts: May acquisitions get 0.65-0.75 (growing season inflates \
vegetation). October and August get 1.0 (baseline and drought signal).

All weights must sum to exactly 1.0.

Return ONLY this JSON:
{{
  "use_case": "interpreted label, max 4 words",
  "reasoning": "2-3 sentences explaining weight choices",
  "weights": {{
    "W_SOIL": 0.0, "W_CLAY": 0.0, "W_MINERAL": 0.0,
    "W_CONSIST": 0.0, "W_VEG": 0.0, "W_MOISTURE": 0.0, "W_ANOMALY": 0.0
  }},
  "anomaly_sign": -1,
  "date_discounts": {{
    "arkadia": 1.0, "arkadia2": 0.70, "magnisia": 1.0, "veroia": 1.0
  }}
}}"""


def prompt_usecase_confirm(use_case: str, reasoning: str) -> str:
    return f"""You interpreted the use case as: {use_case}
Your reasoning: {reasoning}

Write 2-3 sentences confirming what you understood and what you are \
about to analyse. Sound confident. Mention that you are now running \
the full satellite analysis. Do not mention weights or numbers."""


def prompt_narrate(use_case: str, ranking_display: str, per_site_summary: str,
                   winner: str, sensitivity_pct: float,
                   active_factors: str = "", anomaly_direction: str = "") -> str:
    return f"""The hyperspectral analysis is complete. Here are the results:

Use case: {use_case}
Ranking: {ranking_display}

Per-site scores:
{per_site_summary}

Scoring focus for this analysis (use these in your reasoning — do NOT cite \
metrics that weren't weighted):
  Active factors  : {active_factors or 'all factors balanced'}
  Anomaly handling: {anomaly_direction or 'anomalies penalised as risk'}

Sensitivity analysis: {winner} ranked #1 in {sensitivity_pct}% of all weight \
combinations tested.

Write a 4-6 sentence investment recommendation addressed directly to \
the investor. State clearly which site to buy and why.

CRITICAL: only justify the recommendation using the ACTIVE FACTORS listed above. \
For example, if the analysis was weighted on spatial consistency (solar farm \
use case), do NOT praise the winner for soil quality or vegetation — those \
weren't being scored. Refer to the metrics by their physical meaning \
(e.g. "spatial uniformity of the terrain" for consistency, "subsurface anomaly \
burden" for the anomaly score). End with one sentence on risk. Be direct. \
Do not hedge excessively. Do not use bullet points — write flowing prose."""
