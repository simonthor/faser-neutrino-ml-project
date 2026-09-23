"""Convert FASERnu reconstruction ROOT files into a single flat CSV for teaching.

One row in the ROOT files is one *primary track*. One row in the output CSV is one
*reconstructed vertex*, with the track information collapsed into physics-motivated
aggregates. Signal is neutrino interactions (runs 200082/200083/200089), background is
neutral-hadron interactions (runs 100090-100095).

This script is for the maintainer, not for the student: it needs the ROOT files, the
truth parquet files from the faser-nutau repository, and FASERnu context. The student
only ever sees ``data/vertices.csv``.

Run with::

    uv run python convert_root_to_csv.py
"""

from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from geometry import COORDINATE_OFFSET, FIRST_PLATE, LAST_PLATE, ZONE_EDGES, vertices_in_zones
from weights import WEIGHTS

DATA_DIR = Path("data")
TRUTH_DIR = Path("../faser-nutau/data/cleaned-ntup-data")
OUTPUT = DATA_DIR / "vertices.csv"

SIGNAL_RUNS = (200082, 200083, 200089)
BACKGROUND_RUNS = (100090, 100091, 100092, 100093, 100094, 100095)

# Minimum true energy of the incoming neutrino, in MeV. Low-energy neutrinos are poorly
# reconstructed, so they are dropped from the signal sample. Neutral hadrons have no
# equivalent cut: their truth files are not used at all.
INCOMING_ENERGY_MIN_MEV = 200_000

# Momentum reconstruction failure sentinels. -999 means the fit failed, 6999.86 is the
# saturation cap. 100 does not occur in the current files but is filtered defensively.
MOMENTUM_MAX = 6999
MOMENTUM_SENTINEL = 100

BRANCHES = [
    "iev_mc", "x_vtx", "y_vtx", "nuint_type", "plate_vtrk",
    "n_vtrk", "n_vtrk_100mrad", "n_vtrk_ip5", "n_vtrk_ip5_100mrad",
    "slope_vtrk", "tx_vtrk", "ty_vtrk", "ip_pos_vtrk",
    "Prec_vtrk", "Prec_par", "Prec_dau", "npl_vtrk", "nseg_vtrk",
    "kink_angle", "flag_kink_angle", "E_em",
]


def interaction_label(nuint_type: int) -> str:
    """Map a ``nuint_type`` code to a readable interaction label.

    ``nuint_type`` is ``PDG * 10 + is_cc``, so the first digits carry the incoming
    particle and the last digit flags charged current. Neutrinos and antineutrinos are
    merged, as are the six neutral-hadron species.
    """
    pdg, is_cc = divmod(abs(int(nuint_type)), 10)
    if pdg > 100:
        return "background"
    if not is_cc:
        return "NC"
    return {12: "nueCC", 14: "numuCC", 16: "nutauCC"}[pdg]


def delta_phi(vx: pd.Series, vy: pd.Series, keys: pd.Series) -> pd.Series:
    """Azimuthal angle between each track and the vector sum of the other tracks.

    The hadronic system is the sum of the *other* tracks' transverse vectors, so each
    track is compared against everything else at its vertex. Returns 0 where either
    vector vanishes, matching the reference implementation in fasernu-event-selection.
    """
    had_x = vx.groupby(keys).transform("sum") - vx
    had_y = vy.groupby(keys).transform("sum") - vy
    difference = np.arctan2(vy, vx) - np.arctan2(had_y, had_x)
    angle = np.abs(np.arctan2(np.sin(difference), np.cos(difference)))
    degenerate = ((vx == 0) & (vy == 0)) | ((had_x == 0) & (had_y == 0))
    return angle.where(~degenerate, 0.0)


def read_tracks(run: int) -> pd.DataFrame:
    """Read one ROOT file into a track-level DataFrame with sentinels replaced by NaN."""
    (path,) = DATA_DIR.glob(f"result_vtrk_list_after_ks_{run}_*.root")
    with uproot.open(path) as file:
        # Each file holds two cycles of the same tree; the highest is the complete one
        # and the older has 1-8% fewer entries. uproot would pick it anyway, but the
        # choice is too consequential to leave implicit.
        cycle = max(int(key.split(";")[1]) for key in file.keys() if key.startswith("vtrk_ks_Tree"))
        tracks = ak.to_dataframe(file[f"vtrk_ks_Tree;{cycle}"].arrays(BRANCHES, library="ak"))

    for column in ["Prec_vtrk", "Prec_par", "Prec_dau"]:
        usable = (
            (tracks[column] > 0)
            & (tracks[column] < MOMENTUM_MAX)
            & (tracks[column] != MOMENTUM_SENTINEL)
        )
        tracks[column] = tracks[column].where(usable)

    # E_em uses -999 for "not reconstructed".
    tracks["E_em"] = tracks["E_em"].where(tracks["E_em"] >= 0)

    # kink_angle is exactly 0 whenever no kink was found, which is a sentinel rather
    # than a measurement of a very small angle.
    tracks["kink_angle"] = tracks["kink_angle"].where(tracks["flag_kink_angle"] == 1)

    return tracks


def select_vertices(tracks: pd.DataFrame, run: int) -> pd.DataFrame:
    """Drop tracks whose vertex fails the fiducial volume or incoming-energy cut."""
    keys = ["iev_mc", "x_vtx"]
    vertices = tracks.groupby(keys).agg(
        y_vtx=("y_vtx", "first"), vertex_plate=("plate_vtrk", "min")
    )

    # Vertex positions are in um relative to the film corner; the zone polygons are in
    # cm relative to the detector centre.
    positions = np.column_stack(
        [
            vertices.index.get_level_values("x_vtx").to_numpy() / 10_000 + COORDINATE_OFFSET[0],
            vertices["y_vtx"].to_numpy() / 10_000 + COORDINATE_OFFSET[1],
        ]
    )
    keep = vertices_in_zones(positions, ZONE_EDGES) & (
        vertices["vertex_plate"].between(FIRST_PLATE, LAST_PLATE).to_numpy()
    )

    if run in SIGNAL_RUNS:
        truth = ak.from_parquet(
            TRUTH_DIR / f"nu_events_MC{run}.parquet",
            # These files are 0.9-3.7 GB and the unread "primaries" field is nearly all
            # of it, so restricting the columns is what makes this affordable.
            columns=["event_id", "incoming_particle_energy"],
        )
        energy = pd.Series(
            ak.to_numpy(truth.incoming_particle_energy), index=ak.to_numpy(truth.event_id)
        )
        event_numbers = vertices.index.get_level_values("iev_mc").to_numpy()
        keep &= energy.reindex(event_numbers).to_numpy() >= INCOMING_ENERGY_MIN_MEV

    surviving = vertices.index[keep]
    return tracks[pd.MultiIndex.from_frame(tracks[keys]).isin(surviving)]


def build_features(tracks: pd.DataFrame, run: int) -> pd.DataFrame:
    """Collapse the surviving tracks into one row per vertex."""
    keys = pd.MultiIndex.from_frame(tracks[["iev_mc", "x_vtx"]])

    # Unweighted delta phi: every track contributes a unit vector, so the angle
    # describes the topology alone and is defined for every track.
    norm = np.hypot(tracks["tx_vtrk"], tracks["ty_vtrk"]).replace(0, np.nan)
    tracks["ux"] = (tracks["tx_vtrk"] / norm).fillna(0.0)
    tracks["uy"] = (tracks["ty_vtrk"] / norm).fillna(0.0)
    tracks["delta_phi"] = delta_phi(tracks["ux"], tracks["uy"], keys)

    # Momentum-weighted delta phi: each track contributes its transverse momentum, so
    # only tracks with a measured momentum take part and the angle is undefined for a
    # vertex with fewer than two of them.
    transverse = tracks["Prec_vtrk"] / np.sqrt(1 + tracks["tx_vtrk"] ** 2 + tracks["ty_vtrk"] ** 2)
    tracks["delta_phi_p"] = delta_phi(
        (transverse * tracks["tx_vtrk"]).fillna(0.0),
        (transverse * tracks["ty_vtrk"]).fillna(0.0),
        keys,
    )
    has_momentum = tracks["Prec_vtrk"].notna()
    tracks["delta_phi_p"] = tracks["delta_phi_p"].where(
        has_momentum & (has_momentum.groupby(keys).transform("sum") >= 2)
    )

    # Variables from the FASERnu nue/numu CC event selection (Matsukuma, collaboration
    # meeting 2026-07-15, docs/20260714_Ren_FASERnu_event_selection_FASER_CM.pdf). They
    # are built around the leading track, the one with the highest EM-shower energy, and
    # use E_em as the track energy. A vertex with no E_em at all has no leading track.
    # E_em belongs to a shower cluster, so collinear tracks in the same shower share it
    # and a third of vertices have a tie for the highest value. The slides do not say
    # how to break it; here the tied track starting in the most upstream plate wins,
    # then the one most back-to-back with the rest in (unweighted) delta phi.
    ux, uy = tracks["ux"], tracks["uy"]
    ranked = tracks.sort_values(
        ["E_em", "plate_vtrk", "delta_phi"],
        ascending=[False, True, False],
        na_position="last",
    )
    is_lead = pd.Series(False, index=tracks.index)
    is_lead[ranked.index[~ranked.duplicated(["iev_mc", "x_vtx"])]] = True
    is_lead &= tracks["E_em"].notna()
    tracks["dphi_unit"] = tracks["delta_phi"].where(is_lead)
    # dphi_p weights the other tracks by momentum, taking the parent segment's where it
    # is larger. The leading track's own weight cancels out of the sum of the others, so
    # it is set to 1 to keep its direction defined even without a measured momentum.
    # Undefined when none of the other tracks has a momentum.
    momentum = np.fmax(tracks["Prec_vtrk"], tracks["Prec_par"]).fillna(0.0).where(~is_lead, 1.0)
    tracks["dphi_p"] = delta_phi(momentum * ux, momentum * uy, keys).where(is_lead)
    others_with_momentum = (momentum.where(~is_lead, 0.0) > 0).groupby(keys).transform("sum")
    tracks["dphi_p"] = tracks["dphi_p"].where(others_with_momentum > 0)
    tracks["leadtrk_slope"] = tracks["slope_vtrk"].where(is_lead)
    tracks["leadtrk_pt"] = (tracks["E_em"] * tracks["slope_vtrk"]).where(is_lead)
    lead_ux = ux.where(is_lead).groupby(keys).transform("max")
    lead_uy = uy.where(is_lead).groupby(keys).transform("max")
    tracks["opposite_lead"] = (ux * lead_ux + uy * lead_uy) < 0
    tracks["ex"] = tracks["E_em"] * tracks["tx_vtrk"]
    tracks["ey"] = tracks["E_em"] * tracks["ty_vtrk"]
    tracks["ippos3um"] = tracks["ip_pos_vtrk"] < 3
    tracks["nseg5"] = tracks["nseg_vtrk"] > 5

    vertices = tracks.groupby(["iev_mc", "x_vtx"]).agg(
        nuint_type=("nuint_type", "first"),
        n_vtrk=("n_vtrk", "first"),
        n_vtrk_100mrad=("n_vtrk_100mrad", "first"),
        n_vtrk_ip5=("n_vtrk_ip5", "first"),
        n_vtrk_ip5_100mrad=("n_vtrk_ip5_100mrad", "first"),
        slope_mean=("slope_vtrk", "mean"),
        slope_max=("slope_vtrk", "max"),
        slope_std=("slope_vtrk", "std"),
        delta_phi_mean=("delta_phi", "mean"),
        delta_phi_max=("delta_phi", "max"),
        delta_phi_p_mean=("delta_phi_p", "mean"),
        delta_phi_p_max=("delta_phi_p", "max"),
        p_max=("Prec_vtrk", "max"),
        p_sum=("Prec_vtrk", "sum"),
        p_mean=("Prec_vtrk", "mean"),
        n_tracks_with_momentum=("Prec_vtrk", "count"),
        npl_max=("npl_vtrk", "max"),
        ip_pos_mean=("ip_pos_vtrk", "mean"),
        ip_pos_max=("ip_pos_vtrk", "max"),
        n_kinks=("flag_kink_angle", "sum"),
        kink_angle_max=("kink_angle", "max"),
        e_em_max=("E_em", "max"),
        e_em_sum=("E_em", "sum"),
        n_tracks_with_em=("E_em", "count"),
        dphi_unit=("dphi_unit", "max"),
        dphi_p=("dphi_p", "max"),
        r90=("opposite_lead", "sum"),
        ux_sum=("ux", "sum"),
        uy_sum=("uy", "sum"),
        ex_sum=("ex", "sum"),
        ey_sum=("ey", "sum"),
        n_vtrk_ippos3um=("ippos3um", "sum"),
        n_vtrk_nseg5=("nseg5", "sum"),
        leadtrk_slope=("leadtrk_slope", "max"),
        leadtrk_pt=("leadtrk_pt", "max"),
    )

    # A sum over an all-missing group is 0 in pandas, which would claim a measurement of
    # zero where there is no measurement at all.
    vertices["p_sum"] = vertices["p_sum"].where(vertices["n_tracks_with_momentum"] > 0)
    vertices["e_em_sum"] = vertices["e_em_sum"].where(vertices["n_tracks_with_em"] > 0)
    vertices["r90"] = vertices["r90"].where(vertices["dphi_unit"].notna())
    vertices["a_sum"] = np.hypot(vertices["ux_sum"], vertices["uy_sum"])
    vertices["pt_sum"] = np.hypot(vertices["ex_sum"], vertices["ey_sum"]).where(
        vertices["n_tracks_with_em"] > 0
    )

    vertices["interaction"] = vertices["nuint_type"].map(interaction_label)
    vertices["weight"] = WEIGHTS[run]
    helpers = ["nuint_type", "n_tracks_with_em", "ux_sum", "uy_sum", "ex_sum", "ey_sum"]
    return vertices.drop(columns=helpers).reset_index(drop=True)


def main() -> None:
    """Convert every ROOT file and write the combined vertex table."""
    frames = []
    for run in BACKGROUND_RUNS + SIGNAL_RUNS:
        tracks = select_vertices(read_tracks(run), run)
        vertices = build_features(tracks, run)
        print(f"{run}: {len(vertices):>6} vertices")
        frames.append(vertices)

    combined = pd.concat(frames, ignore_index=True)
    columns = ["interaction", "weight"] + [
        column for column in combined.columns if column not in ("interaction", "weight")
    ]
    combined[columns].to_csv(OUTPUT, index=False, float_format="%.6g")

    print(f"\nWrote {len(combined)} vertices to {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} MB)")
    print(combined["interaction"].value_counts().to_string())


if __name__ == "__main__":
    main()
