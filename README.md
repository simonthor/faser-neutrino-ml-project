# Telling neutrinos apart from neutral hadrons in FASERnu

A starter project in data analysis and machine learning, using real simulated data from
the [FASER experiment](https://faser.web.cern.ch/) at CERN.

## The physics, briefly

FASER sits 480 m downstream of the ATLAS collision point at the LHC. Its neutrino
detector, **FASERnu**, is a stack of roughly 700 photographic films interleaved with
tungsten plates. Charged particles passing through leave tracks in the emulsion, which
are scanned and reconstructed in 3D.

When a neutrino strikes a tungsten nucleus it produces a spray of charged particles
starting from a single point. That point is a **vertex**; the particles coming out of it
are the **primary tracks**. Because the neutrino itself is electrically neutral, there is
no track leading *into* the vertex, it appears out of nowhere.

Examples of an electron neutrino interaction (top row, two different views) and a
muon neutrino interaction (bottom row) are shown below.

![Neutrino interactions](images/neutrino_interactions.png)

That last property is the problem. A **neutral hadron** (a neutron, $\Lambda$, 
or $K^0$) is also electrically neutral, also leaves no incoming track, and also produces a
spray of outgoing particles. These are produced in large numbers when muons from the LHC
interact inside the detector itself. Under a microscope the two look alike.

So every clue must come from the *outgoing* tracks: how many there are, how fast they
are, how they are arranged in space. That is what this dataset contains.

- **Signal**: a neutrino interaction
- **Background**: a neutral hadron interaction

Roughly speaking, in the real experiment the background outnumbers the signal about
8 to 1, which is what makes the problem worth solving.

## Getting started

### 1. Install Python and the packages

This project uses [uv](https://docs.astral.sh/uv/), which handles Python versions and
packages together. Install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # macOS / Linux
```

Then, from the project folder:

```bash
uv sync
```

That creates a virtual environment in `.venv/` and installs everything. A **virtual
environment** is a private copy of Python for this project, so packages you install here
will not break anything else on your computer.

<details>
<summary>Without uv (plain Python and pip)</summary>

```bash
python3 -m venv .venv               # create the environment

source .venv/bin/activate           # activate it (macOS / Linux)
.venv\Scripts\activate              # activate it (Windows)

pip install pandas numpy matplotlib scikit-learn jupyter
```

You need to run the `activate` line again in every new terminal. You can tell it worked
because your prompt gains a `(.venv)` prefix.
</details>

### 2. Get the data

`data/vertices.csv` is about 60 MB and is **not stored in this repository**. 
Download it from https://cernbox.cern.ch/s/7NNF2rwlMRuJg4y and place it in the `data/` folder.

### 3. Open the notebook

```bash
uv run jupyter lab explore.ipynb
```

If you are familiar with code editors, you can also open `explore.ipynb` in VS Code or PyCharm.

Run the cells from top to bottom with <kbd>Shift</kbd>+<kbd>Enter</kbd>.

## The data

One row is one reconstructed vertex. 268 898 rows, 33 columns.

| rows | |
|---|---|
| `background` | 162 949 |
| `numuCC` | 65 375 |
| `NC` | 22 630 |
| `nueCC` | 17 356 |
| `nutauCC` | 588 |

`CC` means **charged current**: the neutrino turned into its charged partner (a muon for
`numuCC`, an electron for `nueCC`, a tau for `nutauCC`), which leaves a track. `NC` means
**neutral current**: the neutrino bounced off the nucleus and continued invisibly, so
there is no lepton track and these are the hardest neutrinos to identify.

**The goal is to find all the CC interactions, which are the `numuCC`, `nueCC`, and
`nutauCC` rows.**

### Columns

Every column is built from the primary tracks of the vertex. The names in parentheses
are the branches of the original ROOT files they come from. An empty cell means the
quantity is undefined for that vertex. pandas reads empty cells as `NaN`. The last
column says when a cell is empty and how many of the 268 898 rows are affected.

| column | type | unit | meaning | empty when |
|---|---|---|---|---|
| `interaction` | string | — | What really happened. The label: `numuCC`, `nueCC`, `nutauCC`, `NC` or `background` (from `nuint_type`). | Never |
| `weight` | float | — | How many real interactions this one row represents, for 9.5 fb⁻¹ of LHC data. **Never use as a feature**, see below. | Never |
| `n_vtrk` | int | count | Primary tracks at the vertex (`n_vtrk`). | Never |
| `n_vtrk_100mrad` | int | count | Primary tracks within 100 mrad of the beam direction (`n_vtrk_100mrad`). | Never |
| `n_vtrk_ip5` | int | count | Primary tracks passing within 5 µm of the vertex (`n_vtrk_ip5`). | Never |
| `n_vtrk_ip5_100mrad` | int | count | Primary tracks within 5 µm of the vertex *and* 100 mrad of the beam (`n_vtrk_ip5_100mrad`). | Never |
| `slope_mean` | float | — | Average tan θ of the tracks relative to the beam (`slope_vtrk`). | Never |
| `slope_max` | float | — | Largest tan θ of the tracks. | Never |
| `slope_std` | float | — | Spread (standard deviation) of tan θ of the tracks. | Never, since every vertex has at least 2 tracks |
| `delta_phi_mean` | float | rad | For each track, the azimuthal angle between it and the other tracks added together, every track counting equally. Averaged over the tracks. Near π means back-to-back. | Never |
| `delta_phi_max` | float | rad | The largest of those angles. | Never |
| `delta_phi_p_mean` | float | rad | Like `delta_phi_mean`, but each track is weighted by its transverse momentum, so only tracks with a measured momentum take part. | Fewer than 2 tracks have a momentum (4 877 rows, 1.8%) |
| `delta_phi_p_max` | float | rad | The largest of those angles. | Same as `delta_phi_p_mean` |
| `p_max` | float | GeV | Largest momentum of the tracks, measured from multiple scattering (`Prec_vtrk`). | No track has a momentum (568 rows, 0.2%) |
| `p_sum` | float | GeV | Total momentum of the tracks that have one. | Same as `p_max` |
| `p_mean` | float | GeV | Average momentum of the tracks that have one. | Same as `p_max` |
| `n_tracks_with_momentum` | int | count | Tracks with a measured momentum. A track has none if it has fewer than 10 segments, if the fit failed, or if the fit hit its 7 TeV cap. | Never |
| `npl_max` | int | count | Plates crossed by the longest track (`npl_vtrk`). | Never |
| `ip_pos_mean` | float | µm | Average impact parameter of the tracks: how close they extrapolate back to the vertex (`ip_pos_vtrk`). | Never |
| `ip_pos_max` | float | µm | The largest impact parameter. | Never |
| `n_kinks` | int | count | Tracks with a sudden change of direction (`flag_kink_angle`). This is the signature of a particle decaying in flight. | Never |
| `kink_angle_max` | float | rad | The largest such change of direction (`kink_angle`). | `n_kinks` is 0 (1 369 rows, 0.5%) |
| `e_em_max` | float | GeV | Largest energy of the electromagnetic shower attached to a track (`E_em`). | Never in this file. It would be empty if no track had a shower. |
| `e_em_sum` | float | GeV | Total electromagnetic shower energy. Tracks in the same shower share one `E_em`, so a shared shower is counted once for each of its tracks. | Same as `e_em_max` |
| `dphi_unit` | float | rad | Angle between the leading track and the other tracks added together, every track counting equally. Near π means back-to-back. | Same as `e_em_max`, because the leading track needs an `E_em` |
| `dphi_p` | float | rad | The same, but each other track weighted by its momentum (the larger of `Prec_vtrk` and `Prec_par`). | No track other than the leading one has a momentum (2 997 rows, 1.1%) |
| `r90` | int | count | Other tracks pointing more than 90° away from the leading track. | Same as `dphi_unit` |
| `a_sum` | float | — | Length of the sum of all tracks' unit directions: how much the tracks point the same way. | Never |
| `pt_sum` | float | GeV | Length of the sum of `E_em` × direction over all tracks: how well the transverse momentum balances. | Same as `e_em_max` |
| `n_vtrk_ippos3um` | int | count | Tracks passing within 3 µm of the vertex (`ip_pos_vtrk`). | Never |
| `n_vtrk_nseg5` | int | count | Tracks with more than 5 segments (`nseg_vtrk`). | Never |
| `leadtrk_slope` | float | — | tan θ of the leading track. | Same as `dphi_unit` |
| `leadtrk_pt` | float | GeV | Transverse momentum of the leading track, `E_em` × slope. | Same as `dphi_unit` |

### Caveats

Read these before drawing conclusions.

- **`weight` gives away the answer.** It takes only nine values, and each belongs to
  exactly one class, because signal and background were simulated separately. A model
  given `weight` will score perfectly and have learned nothing. Use it only for scaling
  histograms to the real experiment.
- **A quarter to a third of tracks have no measured momentum.** Momentum is measured from
  how much a track scatters, which needs a long track. `n_tracks_with_momentum` records
  how many went into `p_mean`. Requiring *every* track to have one would be a mistake:
  it happens far more often in signal than background, so it would leak the answer too.
- **There are no vertex coordinates.** Signal and background were generated in different
  volumes, so position would identify the class outright.
- **`dphi_p` is missing when no other track has a momentum**: 0.6% of background rows
  and 1–4% of signal rows.
- **The two Δφ families use different tracks.** `delta_phi_*` uses every track;
  `delta_phi_p_*` uses only those with a measured momentum, and is missing entirely when
  fewer than two qualify (1.8% of rows).
- **Background has about twice the track multiplicity of signal.** This is the single
  strongest difference in the data. It is expected from the strong interaction being
  messier than the weak one, but how much of the gap is real physics and how much comes
  from the energies at which the neutral hadrons were simulated has not been checked.

## Documentation

- **pandas**: tables, `read_csv`, filtering, `groupby`: <https://pandas.pydata.org/docs/>
  - begin with [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- **NumPy**: arrays and numerical maths: <https://numpy.org/doc/stable/>
  - begin with [the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)
- **Matplotlib**: plotting: <https://matplotlib.org/stable/>
  - the [pyplot tutorial](https://matplotlib.org/stable/tutorials/pyplot.html) and the
  [example gallery](https://matplotlib.org/stable/gallery/index.html)
- **scikit-learn**: machine learning: <https://scikit-learn.org/stable/>
  - begin with [an introduction to machine learning](https://scikit-learn.org/stable/tutorial/basic/tutorial.html)

## For maintainers

`convert_root_to_csv.py` builds `data/vertices.csv` from the FASERnu reconstruction ROOT
files in `data/`. It needs those files plus the truth parquet files from the
`faser-nutau` repository, and takes about 15 seconds.

```bash
uv run python convert_root_to_csv.py
```

It applies two cuts:

1. **Fiducial volume**: the Takubo zones in x/y, plates 7–626 in z. This equalises the
   generation volumes of the two classes; it keeps 79% of signal and 98% of background.
2. **Incoming neutrino energy ≥ 200 GeV**, signal only, from the truth files. See
   [ADR 0002](docs/adr/0002-clean-signal-by-incoming-energy.md).

`geometry.py` and `weights.py` are copied from
[`fasernu-event-selection`](https://gitlab.cern.ch/) — `geometry.py` verbatim, `weights.py`
with its `print()` calls removed. Keep them in sync rather than editing them here.

Vocabulary used throughout is defined in [CONTEXT.md](CONTEXT.md).
