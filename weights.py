import numpy as np

# Fraction of fiducial volume that is removed due to subvolumes with low base-track efficiency
rescale_factor = 681081 / 731336
PLATE_MAX_NOMINAL = 626
PLATE_MIN_NOMINAL = 7

# Fraction of simulated neutral hadrons that are inside the fiducial volume
# See neutral_hadron_locations.py for details
nh_fiducial_fraction = 1 - 0.013

n_muons_per_ifb_cm2 = 15650
muminus_fraction = 10.06 / (4.17 + 10.06)
muplus_fraction = 4.17 / (4.17 + 10.06)

# per 5 M muons                                        # Xi and Xibar neutral hadrons
n_nh_from_muminus = [28, 27, 37, 17, 12, 5]  # , 3, 3
n_nh_from_muplus = [42, 54, 75, 30, 21, 16]  # , 5, 1

n_muplus_per_ifb_cm2 = n_muons_per_ifb_cm2 * muplus_fraction
n_muminus_per_ifb_cm2 = n_muons_per_ifb_cm2 * muminus_fraction

F222_ifb = 9.5
F222_area_cm2 = 562.263  # From zone_areas.py

n_nh_muminus = [
    n_nh_from_muminus[i] * n_muminus_per_ifb_cm2 * F222_ifb * F222_area_cm2 / 5e6
    for i in range(len(n_nh_from_muminus))
]
n_nh_muplus = [
    n_nh_from_muplus[i] * n_muplus_per_ifb_cm2 * F222_ifb * F222_area_cm2 / 5e6
    for i in range(len(n_nh_from_muplus))
]
n_nh = np.array(n_nh_muplus) + np.array(n_nh_muminus)

# Only the first 526 plates are reconstructed for large volume samples,
# so a correction is applied to account for the missing plates.
WEIGHTS = {
    200213: 9.5 / (10e3 * 50 / 560),
    200214: 9.5 / (10e3 * 50 / 759),
    200215: 9.5 / (10e3 * 50 / 190),
    200300: 9.5 / (10e3 * 50 / 560),
    200301: 9.5 / (10e3 * 50 / 760),
    200302: 9.5 / (10e3 * 50 / 190),
    200303: 9.5 / (10e3 * 50 / 560),
    200304: 9.5 / (10e3 * 50 / 760),
    200305: 9.5 / (10e3 * 50 / 190),
    200306: 9.5 / (10e3 * 50 / 560),
    200307: 9.5 / (10e3 * 50 / 760),
    200308: 9.5 / (10e3 * 50 / 190),
    200309: 9.5 / (10e3 * 50 / 560),
    200310: 9.5 / (10e3 * 50 / 760),
    200311: 9.5 / (10e3 * 50 / 190),
    200082: 9.5
    / (10e3 * 300 / 560)
    * (PLATE_MAX_NOMINAL - PLATE_MIN_NOMINAL + 1)
    / (526 - PLATE_MIN_NOMINAL + 1),
    200083: 9.5
    / (10e3 * 300 / (760-7))
    * (PLATE_MAX_NOMINAL - PLATE_MIN_NOMINAL + 1)
    / (526 - PLATE_MIN_NOMINAL + 1),
    200089: 9.5
    / (10e3 * 180 / (180-9))
    * (PLATE_MAX_NOMINAL - PLATE_MIN_NOMINAL + 1)
    / (526 - PLATE_MIN_NOMINAL + 1),
    100090: n_nh[0]
    / 50_000  # 1000 events per ntp file, 50 ntp files have been reconstructed
    / nh_fiducial_fraction
    * (PLATE_MAX_NOMINAL - PLATE_MIN_NOMINAL + 1)
    / (526 - PLATE_MIN_NOMINAL + 1),
    100091: n_nh[1]
    / 50_000
    / nh_fiducial_fraction
    * (PLATE_MAX_NOMINAL - PLATE_MIN_NOMINAL + 1)
    / (526 - PLATE_MIN_NOMINAL + 1),
    100092: n_nh[2]
    / 50_000
    / nh_fiducial_fraction
    * (PLATE_MAX_NOMINAL - PLATE_MIN_NOMINAL + 1)
    / (526 - PLATE_MIN_NOMINAL + 1),
    100093: n_nh[3]
    / 50_000
    / nh_fiducial_fraction
    * (PLATE_MAX_NOMINAL - PLATE_MIN_NOMINAL + 1)
    / (526 - PLATE_MIN_NOMINAL + 1),
    100094: n_nh[4]
    / 50_000
    / nh_fiducial_fraction
    * (PLATE_MAX_NOMINAL - PLATE_MIN_NOMINAL + 1)
    / (526 - PLATE_MIN_NOMINAL + 1),
    100095: n_nh[5]
    / 50_000
    / nh_fiducial_fraction
    * (PLATE_MAX_NOMINAL - PLATE_MIN_NOMINAL + 1)
    / (526 - PLATE_MIN_NOMINAL + 1),
}

WEIGHTS = {k: v * rescale_factor for k, v in WEIGHTS.items()}
