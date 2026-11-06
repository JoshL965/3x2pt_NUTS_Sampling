'''
NUTS sampler using cloelike for a 3x2pt analysis.
====================================================
This script relies on the cloe_sampler or cloe_sampler_gpu conda enviroment 
as chnages have been made to the cloelike and cloelib source code to allow for 
NUTS compile.

Data is from the cloe-playground. The current covariance is scaled by f_sky^2 
as without it the posteriors = priors.

Runs best on GPU. Not currently fast on CPU. Does scale better with dimensionality 
than Nautilus but Nautilus does have the added benefit of providing the evidence.
'''

# =======================================================================
# SETUP 
# =======================================================================

# SET WHETHER THE LENS SAMPLE IS EQUAL TO SOURCE SAMPLE.
# IF TRUE THEN SETS SAME REDSHIFT BIN SHIFT FOR BOTH POS AND SHEAR TOMOGRAPHIC BINS.
# IF TRUE LENS AND SOURCE SAMPLES MUST HAVE SAME REDSHIFT DISTRIBUTION 
# IF TRUE REDUCES THE NUMBER OF SAMPLED PARAMETERS BY N_BINS/2
LENS_EQUAL_SOURCE = True

# SET WHETHER TO INCLUDE MAGNIFICATION BIAS TO THE SAMPLED PARAMS
# IF FALSE THEN MAG_BIAS TERMS SET TO FIDUCIAL VALUE OF 0.0 FOR EACH BIN
MAG_BIAS = True 

# SET NUMBER OF TOMOGRAPHIC REDSHIFT BINS
N_BINS = 6 

# SET PATH TO SAVE SAMPLES TO
SAVE_PATH = '/cephfs/jlayton/PhD/Results/w0_wa_varying/Samples' 

USE_MOCK_FROM_MODEL = True

# =======================================================================
# LOAD IMPORTS
# =======================================================================

import os 

from jax.experimental import checkify
# THIS WAS IN AN ATTEMPT TO REMOVE CHECKIFY ERRORS
checkify.check = lambda cond, msg, *args, **kwargs: None

import jax
# SET 64-BIT PRECISION FOR JAX. CRUCIAL FOR 3X2PT.
jax.config.update("jax_enable_x64", True)

import sys
import numpy as np
from scipy import integrate
from copy import deepcopy
import time

# NUMPYRO SAMPLER 
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.reparam import LocScaleReparam
from numpyro.handlers import reparam

import jax.numpy as jnp

# DATA READING
import euclidlib as el
import random

# COSMOLOGY AND THEORY
from cloelib.cosmology.cosmopower_jax_cosmology import CosmoPowerJAXw0waCDMPerturbations
from cloelib.cosmology.jax_cosmology import JAXBackground

# LIKELIHOOD
from cloelike.EuclidLikelihood_photo_Cls import EuclidLikelihood_3x2pt

# SUPRESS WARNINGS
np.seterr(divide='ignore', over='ignore', invalid='ignore')

# =======================================================================
# CREATE FASTER COSMOLOGY 
# =======================================================================

# DEFINE REDSHIFT GRID
Z_MAX = 3.1
N_Z_GRID = 256
_Z_GRID = jnp.linspace(0.0, Z_MAX, N_Z_GRID)
dZ = Z_MAX / (N_Z_GRID - 1)

# DEFINE SPEED OF LIGHT IN M/S. TAKEN FROM CLOELIB AUXILLARY/UNITS.PY.
SPEED_OF_LIGHT = 2.99792458e8

# DEFINE FUNCTION FOR CREATING DIMENSIONLESS HUBBLE PARAMETER
def fast_E_z(z_array, Omega_m0, Omega_k0, w0, wa):
    """Dimensionless Hubble parameter E(z) = H(z) / H0."""
    z_arr = jnp.asarray(z_array)
    zp1 = 1.0 + z_arr
    Ode0 = 1.0 - Omega_m0 - Omega_k0
    ez2 = (
        Omega_m0 * (zp1 ** 3)
        + Omega_k0 * (zp1 ** 2)
        + Ode0 * (zp1 ** (3.0 * (1.0 + w0 + wa))) * jnp.exp(-3.0 * wa * z_arr / zp1)
    )
    return jnp.sqrt(ez2)

# REDEFINE JAX COSMOLOGY CLASS FOR IMPROVED COMPILE TIME. 
# MAINLY OVERIDES SLOW QUADGK INTEGRATORS, SOME CHECKIFY CALLS AND PYTHON LIST COMPREHENSIONS.
# CURRENTLY DOES NOT INCLUDE OPTIONS FOR NON-FLAT CURVATURE.
class FastJAXBackground(JAXBackground):
    """Overrides slow quadgk integrators, checkify, and Python list comprehensions."""

    def __init__(
        self,
        H0: float,
        Omega_b0: float,
        Omega_cdm0: float,
        Omega_k0: float,
        As: float,
        ns: float,
        mnu,
        w0: float,
        wa: float,
        gamma_MG: float,
        N_mnu: int,
        N_ur=None,
        alpha_s: float = 0.0,
        **kwargs,
    ):
        self.H0 = H0
        self.h = H0 / 100.0
        self.Omega_b0 = Omega_b0
        self.Omega_cdm0 = Omega_cdm0
        self.Omega_k0 = Omega_k0
        self.As = As
        self.ns = ns
        self.alpha_s = alpha_s
        self.w0 = w0
        self.wa = wa
        self.gamma_MG = gamma_MG
        self.N_mnu = N_mnu
        self.mnu = jnp.sum(jnp.asarray(mnu, dtype=float))
        self._provided_N_ur = N_ur
        self.Omega_nu0 = self.mnu / (93.14 * (self.h * self.h))
        self.Omega_m0 = self.Omega_b0 + self.Omega_cdm0 + self.Omega_nu0

        # COMPUTE THE COMOVING DISTANCE GRID ONCE PER COSMOLOGY USING TRAPEZOID INTEGRATION
        H_grid = self.H0 * fast_E_z(_Z_GRID, self.Omega_m0, self.Omega_k0, self.w0, self.wa)
        integrand = (SPEED_OF_LIGHT/1000) / H_grid
        self._chi_grid = jnp.concatenate([
            jnp.array([0.0]),
            jnp.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dZ),
        ])

    def _set_neutrino_mass(self, mnu, N_mnu):
        # BYPASSES SLOW CHECKIFY 
        return jnp.sum(jnp.asarray(mnu, dtype=float))

    def hubble_parameter(self, zs, units: str = "km/s/Mpc", **kwargs):
        H = self.H0 * fast_E_z(zs, self.Omega_m0, self.Omega_k0, self.w0, self.wa)
        if units == "1/Mpc":
            return H / (SPEED_OF_LIGHT / 1000.0)
        return H

    def comoving_distance(self, zs: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jnp.interp(zs, _Z_GRID, self._chi_grid)

    def transverse_comoving_distance(self, zs, **kwargs):
        return self.comoving_distance(zs, **kwargs)

    def angular_diameter_distance(self, zs, **kwargs):
        return self.comoving_distance(zs, **kwargs) / (1.0 + jnp.asarray(zs))

    def Omega_b(self, zs):
        zs = jnp.asarray(zs)
        return self.Omega_b0 * (1.0 + zs) ** 3 / fast_E_z(zs, self.Omega_m0, self.Omega_k0, self.w0, self.wa) ** 2

    def Omega_cb(self, zs):
        zs = jnp.asarray(zs)
        return (self.Omega_b0 + self.Omega_cdm0) * (1.0 + zs) ** 3 / fast_E_z(zs, self.Omega_m0, self.Omega_k0, self.w0, self.wa) ** 2


# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

# LOACTION OF COVMAT, MIXMAT, REDSHIFT DISTRIBUTION, AND SYNTHETIC CELLS
file_path = '/cephfs/jlayton/PhD/cloe_playground/'

# LOAD REDSHIFT DISTRIBUTION
z_nz, nz_example = el.phz.redshift_distributions(file_path+'nz_example.fits')

# CREATE UNIFORM REDSHIFT GRID FOR INTERPOLATION
myz = np.linspace(1e-4, 3.0, 100)

def normalize_and_resample(nz_dict, z_grid, z_target):
    """Normalize and resample n(z) to a target redshift grid."""
    nz_array = np.vstack([nz / integrate.trapezoid(nz, z_grid) 
                          for nz in nz_dict.values()])
    return np.array([np.interp(z_target, z_grid, nz) for nz in nz_array])

my_dndz_pos_norm = normalize_and_resample(nz_example, z_nz, myz)
my_dndz_she_norm = normalize_and_resample(nz_example, z_nz, myz)

# LOAD POWER SPECTRA AND MIXING MATRICES
cells_data = el.le3.pk_wl.angular_power_spectra(file_path+'synthetic_coupled_cells.fits')
mixmat = el.le3.pk_wl.mixing_matrices(file_path+'mixmats_example.fits')

# LOAD COVARIANCE MATRIX
# MULTIPLIED BY F_SKY^2 TO GIVE VALID POSTERIORS. 
# THIS NEEDS TO BE CHANGED FOR ACTUAL VALID COV MATRIX (373.1)
full_cov = np.load(file_path+'covmat_2D.npz')['Gauss']/373.1

# ============================================================================
# BUILD LIKELIHOOD DATA AND SETTINGS
# ============================================================================

def build_data(ell_key, cov, include_pos=False, include_she=False):
    """Organize data dictionary for likelihood."""
    data = {
        'cells': cells_data,
        'ells': cells_data[ell_key].ell,
        'z_arr': myz,
        'cov': cov,
        'mixmat': mixmat,
    }
    
    if include_pos:
        data['dndz_pos'] = my_dndz_pos_norm
    if include_she:
        data['dndz_she'] = my_dndz_she_norm
    return data

def build_settings(cls):
    """Define scale cuts for angular scales."""
    scale_cuts = {}
    for key in cls:
        if key[:2] == ('POS', 'POS'):
            scale_cuts[key] = [10, 750]
        elif key[:2] == ('POS', 'SHE'):
            scale_cuts[key] = [10, 750]
        elif key[:2] == ('SHE', 'SHE'):
            scale_cuts[key] = [10, 1500]
    
    return {
        'n_ell_bins': 32,
        'scale_cuts': scale_cuts,
        'galaxy_bias_model': "per_bin",
        'ia_model': 'NLA',

    }

# PREPARE DATA FOR 3X2PT ANALYSIS
data_3x2pt = build_data(('POS', 'POS', 1, 1), full_cov, 
                        include_pos=True, include_she=True)
settings_3x2pt = build_settings(cells_data)

like_instance = EuclidLikelihood_3x2pt(
    data=data_3x2pt,
    settings=settings_3x2pt,
    Background=FastJAXBackground,
    LinPerturbations=CosmoPowerJAXw0waCDMPerturbations.Linear,
    NonLinPerturbations=CosmoPowerJAXw0waCDMPerturbations.NonLinear,
    mode='coupled'
)

# PRE-ALLOCATE STATIC JAX SCALARS FOR FIXED PARAMS SO JAX DOESNT RE-CONVERY PYTHON FLOATS
# IF USE_MOCK_FROM_MODEL IS TRUE, USE THIS DICTIONARY TO SET VALUES FOR MOCK DATA
# 
_raw_default_pars = {
    "H0": 70.0, "Omega_cdm0": 0.25, "Omega_b0": 0.05, "ns": 0.96, "As": 2.1e-9,
    "w0": -1.0, "wa": 0.0, "Omega_k0": 0, "mnu": 0.06, "gamma_MG": 0.545,
    "N_mnu": 1, "alpha_s": 0.0, "AIA": 1.12, "CIA": 0.0134, "EtaIA": 2.06,
    "log10TAGN": 7.86,
    "b1_photo_bin0": 1.0, "b1_photo_bin1": 1.0, "b1_photo_bin2": 1.0,
    "b1_photo_bin3": 1.0, "b1_photo_bin4": 1.0, "b1_photo_bin5": 1.0,
    "magnification_bias_1": 0.0, "magnification_bias_2": 0.0,
    "magnification_bias_3": 0.0, "magnification_bias_4": 0.0,
    "magnification_bias_5": 0.0, "magnification_bias_6": 0.0,
    "dz_pos_1": 0.0, "dz_pos_2": 0.0, "dz_pos_3": 0.0,
    "dz_pos_4": 0.0, "dz_pos_5": 0.0, "dz_pos_6": 0.0,
    "multiplicative_bias_1": 0.0, "multiplicative_bias_2": 0.0,
    "multiplicative_bias_3": 0.0, "multiplicative_bias_4": 0.0,
    "multiplicative_bias_5": 0.0, "multiplicative_bias_6": 0.0,
    "dz_shear_1": 0.0, "dz_shear_2": 0.0, "dz_shear_3": 0.0,
    "dz_shear_4": 0.0, "dz_shear_5": 0.0, "dz_shear_6": 0.0,
    "width_shear_1": 1.0, "width_shear_2": 1.0, "width_shear_3": 1.0,
    "width_shear_4": 1.0, "width_shear_5": 1.0, "width_shear_6": 1.0,
    "width_pos_1": 1.0, "width_pos_2": 1.0, "width_pos_3": 1.0,
    "width_pos_4": 1.0, "width_pos_5": 1.0, "width_pos_6": 1.0,
}

default_pars = {
    k: (v if isinstance(v, int) else jnp.asarray(v, dtype=jnp.float64))
    for k, v in _raw_default_pars.items()
}

# CALL MASKING VECTOR AND MASKED COV MATRIX SO THEY ARE CACHED AND NUTS DOESNT TRY TRACING 
like_instance.get_masking_vector_cached() 
like_instance.get_covariance_matrix_masked_inv()

# CREATE WRAPPER FOR CLOELIKE LIKELIHOOD
def log_like_wrapper(sampled_pars):
    full_pars = default_pars.copy()
    full_pars.update(sampled_pars)
    return like_instance.loglike(full_pars)

# CREATE KEYS 
B1_KEYS = tuple(f"b1_photo_bin{i}" for i in range(6))
DZ_POS_KEYS = tuple(f"dz_pos_{i+1}" for i in range(6))
DZ_SHEAR_KEYS = tuple(f"dz_shear_{i+1}" for i in range(6))
MAG_BIAS_KEYS = tuple(f"magnification_bias_{i+1}" for i in range(6))

# ============================================================================
# SYNTHETIC DATA
# ============================================================================
if USE_MOCK_FROM_MODEL:
    fiducial_theory = np.asarray(like_instance.get_theory_vector_masked(default_pars))
    like_instance.get_data_vector_masked = lambda: fiducial_theory


# NEED TO CALL THIS OR FOR SOME REASON NUTS THROWS CHECKIFY ERROR
jitted_fn = jax.jit(log_like_wrapper) 

# ============================================================================
# BUILD SAMPLER
# ============================================================================

# DEFINE THE PRIORS AND LIKELIHOOD FOR SAMPLING
def cosmo_model():

    # SET COSMOLOGICAL PARAMETERS
    ombh2 = numpyro.sample('ombh2', dist.Normal(0.0245, 0.00038))
    omch2 = numpyro.sample('omch2', dist.Uniform(0.06, 0.2))
    logAs = numpyro.sample('logAs', dist.Uniform(2.2, 3.8))
    ns    = numpyro.sample('ns', dist.Uniform(0.6, 1.2))
    H0    = numpyro.sample('H0', dist.Uniform(55.0, 91.0))
    w0    = numpyro.sample('w0', dist.Uniform(-3.0, 1.0))
    wa    = numpyro.sample('wa', dist.Uniform(-3.0, 2.0))

    numpyro.factor('w0+wa<0', jnp.where(w0 + wa < 0.0, 0.0, -jnp.inf))
    # SET NLA IA PARAMETERS
    AIA   = numpyro.sample('AIA', dist.Uniform(-5, 5))
    EtaIA = numpyro.sample('EtaIA', dist.Uniform(-5, 5))

    # DEFINE DETRMINISTIC PARAMETERS TO BE TRACKED
    inv_h2     = 10000.0 / (H0**2)
    Omega_cdm0 = numpyro.deterministic('Omega_cdm0', omch2 * inv_h2)
    Omega_b0   = numpyro.deterministic('Omega_b0', ombh2 * inv_h2)
    As         = numpyro.deterministic('As', jnp.exp(logAs) * 1e-10)

    sampled_pars = {
        'H0': H0,
        'Omega_cdm0': Omega_cdm0,
        'Omega_b0': Omega_b0,
        'ns': ns,
        'As': As,
        'AIA': AIA,
        'EtaIA': EtaIA,
        'w0': w0,
        'wa': wa,
    }

    # DEFINE REDSHIFT BIN SHIFTS (APPLIES SAME BIN SHIFT IF LENS_EQUAL_SOURCE=TRUE)
    if LENS_EQUAL_SOURCE:
        dz_pos = numpyro.sample('dz_pos', dist.Normal(0.0, 0.01).expand([N_BINS]))
        dz_shear = numpyro.deterministic('dz_shear', dz_pos)
        for i in range(N_BINS):
            sampled_pars[DZ_POS_KEYS[i]] = dz_pos[i]
            sampled_pars[DZ_SHEAR_KEYS[i]] = dz_shear[i]
    else:
        dz_pos = numpyro.sample('dz_pos', dist.Normal(0.0, 0.01).expand([N_BINS]))
        dz_shear = numpyro.sample('dz_shear', dist.Normal(0.0, 0.01).expand([N_BINS]))
        for i in range(N_BINS):
            sampled_pars[DZ_POS_KEYS[i]] = dz_pos[i]
            sampled_pars[DZ_SHEAR_KEYS[i]] = dz_shear[i]

    # IF MAG_BIAS=TRUE THEN DEFINE MAGNIFICATION BIAS PARAMETERS
    if MAG_BIAS:
        mag_bias = numpyro.sample('magnification_bias', dist.Uniform(-2.0, 2.0).expand([N_BINS]))
        for i in range(N_BINS):
            sampled_pars[MAG_BIAS_KEYS[i]] = mag_bias[i]

    # DEFINE PER_BIN GALAXY BIAS PARAMETERS
    b1_photo = numpyro.sample('b1_photo', dist.Uniform(0.0, 3.0).expand([N_BINS]))
    for i in range(N_BINS):
        sampled_pars[B1_KEYS[i]] = b1_photo[i]

    # CALCULATE LOG-LIKELIHOOD USING CLOELIKE
    numpyro.factor('log_like', jitted_fn(sampled_pars))

# SETTINGS FOR RUNNING THE SAMPLER
def run_sampler():
    os.makedirs(SAVE_PATH, exist_ok=True)
    # USE NON-CENTRED REPARAMITRISATION N(0,1)
    reparam_config = {
        'ombh2': LocScaleReparam(centered=0),
        'dz_pos': LocScaleReparam(centered=0),
        }
    if not LENS_EQUAL_SOURCE:
        reparam_config['dz_shear'] = LocScaleReparam(centered=0)

    # CREATE REPARIMITARISED MODEL
    reparameterized_model = reparam(cosmo_model, config=reparam_config)
    kernel = NUTS(
        reparameterized_model,
        target_accept_prob=0.95, # CAN INCREASE IF TOO MANY DIVERGENCES 
        dense_mass=True, # HELPS FOR THE HIGHLY CORRELATED PARAMETERS
        max_tree_depth=(6,10), # HELPS WITH BAD TRAJECTORIES, AFTER ~500 STEPS SHOULDNT HIT MAX_TREE_DEPTH
    )

    mcmc = MCMC(
        kernel,
        num_warmup=1000,
        num_samples=3000, # SHOULD BE ENOUGH FOR CONVERGENCE, SHOULD BE LOWERED IF MORE CHAINS
        num_chains=4, # 4 CHAINS BEST FOR 1 GPU, 8-16 CHAINS BEST FOR >16 CPUS
        chain_method='vectorized', # VECTORIZED BEST FOR GPU, PARALLEL BEST FOR CPU
        progress_bar=True # SHOULD SPEED UP SAMPLING WHEN FALSE
    )
    rng_key = jax.random.PRNGKey(random.randint(0, 2**32 - 1))

    print("=" * 60)
    print("Running sampler")
    print("=" * 60)

    mcmc.run(rng_key, extra_fields=('num_steps',))
    mcmc.print_summary(exclude_deterministic=False)
    print(mcmc.get_extra_fields())
    raw_samples = mcmc.get_samples()

    # REMOVE _DECENTERED FROM THE NON-CENTRED REPARAMITRISATION
    samples_dict = {
        k: np.asarray(v) for k, v in raw_samples.items()
        if not k.endswith('_decentered')
    }

    les = 'LES' if LENS_EQUAL_SOURCE else 'LPS'
    mb = '_MagBias' if MAG_BIAS else ''

    np.savez_compressed(
        os.path.join(SAVE_PATH, f'NUTS_3x2pt_{N_BINS}bins_{les}{mb}_samples.npz'),
        **samples_dict
    )

run_sampler()