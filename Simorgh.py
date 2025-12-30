import os
import sys
import argparse
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from collections import deque, defaultdict, namedtuple
from pathlib import Path
from typing import List, Dict, Any, Tuple
import networkx as nx
from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("Simorgh-V20")

# Check for Optional Libraries
try:
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    logger.warning("Streamlit/Plotly not found. Dashboard mode disabled.")

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not found. API mode disabled.")

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not found. Running in slow mode.")
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(n):
        return range(n)

# ═══════════════════════════════════════════════════════════════════════════
# EXTENDED ACTION SET (Now 10 Actions)
# ═══════════════════════════════════════════════════════════════════════════

ACT_NAMES = [
    "WAIT", "RIOT POLICE", "NET CUT", "INFILTRATE", 
    "REFORM", "PROPAGANDA", "GREY ZONE RENTS", 
    "TARGETED ARRESTS", "CYBER DEFENSE", "DIPLOMATIC ENGAGEMENT"
]

class RLConfig:
    """Hyperparameters for SOTA Agent"""
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    memory_size: int = 200000  # Increased for complex scenarios
    batch_size: int = 128
    warmup_steps: int = 200
    target_update_freq: int = 2
    use_per: bool = True

RL_CFG = RLConfig()

# ═══════════════════════════════════════════════════════════════════════════
# ADVANCED CONFIGURATION WITH NEW MODULES
# ═══════════════════════════════════════════════════════════════════════════

class EconomicHyperparams:
    base_cpi: float = 0.45
    food_inflation_bias: float = 1.5
    housing_burden: float = 0.60
    gini_coefficient: float = 0.42
    subsistence_threshold: float = 0.25
    shadow_economy_size: float = 0.35
    # New: Behavioral Economics
    speculation_amplifier: float = 2.5
    panic_buying_threshold: float = 0.65
    expectation_anchor_decay: float = 0.92

class SociologicalHyperparams:
    ethnic_friction_matrix: np.ndarray = np.array([
        [0.1, 0.4, 0.5, 0.3],
        [0.4, 0.1, 0.3, 0.2],
        [0.5, 0.3, 0.0, 0.2],
        [0.3, 0.2, 0.2, 0.0]
    ])
    intergroup_bridging: float = 0.15
    strong_tie_weight: float = 0.8
    weak_tie_weight: float = 0.2
    preference_falsification: float = 0.4
    # New: Collective Action Thresholds
    min_threshold: float = 0.05
    max_threshold: float = 0.80
    # New: Protest Fatigue
    stamina_drain_rate: float = 0.05
    stamina_recovery_rate: float = 0.02

class RegimeHyperparams:
    irgc_loyalty: float = 0.85
    artesh_loyalty: float = 0.60
    basij_density: float = 0.05
    digital_sovereignty: float = 0.7
    elite_fragmentation: float = 0.2
    # New: Security Forces
    initial_security_morale: float = 0.80
    morale_drain_per_operation: float = 0.03
    morale_recovery_rate: float = 0.01
    defection_threshold: float = 0.30

class InformationWarfareParams:
    """New Module: Information Warfare"""
    external_agitation_baseline: float = 0.3
    astroturfing_capacity: float = 0.5
    state_bot_efficiency: float = 0.4
    opposition_bot_efficiency: float = 0.6
    viral_spark_probability: float = 0.02
    echo_chamber_polarization: float = 0.7
    information_saturation_threshold: float = 0.85

class IranConfig:
    econ: EconomicHyperparams = EconomicHyperparams()
    soc: SociologicalHyperparams = SociologicalHyperparams()
    regime: RegimeHyperparams = RegimeHyperparams()
    info_war: InformationWarfareParams = InformationWarfareParams()
    
    sanctions_base: float = 0.3
    regime_legitimacy_base: float = 0.4
    regional_development_index: List[float] = [1.0, 0.85, 0.80, 0.40]

class SystemConfig:
    n_actors: int = 2000  # Increased for richer dynamics
    n_cities: int = 4
    simulation_days: int = 365
    temporal_resolution: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_gpu: bool = torch.cuda.is_available()
    hysteresis_factor: float = 0.95
    entropy_threshold: float = 0.7
    iran: IranConfig = IranConfig()
    checkpoint_dir: str = "./checkpoints"

CFG = SystemConfig()

# ═══════════════════════════════════════════════════════════════════════════
# KERNEL 1: MULTI-FACTORIAL GRIEVANCE WITH TRAUMA MEMORY
# ═══════════════════════════════════════════════════════════════════════════

@njit(fastmath=True)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

@njit(parallel=True, fastmath=True)
def kernel_advanced_grievance_v2(
    current_grievance, trauma_memory, social_scars, resources, expectations,
    age, city_id, urban, ethnicity,
    cpi_food, cpi_housing, currency_value,
    sanction_intensity, regime_legitimacy,
    regional_dev, ethnic_bias,
    neighbor_wealth_avg, vpn_access,
    external_agitation, perceived_inflation,
    stamina
):
    n = len(current_grievance)
    res = np.empty(n, dtype=np.float32)
    
    SUBSISTENCE_FLOOR = 0.2
    YOUTH_BULGE_AGE = 30.0
    
    for i in prange(n):
        # Economic Calculation
        if urban[i]:
            inflation_impact = (0.7 * cpi_housing) + (0.3 * cpi_food)
        else:
            inflation_impact = (0.4 * cpi_housing) + (0.6 * cpi_food)
        
        # Behavioral Economics: Perceived vs Real Inflation
        real_inflation = inflation_impact
        perceived_inflation_i = real_inflation * (1.0 + external_agitation * 0.5)
        
        real_purchasing_power = resources[i] / (1.0 + perceived_inflation_i)
        expectation_gap = max(0.0, (expectations[i] - real_purchasing_power))
        
        subsistence_stress = 0.0
        if real_purchasing_power < SUBSISTENCE_FLOOR:
            subsistence_stress = 2.0 * np.exp(3.0 * (SUBSISTENCE_FLOOR - real_purchasing_power))
        
        neighbor_envy = max(0.0, neighbor_wealth_avg[i] - real_purchasing_power)
        
        # Information Warfare Impact
        awareness = 0.5 + (0.5 * vpn_access[i])
        city_idx = int(city_id[i])
        eth_idx = int(ethnicity[i])
        
        structural_pain = (1.0 - regional_dev[city_idx]) + (ethnic_bias[eth_idx] * sanction_intensity)
        political_anger = structural_pain * awareness * (1.0 - regime_legitimacy)
        
        # Age Factor with Generational Gap
        age_factor = 1.0
        if age[i] < YOUTH_BULGE_AGE:
            # Youth: More sensitive to digital deprivation and freedoms
            digital_deprivation = (1.0 - vpn_access[i]) * 0.3
            age_factor = 1.5 + (0.1 * expectation_gap) + digital_deprivation
        else:
            # Older: More sensitive to inflation and stability
            age_factor = 1.0 + (0.2 * perceived_inflation_i)
        
        # Trauma Memory Impact (Social Scars)
        trauma_amplifier = 1.0 + (social_scars * 0.5)
        
        # Fatigue Consideration
        stamina_penalty = 1.0 - (0.3 * (1.0 - stamina[i]))
        
        total_g = (
            (0.25 * expectation_gap) +
            (0.20 * subsistence_stress) +
            (0.10 * neighbor_envy) +
            (0.35 * political_anger) +
            (0.10 * external_agitation)
        ) * age_factor * trauma_amplifier * stamina_penalty
        
        # Update Trauma Memory (Long-term grievance accumulation)
        trauma_memory[i] = (0.995 * trauma_memory[i]) + (0.005 * total_g)
        final_g = 0.65 * total_g + 0.35 * trauma_memory[i]
        res[i] = min(1.0, max(0.0, final_g))
    
    return res

# ═══════════════════════════════════════════════════════════════════════════
# KERNEL 2: COMPLEX CONTAGION WITH THRESHOLD MODEL
# ═══════════════════════════════════════════════════════════════════════════

@njit(parallel=True, fastmath=True)
def kernel_threshold_diffusion(
    active_state, activation_potential, activation_thresholds,
    adj_indices, adj_indptr, adj_weights,
    grievance, risk_aversion, centrality,
    police_presence, global_coordination_signal,
    collective_courage_boost, radicalization_index,
    stamina
):
    n = len(active_state)
    new_active = np.empty(n, dtype=np.bool_)
    new_potential = np.empty(n, dtype=np.float32)
    
    BACKFIRE_KINK = 0.65
    
    for i in prange(n):
        start = adj_indptr[i]
        end = adj_indptr[i+1]
        
        if start == end:
            new_active[i] = active_state[i]
            new_potential[i] = 0.0
            continue
        
        neighbors = adj_indices[start:end]
        weights = adj_weights[start:end]
        
        weighted_active_sum = 0.0
        total_weight = 0.0
        
        for k in range(len(neighbors)):
            neighbor = neighbors[k]
            w = weights[k]
            total_weight += w
            if active_state[neighbor]:
                weighted_active_sum += w
        
        social_pressure = 0.0
        if total_weight > 0:
            social_pressure = weighted_active_sum / total_weight
        
        # Threshold Model: Only activate if social pressure exceeds personal threshold
        personal_threshold = activation_thresholds[i]
        
        # Collective Courage amplifies social pressure
        effective_pressure = social_pressure * (1.0 + collective_courage_boost)
        
        perceived_safety = sigmoid(10 * (effective_pressure - 0.3))
        local_police = police_presence[i]
        
        # Radicalization reduces fear of repression
        fear_reduction = radicalization_index * 0.5
        fear_cost = local_police * (1.0 - perceived_safety) * (1.0 - fear_reduction)
        
        if local_police > BACKFIRE_KINK and perceived_safety > 0.4:
            fear_cost *= -0.5  # Backfire effect
        
        # Stamina check: Can't protest if exhausted
        if stamina[i] < 0.1:
            total_drive = 0.0
        else:
            total_drive = (0.4 * grievance[i]) + \
                          (0.4 * effective_pressure) + \
                          (0.2 * global_coordination_signal)
        
        resistance = risk_aversion[i] + fear_cost
        d_potential = (total_drive - resistance)
        
        current_p = activation_potential[i] + (d_potential * 0.2)
        current_p = min(1.0, max(-0.5, current_p))
        new_potential[i] = current_p
        
        # Decision Logic with Threshold
        if active_state[i]:
            # Already active: deactivate if potential drops OR stamina depleted
            if current_p < 0.2 or stamina[i] < 0.05:
                new_active[i] = False
            else:
                new_active[i] = True
        else:
            # Not active: activate if potential AND social pressure exceed threshold
            if effective_pressure >= personal_threshold and current_p > 0.5:
                prob_activation = sigmoid(15 * (current_p - 0.5))
                if np.random.random() < prob_activation:
                    new_active[i] = True
                else:
                    new_active[i] = False
            else:
                new_active[i] = False
    
    return new_active, new_potential

# ═══════════════════════════════════════════════════════════════════════════
# KERNEL 3: ENTROPY-DRIVEN PANIC WITH INFORMATION SATURATION
# ═══════════════════════════════════════════════════════════════════════════

@njit(parallel=True, fastmath=True)
def kernel_entropy_panic_v2(
    panic_state, adj_indices, adj_indptr,
    internet_integrity, violence_visibility,
    time_of_day, urban_density,
    info_saturation, state_bot_activity, opposition_bot_activity
):
    n = len(panic_state)
    new_panic = np.empty(n, dtype=np.float32)
    
    # Information Warfare: Net effect of bot armies
    bot_war_effect = opposition_bot_activity - state_bot_activity
    
    # Information Saturation causes confusion (reduces panic spread)
    saturation_dampening = 1.0 - (info_saturation * 0.4)
    
    uncertainty = 1.0 - internet_integrity
    
    night_factor = 1.0
    if (time_of_day > 18.0) or (time_of_day < 4.0):
        night_factor = 1.3
    
    for i in prange(n):
        instinct = violence_visibility * night_factor
        
        # Bot influence on individual panic
        bot_influence = bot_war_effect * 0.3
        
        start = adj_indptr[i]
        end = adj_indptr[i+1]
        neighbors = adj_indices[start:end]
        
        local_hysteria = 0.0
        if len(neighbors) > 0:
            sum_p = 0.0
            max_p = 0.0
            for neighbor in neighbors:
                p_val = panic_state[neighbor]
                sum_p += p_val
                if p_val > max_p:
                    max_p = p_val
            
            avg_p = sum_p / len(neighbors)
            local_hysteria = (avg_p * (1.0 - uncertainty)) + (max_p * uncertainty)
        
        density_mod = 1.0 + (urban_density[i] * 0.5)
        current = panic_state[i]
        
        input_stimulus = (
            (instinct * 0.3) + 
            (local_hysteria * 0.5 * density_mod) + 
            (bot_influence * 0.2)
        ) * saturation_dampening
        
        if input_stimulus > current:
            next_val = current + 0.3 * (input_stimulus - current)
        else:
            next_val = current * 0.95
        
        new_panic[i] = min(1.0, max(0.0, next_val))
    
    return new_panic

# ═══════════════════════════════════════════════════════════════════════════
# KERNEL 4: SPECULATIVE RUMORS AND PANIC BUYING
# ═══════════════════════════════════════════════════════════════════════════

@njit(parallel=True, fastmath=True)
def kernel_speculative_shock(
    resources, expectations, cpi_base, panic_level,
    rumor_intensity, speculation_amplifier,
    panic_buying_threshold
):
    n = len(resources)
    new_expectations = np.empty(n, dtype=np.float32)
    hoarding_demand = np.empty(n, dtype=np.float32)
    
    for i in prange(n):
        # Rumors amplify perceived inflation
        rumor_shock = rumor_intensity * np.random.random() * speculation_amplifier
        perceived_inflation = cpi_base * (1.0 + rumor_shock)
        
        # Panic Buying: If panic exceeds threshold, start hoarding
        if panic_level[i] > panic_buying_threshold:
            hoarding_demand[i] = panic_level[i] * 0.5
        else:
            hoarding_demand[i] = 0.0
        
        # Update Expectations (adaptive)
        new_expectations[i] = expectations[i] * 0.95 + perceived_inflation * 0.05
    
    return new_expectations, hoarding_demand

# ═══════════════════════════════════════════════════════════════════════════
# POPULATION WITH EXTENDED ATTRIBUTES
# ═══════════════════════════════════════════════════════════════════════════

class EnhancedPopulation:
    def __init__(self, config: SystemConfig):
        self.cfg = config
        self.n = config.n_actors
        self.use_gpu = config.use_gpu
        
        if self.use_gpu:
            try:
                import cupy as cp
                self.xp = cp
            except ImportError:
                self.use_gpu = False
                self.xp = np
                logger.warning("CuPy not found. Fallback to CPU/Numpy.")
        else:
            self.xp = np
        
        self._init_population()
        self._build_network()
    
    def _init_population(self):
        # Basic Demographics
        self.ages = np.random.randint(15, 70, self.n).astype(np.float32)
        self.resources = np.random.lognormal(0, 0.5, self.n).astype(np.float32)
        self.resources = (self.resources - self.resources.min()) / (self.resources.max() - self.resources.min())
        
        self.ethnicities = np.random.choice([0, 1, 2, 3], self.n, p=[0.55, 0.20, 0.15, 0.10]).astype(np.float32)
        self.city_ids = np.random.randint(0, self.cfg.n_cities, self.n).astype(np.float32)
        self.urban = np.random.choice([True, False], self.n, p=[0.75, 0.25])
        
        # Psychological States
        self.grievance = np.zeros(self.n, dtype=np.float32)
        self.panic = np.zeros(self.n, dtype=np.float32)
        self.stubbornness = np.random.beta(2, 5, self.n).astype(np.float32)
        self.charisma = np.random.beta(1, 10, self.n).astype(np.float32)
        
        # Protest States
        self.active = np.zeros(self.n, dtype=bool)
        self.stamina = np.ones(self.n, dtype=np.float32)  # New: Protest Fatigue
        
        # Individual Resilience Profiles (New)
        profile_probs = [0.15, 0.65, 0.20]  # Loyalists, Silent Majority, Activists
        self.resilience_profiles = np.random.choice([0, 1, 2], self.n, p=profile_probs).astype(np.int32)
        
        # Activation Thresholds (New: Threshold Model)
        base_thresholds = np.random.uniform(
            self.cfg.iran.soc.min_threshold,
            self.cfg.iran.soc.max_threshold,
            self.n
        ).astype(np.float32)
        
        # Adjust thresholds by profile
        for i in range(self.n):
            if self.resilience_profiles[i] == 0:  # Loyalists
                base_thresholds[i] = min(0.95, base_thresholds[i] + 0.3)
            elif self.resilience_profiles[i] == 2:  # Activists
                base_thresholds[i] = max(0.05, base_thresholds[i] - 0.2)
        
        self.activation_thresholds = base_thresholds
        
        # Agent Types
        self.types = np.zeros(self.n, dtype=np.int32)
        self.types[np.argsort(self.charisma)[-int(self.n*0.02):]] = 1  # Leaders
        self.types[np.random.choice(self.n, int(self.n*0.03), replace=False)] = 3  # Influencers
        
        # Convert to GPU if available
        if self.use_gpu:
            self.ages = self.xp.asarray(self.ages)
            self.resources = self.xp.asarray(self.resources)
            self.ethnicities = self.xp.asarray(self.ethnicities)
            self.city_ids = self.xp.asarray(self.city_ids)
            self.urban = self.xp.asarray(self.urban)
            self.grievance = self.xp.asarray(self.grievance)
            self.panic = self.xp.asarray(self.panic)
            self.stubbornness = self.xp.asarray(self.stubbornness)
            self.charisma = self.xp.asarray(self.charisma)
            self.active = self.xp.asarray(self.active)
            self.stamina = self.xp.asarray(self.stamina)
            self.types = self.xp.asarray(self.types)
            self.resilience_profiles = self.xp.asarray(self.resilience_profiles)
            self.activation_thresholds = self.xp.asarray(self.activation_thresholds)
    
    def _build_network(self):
        logger.info("Building Enhanced Small-World Network...")
        G = nx.watts_strogatz_graph(self.n, k=12, p=0.12)
        
        # Add preferential attachment for influencers
        influencer_indices = np.where(self.types == 3)[0]
        for inf_idx in influencer_indices[:min(10, len(influencer_indices))]:
            random_targets = np.random.choice(self.n, size=20, replace=False)
            for target in random_targets:
                if target != inf_idx:
                    G.add_edge(int(inf_idx), int(target))
        
        self.adj_matrix = nx.to_scipy_sparse_array(G, format='csr')
        logger.info(f"Network built: {self.n} nodes, {G.number_of_edges()} edges")
    
    def reset_state(self):
        if self.use_gpu:
            self.grievance.fill(0)
            self.panic.fill(0)
            self.active.fill(False)
            self.stamina.fill(1.0)
        else:
            self.grievance = np.zeros(self.n, dtype=np.float32)
            self.panic = np.zeros(self.n, dtype=np.float32)
            self.active = np.zeros(self.n, dtype=bool)
            self.stamina = np.ones(self.n, dtype=np.float32)

# ═══════════════════════════════════════════════════════════════════════════
# DEEP LEARNING CORE (Same as V19 but with larger action space)
# ═══════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity, use_per=False):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.use_per = use_per
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max(self.priorities) if self.priorities else 1.0)

    def sample(self, batch_size):
        if self.use_per:
            probs = np.array(self.priorities)
            probs = probs / probs.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        else:
            indices = np.random.choice(len(self.buffer), batch_size)
        
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-0.4) if self.use_per else np.ones(batch_size)
        weights = weights / weights.max()
        
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio + 1e-5

    def __len__(self):
        return len(self.buffer)

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=4, num_layers=2, seq_len=10):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                 dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        x = self.transformer(x)
        x = self.ln(x)
        return x[:, -1, :]

class DistributionalCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_quantiles=32, seq_len=10):
        super(DistributionalCritic, self).__init__()
        self.n_quantiles = n_quantiles
        self.state_encoder = TransformerEncoder(state_dim, hidden_dim, seq_len=seq_len)
        self.l1 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, n_quantiles)

    def forward(self, state_seq, action):
        s_emb = self.state_encoder(state_seq)
        sa = torch.cat([s_emb, action], 1)
        x = F.gelu(self.l1(sa))
        x = F.gelu(self.l2(x))
        quantiles = self.out(x)
        return quantiles

class HierarchicalActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, seq_len=10):
        super(HierarchicalActor, self).__init__()
        self.backbone = TransformerEncoder(state_dim, hidden_dim, seq_len=seq_len)
        
        self.manager_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.worker_l1 = NoisyLinear(hidden_dim * 2, hidden_dim)
        self.worker_l2 = NoisyLinear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state_seq):
        s_emb = self.backbone(state_seq)
        goal = self.manager_head(s_emb)
        worker_input = torch.cat([s_emb, goal], dim=1)
        
        x = F.gelu(self.worker_l1(worker_input))
        x = F.gelu(self.worker_l2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std, goal

    def sample(self, state_seq):
        mean, log_std, goal = self.forward(state_seq)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, goal

class SOTA_Agent:
    def __init__(self, state_dim, action_dim, cfg: RLConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = 10
        self.n_quantiles = 32
        
        self.actor = HierarchicalActor(state_dim, action_dim, seq_len=self.seq_len).to(self.device)
        self.critic = DistributionalCritic(state_dim, action_dim, n_quantiles=self.n_quantiles, seq_len=self.seq_len).to(self.device)
        self.critic_target = DistributionalCritic(state_dim, action_dim, n_quantiles=self.n_quantiles, seq_len=self.seq_len).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_opt = optim.AdamW(self.actor.parameters(), lr=cfg.lr_actor, weight_decay=1e-4)
        self.critic_opt = optim.AdamW(self.critic.parameters(), lr=cfg.lr_critic, weight_decay=1e-4)
        
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg.lr_alpha)
        self.alpha = self.log_alpha.exp()
        
        self.memory = ReplayBuffer(cfg.memory_size, use_per=True)
        self.steps = 0
        self.state_history = deque(maxlen=self.seq_len)
        
        # Add gradient clipping value
        self.max_grad_norm = 1.0

    def get_contextual_state(self, state):
        # Ensure state has no NaN
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
        
        while len(self.state_history) < self.seq_len:
            self.state_history.append(state)
        self.state_history.append(state)
        return np.array(list(self.state_history))

    def select_action(self, state, evaluate=False):
        # Safety check
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
        
        state_seq = self.get_contextual_state(state)
        state_seq_t = torch.FloatTensor(state_seq).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if evaluate:
                mean, log_std, _ = self.actor(state_seq_t)
                # Check for NaN in output
                if torch.isnan(mean).any() or torch.isnan(log_std).any():
                    logger.warning("NaN detected in actor output during evaluation, returning random action")
                    return np.random.randn(10) * 0.1
                action = torch.tanh(mean)
            else:
                action, _, _ = self.actor.sample(state_seq_t)
                if torch.isnan(action).any():
                    logger.warning("NaN detected in sampled action, returning random action")
                    return np.random.randn(10) * 0.1
        
        return action.cpu().numpy()[0]

    def quantile_loss(self, current_quantiles, target_quantiles, weights):
        pairwise_delta = target_quantiles.unsqueeze(2) - current_quantiles.unsqueeze(1)
        abs_pairwise_delta = torch.abs(pairwise_delta)
        huber_loss = torch.where(abs_pairwise_delta < 1.0,
                               0.5 * pairwise_delta ** 2,
                               abs_pairwise_delta - 0.5)
        
        n_quantiles = self.n_quantiles
        tau = torch.arange(0.5 / n_quantiles, 1, 1 / n_quantiles, device=self.device).view(1, n_quantiles, 1)
        loss = (torch.abs(tau - (pairwise_delta.detach() < 0).float()) * huber_loss).mean(dim=2).sum(dim=1)
        
        # Check for NaN in loss
        if torch.isnan(loss).any():
            logger.warning("NaN detected in quantile loss")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return (weights * loss).mean()

    def update(self, batch_size):
        if len(self.memory) < self.cfg.warmup_steps:
            return {}
        
        try:
            samples, indices, weights = self.memory.sample(batch_size)
            
            b_states = np.array([s[0] for s in samples])
            b_next_states = np.array([s[3] for s in samples])
            
            # Clean states
            b_states = np.nan_to_num(b_states, nan=0.0, posinf=1.0, neginf=0.0)
            b_next_states = np.nan_to_num(b_next_states, nan=0.0, posinf=1.0, neginf=0.0)
            
            states_seq = np.repeat(b_states[:, np.newaxis, :], self.seq_len, axis=1)
            next_states_seq = np.repeat(b_next_states[:, np.newaxis, :], self.seq_len, axis=1)

            states = torch.FloatTensor(states_seq).to(self.device)
            actions = torch.FloatTensor(np.array([s[1] for s in samples])).to(self.device)
            rewards = torch.FloatTensor(np.array([s[2] for s in samples])).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(next_states_seq).to(self.device)
            dones = torch.FloatTensor(np.array([s[4] for s in samples])).unsqueeze(1).to(self.device)
            weights_t = torch.FloatTensor(weights).to(self.device)
            
            # Clip rewards to prevent explosion
            rewards = torch.clamp(rewards, -1000, 1000)

            with torch.no_grad():
                next_actions, next_log_probs, _ = self.actor.sample(next_states)
                
                # Check for NaN
                if torch.isnan(next_actions).any() or torch.isnan(next_log_probs).any():
                    logger.warning("NaN detected in actor sample, skipping update")
                    return {}
                
                target_quantiles = self.critic_target(next_states, next_actions)
                target_quantiles = rewards + (1 - dones) * self.cfg.gamma * target_quantiles
                target_quantiles -= self.alpha * next_log_probs

            current_quantiles = self.critic(states, actions)
            critic_loss = self.quantile_loss(current_quantiles, target_quantiles, weights_t)
            
            if torch.isnan(critic_loss):
                logger.warning("NaN in critic loss, skipping update")
                return {}

            self.critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_opt.step()

            new_actions, log_probs, goal_vector = self.actor.sample(states)
            
            if torch.isnan(new_actions).any() or torch.isnan(log_probs).any():
                logger.warning("NaN in actor sample for policy update, skipping")
                return {'critic_loss': critic_loss.item()}
            
            q_quantiles = self.critic(states, new_actions)
            q_value = q_quantiles.mean(dim=1, keepdim=True)
            
            actor_loss = (self.alpha * log_probs - q_value).mean()
            goal_diversity_loss = -torch.var(goal_vector, dim=0).mean() * 0.1
            total_actor_loss = actor_loss + goal_diversity_loss
            
            if torch.isnan(total_actor_loss):
                logger.warning("NaN in actor loss, skipping actor update")
                return {'critic_loss': critic_loss.item()}

            self.actor_opt.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_opt.step()

            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            if not torch.isnan(alpha_loss):
                self.alpha_opt.zero_grad()
                alpha_loss.backward()
                self.alpha_opt.step()
                self.alpha = self.log_alpha.exp()

            if self.steps % self.cfg.target_update_freq == 0:
                for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                    target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)
            
            if self.cfg.use_per:
                with torch.no_grad():
                    q_mean = current_quantiles.mean(1)
                    t_mean = target_quantiles.mean(1)
                    new_priorities = torch.abs(q_mean - t_mean).cpu().numpy().flatten()
                    new_priorities = np.nan_to_num(new_priorities, nan=1.0)
                self.memory.update_priorities(indices, new_priorities)

            self.steps += 1
            return {
                'critic_loss': critic_loss.item(),
                'actor_loss': actor_loss.item(),
                'q_value': q_value.mean().item(),
                'alpha': self.alpha.item()
            }
            
        except Exception as e:
            logger.error(f"Error in update: {e}")
            return {}

# ═══════════════════════════════════════════════════════════════════════════
# MACRO ECONOMIC STATE WITH BEHAVIORAL ECONOMICS
# ═══════════════════════════════════════════════════════════════════════════

class MacroEconomicState:
    def __init__(self, cfg: IranConfig):
        self.cfg = cfg
        self.oil_price = 75.0
        self.oil_export_vol = 1.2
        self.forex_rate = 50000.0
        self.reserves = 30.0
        self.budget_deficit = 0.0
        
        self.cpi_food = cfg.econ.base_cpi
        self.cpi_housing = cfg.econ.base_cpi * 1.2
        
        # New: Behavioral Economics
        self.rumor_intensity = 0.0
        self.expectation_anchor = cfg.econ.base_cpi
        self.hoarding_index = 0.0
        self.black_market_ratio = 0.0
    
    def step(self, action_cost_billion_usd, sanctions_intensity, panic_level_avg):
        # Oil Revenue
        effective_sales = self.oil_export_vol * (1.0 - sanctions_intensity * 0.8)
        monthly_revenue = (effective_sales * self.oil_price * 30) / 1000.0
        
        # Forex Dynamics
        forex_demand = 2.0 + (self.cpi_food * 0.5)
        gap = forex_demand - monthly_revenue
        
        if gap > 0:
            if self.reserves > gap:
                self.reserves -= gap
            else:
                devaluation_pressure = (gap - self.reserves) / 10.0
                # Cap devaluation to prevent overflow
                devaluation_pressure = min(devaluation_pressure, 0.5)  # Max 50% per step
                self.forex_rate *= (1.0 + devaluation_pressure)
                # Hard cap on forex rate
                self.forex_rate = min(self.forex_rate, 500000.0)  # Max 500k IRR
                self.reserves = 0.0
        
        # Inflation Sources
        import_inflation = max(0, (self.forex_rate - 50000) / 50000)
        import_inflation = min(import_inflation, 2.0)  # Cap at 200%
        
        self.cpi_food += import_inflation * 0.1
        self.cpi_housing += import_inflation * 0.05
        
        # Cap inflation to prevent explosion
        self.cpi_food = min(self.cpi_food, 3.0)  # Max 300% inflation
        self.cpi_housing = min(self.cpi_housing, 3.0)
        
        # New: Speculative Rumors (random shocks)
        if np.random.random() < 0.15:  # 15% chance per step
            self.rumor_intensity = np.random.uniform(0.1, 0.3)
        else:
            self.rumor_intensity *= 0.8  # Decay
        
        # New: Panic Buying Impact
        if panic_level_avg > self.cfg.econ.panic_buying_threshold:
            hoarding_inflation = (panic_level_avg - self.cfg.econ.panic_buying_threshold) * 0.2
            hoarding_inflation = min(hoarding_inflation, 0.5)  # Cap hoarding effect
            self.cpi_food += hoarding_inflation
            self.hoarding_index = panic_level_avg
        else:
            self.hoarding_index *= 0.9
        
        # Cap again after all additions
        self.cpi_food = min(self.cpi_food, 3.0)
        self.cpi_housing = min(self.cpi_housing, 3.0)
        
        # New: Black Market Growth (when inflation is extreme)
        if self.cpi_food > 0.8:
            self.black_market_ratio = min(0.6, self.black_market_ratio + 0.05)
        else:
            self.black_market_ratio *= 0.95
        
        # Budget
        self.budget_deficit += action_cost_billion_usd - monthly_revenue
        # Cap deficit to prevent overflow
        self.budget_deficit = np.clip(self.budget_deficit, -1000, 1000)
        
        # New: Expectation Anchor (slow adjustment)
        self.expectation_anchor = (
            self.expectation_anchor * self.cfg.econ.expectation_anchor_decay +
            self.cpi_food * (1 - self.cfg.econ.expectation_anchor_decay)
        )

# ═══════════════════════════════════════════════════════════════════════════
# INFORMATION WARFARE ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class InformationWarfareEngine:
    def __init__(self, cfg: InformationWarfareParams, n_actors: int):
        self.cfg = cfg
        self.n = n_actors
        
        # State Variables
        self.external_agitation = cfg.external_agitation_baseline
        self.state_bot_activity = 0.0
        self.opposition_bot_activity = cfg.opposition_bot_efficiency
        self.info_saturation = 0.0
        
        # Viral Sparks (random influential posts)
        self.viral_nodes = []
        self.viral_duration = {}
    
    def step(self, internet_integrity, propaganda_budget, net_cut_active):
        # External Agitation (constant pressure from abroad)
        self.external_agitation = self.cfg.external_agitation_baseline * internet_integrity
        
        # State Bots (funded by propaganda budget)
        self.state_bot_activity = min(1.0, propaganda_budget * self.cfg.state_bot_efficiency)
        
        # Opposition Bots (reduced by net cuts)
        if net_cut_active:
            self.opposition_bot_activity *= 0.2
        else:
            self.opposition_bot_activity = self.cfg.opposition_bot_efficiency
        
        # Information Saturation (too many conflicting signals)
        total_info_flow = self.state_bot_activity + self.opposition_bot_activity + self.external_agitation
        if total_info_flow > 1.5:
            self.info_saturation = min(1.0, self.info_saturation + 0.1)
        else:
            self.info_saturation *= 0.9
        
        # Viral Sparks (random influencer moments)
        if np.random.random() < self.cfg.viral_spark_probability:
            new_viral = np.random.randint(0, self.n)
            self.viral_nodes.append(new_viral)
            self.viral_duration[new_viral] = 5  # Lasts 5 steps
        
        # Decay viral effects
        expired = []
        for node in self.viral_nodes:
            self.viral_duration[node] -= 1
            if self.viral_duration[node] <= 0:
                expired.append(node)
        
        for node in expired:
            self.viral_nodes.remove(node)
            del self.viral_duration[node]
    
    def get_influence_multiplier(self, node_id):
        """Returns influence boost for viral nodes"""
        if node_id in self.viral_nodes:
            return 100.0  # Massive temporary influence
        return 1.0

# ═══════════════════════════════════════════════════════════════════════════
# SECURITY FORCES MODULE (Trauma, Morale, Institutional Learning)
# ═══════════════════════════════════════════════════════════════════════════

class SecurityForcesModule:
    def __init__(self, cfg: RegimeHyperparams):
        self.cfg = cfg
        
        # Core States
        self.security_morale = cfg.initial_security_morale
        self.social_scars = 0.0  # Trauma Memory
        self.violence_index = 0.0
        self.radicalization_index = 0.0
        
        # Institutional Learning
        self.tactic_efficiency = {
            'riot_police': 1.0,
            'net_cut': 1.0,
            'infiltrate': 1.0,
            'targeted_arrest': 1.0
        }
        self.tactic_usage_count = defaultdict(int)
        
        # Loyalty Network
        self.loyalist_strength = 0.5
    
    def apply_repression(self, tactic_name, intensity, mobilization_level):
        """Apply repression and track consequences"""
        # Morale Drain
        morale_cost = self.cfg.morale_drain_per_operation * intensity
        if mobilization_level > 0.4:  # High mobilization = harder operations
            morale_cost *= 1.5
        
        self.security_morale = max(0.0, self.security_morale - morale_cost)
        
        # Violence Index (accumulates)
        violence_contribution = intensity * 0.3
        self.violence_index = min(1.0, self.violence_index + violence_contribution)
        self.violence_index *= 0.98  # Slow decay
        
        # Social Scars (long-term trauma)
        scar_increment = intensity * violence_contribution * 0.1
        self.social_scars = min(1.0, self.social_scars + scar_increment)
        self.social_scars *= 0.999  # Very slow decay (years)
        
        # Radicalization (if violence is extreme)
        if self.violence_index > 0.7:
            radicalization_rate = (self.violence_index - 0.7) * 0.2
            self.radicalization_index = min(1.0, self.radicalization_index + radicalization_rate)
        else:
            self.radicalization_index *= 0.95
        
        # Institutional Learning
        self.tactic_usage_count[tactic_name] += 1
        learning_bonus = min(0.3, self.tactic_usage_count[tactic_name] * 0.02)
        self.tactic_efficiency[tactic_name] = 1.0 + learning_bonus
        
        # Check for Defection Risk
        defection_risk = 0.0
        if self.security_morale < self.cfg.defection_threshold:
            defection_risk = (self.cfg.defection_threshold - self.security_morale) * 2.0
        
        return {
            'morale': self.security_morale,
            'violence': self.violence_index,
            'scars': self.social_scars,
            'radicalization': self.radicalization_index,
            'defection_risk': defection_risk,
            'efficiency': self.tactic_efficiency[tactic_name]
        }
    
    def recover_morale(self):
        """Gradual morale recovery during calm periods"""
        self.security_morale = min(
            self.cfg.initial_security_morale,
            self.security_morale + self.cfg.morale_recovery_rate
        )
    
    def strengthen_loyalists(self, budget):
        """Grey Zone Action: Distribute rents to loyalists"""
        self.loyalist_strength = min(1.0, self.loyalist_strength + budget * 0.1)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION ENGINE: SIMORGH V20
# ═══════════════════════════════════════════════════════════════════════════

class SimorghTwinV20:
    def __init__(self, config: SystemConfig = None, mode='train'):
        self.cfg = config or CFG
        self.mode = mode
        
        # Core Components
        self.pop = EnhancedPopulation(self.cfg)
        self._init_advanced_memory_arrays()
        
        self.econ = MacroEconomicState(self.cfg.iran)
        self.info_war = InformationWarfareEngine(self.cfg.iran.info_war, self.pop.n)
        self.security = SecurityForcesModule(self.cfg.iran.regime)
        
        # RL Agent (now with 10 actions)
        self.agent = SOTA_Agent(state_dim=20, action_dim=10, cfg=RL_CFG)
        
        # Simulation State
        self.day = 0
        self.sanctions = self.cfg.iran.sanctions_base
        self.legitimacy = self.cfg.iran.regime_legitimacy_base
        self.internet_integrity = 1.0
        self.net_cut_active = False
        self.cohesion = 0.5
        self.collective_courage = 0.0
        
        # History Buffer
        self.history_len = self.agent.seq_len
        self.state_buffer = deque(maxlen=self.history_len)
        
        # Geopolitical Events
        self.last_diplomatic_event = 0
        
        logger.info(f"✓ SimorghTwin V20 initialized with {self.pop.n} agents")
    
    def _init_advanced_memory_arrays(self):
        n = self.pop.n
        xp = self.pop.xp if self.cfg.use_gpu else np
        
        self.trauma_memory = xp.zeros(n, dtype=xp.float32)
        self.expectations = xp.ones(n, dtype=xp.float32) * 0.5
        self.activation_potential = xp.zeros(n, dtype=xp.float32)
        self.neighbor_wealth = xp.random.rand(n).astype(xp.float32)
        self.vpn_access = xp.random.choice([0.0, 1.0], n, p=[0.4, 0.6]).astype(xp.float32)
    
    def reset(self):
        self.pop.reset_state()
        self.day = 0
        self.econ = MacroEconomicState(self.cfg.iran)
        self.info_war = InformationWarfareEngine(self.cfg.iran.info_war, self.pop.n)
        self.security = SecurityForcesModule(self.cfg.iran.regime)
        
        self.sanctions = self.cfg.iran.sanctions_base
        self.legitimacy = self.cfg.iran.regime_legitimacy_base
        self.internet_integrity = 1.0
        self.net_cut_active = False
        self.collective_courage = 0.0
        
        if self.cfg.use_gpu:
            self.trauma_memory.fill(0)
            self.activation_potential.fill(0)
        
        initial_state = self._capture_snapshot()
        self.state_buffer.clear()
        for _ in range(self.history_len):
            self.state_buffer.append(initial_state)
        
        return self.get_state()
    
    def _capture_snapshot(self):
        if self.cfg.use_gpu:
            mob = float(self.pop.active.sum() / self.pop.n)
            panic_idx = float(self.pop.panic.mean())
            griev_idx = float(self.pop.grievance.mean())
            stamina_avg = float(self.pop.stamina.mean())
        else:
            mob = float(self.pop.active.sum() / self.pop.n)
            panic_idx = float(self.pop.panic.mean())
            griev_idx = float(self.pop.grievance.mean())
            stamina_avg = float(self.pop.stamina.mean())
        
        # Normalize and clip all values to prevent NaN
        state = np.array([
            np.clip(mob, 0, 1),  # 0
            np.clip(self.econ.cpi_food / 2.0, 0, 1),  # 1 - Normalized
            np.clip(self.econ.cpi_housing / 2.0, 0, 1),  # 2 - Normalized
            np.clip(self.econ.forex_rate / 200000.0, 0, 1),  # 3 - Normalized
            np.clip(self.sanctions, 0, 1),  # 4
            np.clip(self.legitimacy, 0, 1),  # 5
            np.clip(self.econ.reserves / 100.0, 0, 1),  # 6
            np.clip(self.internet_integrity, 0, 1),  # 7
            np.clip(griev_idx, 0, 1),  # 8
            np.clip(panic_idx, 0, 1),  # 9
            np.clip(self.econ.budget_deficit / 500.0, -1, 1),  # 10 - Normalized
            np.clip(float(self.day / 365.0), 0, 1),  # 11
            np.clip(self.security.security_morale, 0, 1),  # 12
            np.clip(self.security.social_scars, 0, 1),  # 13
            np.clip(self.security.violence_index, 0, 1),  # 14
            np.clip(self.security.radicalization_index, 0, 1),  # 15
            np.clip(self.info_war.external_agitation, 0, 1),  # 16
            np.clip(self.info_war.info_saturation, 0, 1),  # 17
            np.clip(stamina_avg, 0, 1),  # 18
            np.clip(self.collective_courage, 0, 1)  # 19
        ], dtype=np.float32)
        
        # Safety check: Replace any NaN with 0
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
        
        return state
    
    def get_state(self):
        return self._capture_snapshot()
    
    def load_agent(self, path):
        if os.path.exists(path):
            self.agent.actor.load_state_dict(torch.load(path))
            logger.info(f"Loaded agent from {path}")
    
    def step(self, action):
        action_idx = np.argmax(action)
        
        # Initialize action costs and effects
        cost_billion = 0.0
        repression_force = 0.0
        reform_signal = 0.0
        propaganda_budget = 0.0
        grey_zone_budget = 0.0
        targeted_arrest_intensity = 0.0
        cyber_defense_active = False
        diplomatic_effort = 0.0
        
        # Execute Action
        if action_idx == 0:  # WAIT
            pass
        
        elif action_idx == 1:  # RIOT POLICE
            repression_force = 0.8
            cost_billion = 0.5
            self.legitimacy -= 0.01
            sec_result = self.security.apply_repression('riot_police', 0.8, self.pop.active.sum() / self.pop.n)
            repression_force *= sec_result['efficiency']
        
        elif action_idx == 2:  # NET CUT
            self.internet_integrity = 0.1
            self.net_cut_active = True
            cost_billion = 2.0
            # Cyber Sabotage Effect: Disrupts services
            self.econ.cpi_food += 0.05  # Service disruption causes shortages
        
        elif action_idx == 3:  # INFILTRATE
            repression_force = 0.3
            cost_billion = 0.1
            sec_result = self.security.apply_repression('infiltrate', 0.3, self.pop.active.sum() / self.pop.n)
        
        elif action_idx == 4:  # REFORM
            cost_billion = 5.0
            reform_signal = 0.1
            self.legitimacy += 0.02
        
        elif action_idx == 5:  # PROPAGANDA
            propaganda_budget = 1.0
            cost_billion = 0.2
        
        elif action_idx == 6:  # GREY ZONE RENTS
            grey_zone_budget = 3.0
            cost_billion = 3.0
            self.security.strengthen_loyalists(grey_zone_budget)
            # Reduce grievance for loyalists only
            if not self.cfg.use_gpu:
                loyalist_mask = (self.pop.resilience_profiles == 0)
                self.pop.grievance[loyalist_mask] *= 0.8
        
        elif action_idx == 7:  # TARGETED ARRESTS
            targeted_arrest_intensity = 0.6
            cost_billion = 0.3
            sec_result = self.security.apply_repression('targeted_arrest', 0.6, self.pop.active.sum() / self.pop.n)
            # Remove leaders from network
            if not self.cfg.use_gpu:
                leader_mask = (self.pop.types == 1) & self.pop.active
                n_arrested = min(int(leader_mask.sum() * 0.3), 10)
                if n_arrested > 0:
                    arrested_indices = np.where(leader_mask)[0][:n_arrested]
                    self.pop.active[arrested_indices] = False
                    self.pop.grievance[arrested_indices] = 0.0
        
        elif action_idx == 8:  # CYBER DEFENSE
            cyber_defense_active = True
            cost_billion = 1.0
            # Reduces opposition bot effectiveness
            self.info_war.opposition_bot_activity *= 0.5
        
        elif action_idx == 9:  # DIPLOMATIC ENGAGEMENT
            diplomatic_effort = 1.0
            cost_billion = 2.0
            # Chance to reduce sanctions
            if np.random.random() < 0.3:
                self.sanctions = max(0.0, self.sanctions - 0.05)
                self.last_diplomatic_event = self.day
        
        # === STEP 1: MACRO ECONOMICS ===
        panic_avg = float(self.pop.panic.mean()) if not self.cfg.use_gpu else float(self.pop.panic.mean())
        self.econ.step(cost_billion, self.sanctions, panic_avg)
        
        # Diplomatic News Shock (10% chance)
        if np.random.random() < 0.10 and (self.day - self.last_diplomatic_event) > 30:
            shock_type = np.random.choice(['positive', 'negative'])
            if shock_type == 'positive':
                self.econ.cpi_food *= 0.90  # Good news reduces inflation
                self.legitimacy += 0.03
            else:
                self.econ.cpi_food *= 1.10  # Bad news increases inflation
                self.sanctions += 0.05
            self.last_diplomatic_event = self.day
        
        # === STEP 2: INFORMATION WARFARE ===
        self.info_war.step(self.internet_integrity, propaganda_budget, self.net_cut_active)
        
        # === STEP 3: BEHAVIORAL ECONOMICS - SPECULATIVE SHOCKS ===
        if not self.cfg.use_gpu:
            new_expectations, hoarding_demand = kernel_speculative_shock(
                self.pop.resources,
                self.expectations,
                self.econ.cpi_food,
                self.pop.panic,
                self.econ.rumor_intensity,
                self.cfg.iran.econ.speculation_amplifier,
                self.cfg.iran.econ.panic_buying_threshold
            )
            self.expectations = new_expectations
            
            # Hoarding causes supply shock inflation
            if hoarding_demand.sum() > 0:
                hoarding_inflation = hoarding_demand.mean() * 0.1
                self.econ.cpi_food += hoarding_inflation
        
        # === STEP 4: GRIEVANCE CALCULATION ===
        if not self.cfg.use_gpu:
            new_grievance = kernel_advanced_grievance_v2(
                self.pop.grievance,
                self.trauma_memory,
                self.security.social_scars,
                self.pop.resources,
                self.expectations,
                self.pop.ages,
                self.pop.city_ids,
                self.pop.urban,
                self.pop.ethnicities,
                self.econ.cpi_food,
                self.econ.cpi_housing,
                self.econ.forex_rate,
                self.sanctions,
                self.legitimacy,
                np.array(self.cfg.iran.regional_development_index),
                np.array(self.cfg.iran.soc.ethnic_friction_matrix[0]),
                self.neighbor_wealth,
                self.vpn_access,
                self.info_war.external_agitation,
                self.econ.cpi_food * (1.0 + self.econ.rumor_intensity),
                self.pop.stamina
            )
            self.pop.grievance = new_grievance
        
        # === STEP 5: PANIC PROPAGATION ===
        time_of_day = (self.day * 24.0) % 24.0
        violence_vis = self.security.violence_index * (1.0 - self.internet_integrity * 0.5)
        
        if not self.cfg.use_gpu:
            urban_density = self.pop.urban.astype(np.float32) * 0.8
            new_panic = kernel_entropy_panic_v2(
                self.pop.panic,
                self.pop.adj_matrix.indices,
                self.pop.adj_matrix.indptr,
                self.internet_integrity,
                violence_vis,
                time_of_day,
                urban_density,
                self.info_war.info_saturation,
                self.info_war.state_bot_activity,
                self.info_war.opposition_bot_activity
            )
            self.pop.panic = new_panic
        
        # === STEP 6: COLLECTIVE COURAGE (Avalanche Effect) ===
        current_mobilization = self.pop.active.sum() / self.pop.n
        if current_mobilization > 0.1:
            self.collective_courage = min(1.0, current_mobilization * 2.0)
        else:
            self.collective_courage *= 0.95
        
        # === STEP 7: COMPLEX CONTAGION WITH THRESHOLD MODEL ===
        global_signal = (reform_signal * -1.0) + (propaganda_budget * -0.3)
        if self.internet_integrity > 0.8:
            global_signal += 0.3  # Open internet helps coordination
        
        if not self.cfg.use_gpu:
            police_distribution = np.full(self.pop.n, repression_force, dtype=np.float32)
            
            # Targeted arrests reduce police presence overall (resources diverted)
            if targeted_arrest_intensity > 0:
                police_distribution *= 0.7
            
            new_active, new_potential = kernel_threshold_diffusion(
                self.pop.active,
                self.activation_potential,
                self.pop.activation_thresholds,
                self.pop.adj_matrix.indices,
                self.pop.adj_matrix.indptr,
                self.pop.adj_matrix.data,
                self.pop.grievance,
                self.pop.stubbornness,
                self.pop.charisma,
                police_distribution,
                global_signal,
                self.collective_courage,
                self.security.radicalization_index,
                self.pop.stamina
            )
            self.pop.active = new_active
            self.activation_potential = new_potential
        
        # === STEP 8: PROTEST FATIGUE ===
        if not self.cfg.use_gpu:
            # Drain stamina for active protesters
            stamina_drain = self.pop.active.astype(np.float32) * self.cfg.iran.soc.stamina_drain_rate
            self.pop.stamina -= stamina_drain
            
            # Recover stamina for inactive
            stamina_recovery = (~self.pop.active).astype(np.float32) * self.cfg.iran.soc.stamina_recovery_rate
            self.pop.stamina += stamina_recovery
            
            # Shock events can restore stamina (martyrdom effect)
            if self.security.violence_index > 0.8:
                shock_restoration = np.random.random(self.pop.n) * 0.3
                self.pop.stamina = np.minimum(1.0, self.pop.stamina + shock_restoration)
            
            self.pop.stamina = np.clip(self.pop.stamina, 0.0, 1.0)
        
        # === STEP 9: RADICALIZATION (Network Structure Change) ===
        if self.security.radicalization_index > 0.6:
            # Network becomes cellular (harder to infiltrate)
            # In practice, this reduces the effectiveness of INFILTRATE action
            # Already handled in security module efficiency tracking
            pass
        
        # === STEP 10: GREY ZONE EFFECTS (Loyalist Interference) ===
        if grey_zone_budget > 0 and not self.cfg.use_gpu:
            # Loyalists try to suppress protests in their neighborhoods
            loyalist_indices = np.where(self.pop.resilience_profiles == 0)[0]
            for loyalist_idx in loyalist_indices[:min(50, len(loyalist_indices))]:
                # Get neighbors
                start = self.pop.adj_matrix.indptr[loyalist_idx]
                end = self.pop.adj_matrix.indptr[loyalist_idx + 1]
                neighbors = self.pop.adj_matrix.indices[start:end]
                
                # Reduce their activation potential
                for neighbor in neighbors:
                    if self.pop.active[neighbor]:
                        # Chance to discourage
                        if np.random.random() < self.security.loyalist_strength * 0.3:
                            self.activation_potential[neighbor] -= 0.2
        
        # === STEP 11: SECURITY FORCE RECOVERY ===
        if repression_force == 0 and targeted_arrest_intensity == 0:
            self.security.recover_morale()
        
        # === STEP 12: INTERNET RECOVERY ===
        self.day += 1
        if not self.net_cut_active:
            self.internet_integrity = min(1.0, self.internet_integrity + 0.1)
        self.net_cut_active = False  # Reset for next step
        
        # === STEP 13: CALCULATE REWARD ===
        current_state = self._capture_snapshot()
        self.state_buffer.append(current_state)
        
        reward = self._calculate_complex_reward_v2(current_state, cost_billion, action_idx)
        
        # === STEP 14: TERMINATION CONDITIONS ===
        done = False
        termination_reason = None
        
        # Regime Collapse
        if current_state[0] > 0.75:  # Mobilization > 75%
            done = True
            termination_reason = "regime_collapse"
        
        # Economic Collapse
        if self.econ.reserves <= 0 and self.econ.budget_deficit > 200:
            done = True
            termination_reason = "economic_collapse"
        
        # Security Force Defection
        if self.security.security_morale < 0.15:
            done = True
            termination_reason = "security_defection"
        
        # Radicalization Critical
        if self.security.radicalization_index > 0.9:
            done = True
            termination_reason = "civil_war"
        
        # Time Limit
        if self.day >= self.cfg.simulation_days:
            done = True
            termination_reason = "time_limit"
        
        info = {
            'cost': cost_billion,
            'deficit': self.econ.budget_deficit,
            'action_idx': action_idx,
            'action_name': ACT_NAMES[action_idx],
            'morale': self.security.security_morale,
            'scars': self.security.social_scars,
            'violence': self.security.violence_index,
            'radicalization': self.security.radicalization_index,
            'termination_reason': termination_reason,
            'stamina_avg': current_state[18],
            'collective_courage': self.collective_courage
        }
        
        return current_state, reward, done, info
    
    def _calculate_complex_reward_v2(self, state, cost_billion, action_idx):
        """Enhanced reward function with multiple objectives"""
        mob = state[0]
        inflation = state[1]
        legitimacy = state[5]
        reserves = state[6]
        morale = state[12]
        scars = state[13]
        violence = state[14]
        radicalization = state[15]
        
        # Primary: Prevent Mobilization (exponential penalty)
        r_mob = -np.power(mob * 10.0, 2.5)
        
        # Economic Stability
        r_cost = -cost_billion * 0.1
        r_inf = -max(0, inflation - 0.4) * 5.0
        r_reserves = (reserves - 0.2) * 2.0
        
        # Political Legitimacy
        r_leg = (legitimacy - 0.2) * 3.0
        
        # Security Force Health
        r_morale = morale * 2.0
        
        # Long-term Consequences (Trauma)
        r_scars = -scars * 10.0  # Heavy penalty for social trauma
        
        # Violence Penalty
        r_violence = -violence * 8.0
        
        # Radicalization Penalty (catastrophic)
        r_radical = -np.power(radicalization * 10.0, 2.0)
        
        # Action-specific penalties
        r_action = 0.0
        if action_idx == 2:  # NET CUT - very costly politically
            r_action = -5.0
        elif action_idx == 4:  # REFORM - reward for constructive action
            r_action = 3.0
        
        total_reward = (
            r_mob * 0.30 +
            (r_cost + r_inf + r_reserves) * 0.15 +
            r_leg * 0.10 +
            r_morale * 0.05 +
            r_scars * 0.15 +
            r_violence * 0.10 +
            r_radical * 0.10 +
            r_action * 0.05
        )
        
        return total_reward
    
    def train(self, n_episodes=500):
        logger.info(f"🚀 Starting SOTA Training Protocol V20 ({n_episodes} episodes)...")
        checkpoint_dir = Path(self.cfg.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        episode_rewards = []
        episode_lengths = []
        
        for ep in range(n_episodes):
            state = self.reset()
            episode_reward = 0
            ep_costs = 0
            steps = 0
            
            while True:
                if len(self.agent.memory) < RL_CFG.warmup_steps:
                    # Random exploration
                    action_idx = np.random.randint(0, 10)
                    action = np.zeros(10)
                    action[action_idx] = 1.0
                else:
                    action = self.agent.select_action(state, evaluate=False)
                
                next_state, reward, done, info = self.step(action)
                self.agent.memory.push(state, action, reward, next_state, done)
                
                episode_reward += reward
                ep_costs += info['cost']
                steps += 1
                
                if len(self.agent.memory) >= RL_CFG.warmup_steps:
                    metrics = self.agent.update(RL_CFG.batch_size)
                
                if done:
                    break
                state = next_state
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            
            # Logging
            if (ep + 1) % 5 == 0:
                avg_reward = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards)
                avg_length = np.mean(episode_lengths[-20:]) if len(episode_lengths) >= 20 else np.mean(episode_lengths)
                
                logger.info(
                    f"Ep {ep+1:03d} | R: {episode_reward:7.1f} (Avg: {avg_reward:7.1f}) | "
                    f"Steps: {steps:03d} | Cost: ${ep_costs:4.1f}B | "
                    f"Mob_Max: {state[0]:.2%} | Morale: {state[12]:.2f} | "
                    f"Scars: {state[13]:.2f} | Reason: {info.get('termination_reason', 'N/A')}"
                )
            
            # Save checkpoints
            if (ep + 1) % 50 == 0:
                torch.save(
                    self.agent.actor.state_dict(),
                    str(checkpoint_dir / f'actor_ep{ep+1}.pt')
                )
                logger.info(f"💾 Checkpoint saved at episode {ep+1}")
        
        # Save final model
        torch.save(
            self.agent.actor.state_dict(),
            str(checkpoint_dir / 'actor_final.pt')
        )
        logger.info("✅ Training completed!")

# ═══════════════════════════════════════════════════════════════════════════
# HISTORICAL VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

class HistoricalValidator:
    def __init__(self):
        self.events = {
            '2009_green': {
                'name': '2009 Green Movement',
                'duration': 180,
                'trigger_day': 10,
                'params': {
                    'inflation': 0.15,
                    'sanctions': 0.20,
                    'legitimacy': 0.50,
                    'repression': 0.60
                },
                'observed_mobilization': [0.0]*10 + [0.02, 0.08, 0.15, 0.25, 0.32, 0.35, 0.38, 0.40, 0.42, 0.43] + [0.42]*160
            },
            '2022_mahsa': {
                'name': '2022 Mahsa Amini Protests',
                'duration': 150,
                'trigger_day': 8,
                'params': {
                    'inflation': 0.50,
                    'sanctions': 0.40,
                    'legitimacy': 0.30,
                    'repression': 0.75
                },
                'observed_mobilization': [0.0]*8 + [0.03, 0.12, 0.22, 0.32, 0.38, 0.42, 0.45, 0.47] + [0.45]*134
            }
        }
    
    def validate(self, model, event_name='2022_mahsa', plot=False):
        if event_name not in self.events:
            logger.error(f"Unknown event: {event_name}")
            return None
        
        event = self.events[event_name]
        logger.info(f"📊 Validating against: {event['name']}")
        
        # Configure environment
        env_params = event['params']
        model.econ.cpi_food = env_params['inflation']
        model.sanctions = env_params['sanctions']
        model.legitimacy = env_params['legitimacy']
        
        sim_mobilization = []
        model.reset()
        
        for day in range(event['duration']):
            # Trigger event
            if day == event['trigger_day']:
                n_seed = int(model.pop.n * 0.001)
                seed_indices = np.random.choice(model.pop.n, n_seed, replace=False)
                
                # Fix: Use model.pop.active and model.activation_potential
                if isinstance(model.pop.active, np.ndarray):
                    model.pop.active[seed_indices] = True
                    model.activation_potential[seed_indices] = 0.8
                else:
                    # GPU case
                    model.pop.active[seed_indices] = True
                    model.activation_potential[seed_indices] = 0.8
            
            # Baseline action (WAIT with occasional repression)
            current_mob = sim_mobilization[-1] if sim_mobilization else 0.0
            
            if current_mob > 0.2:
                action = np.zeros(10)
                action[1] = 1.0  # RIOT POLICE
            else:
                action = np.zeros(10)
                action[0] = 1.0  # WAIT
            
            _, _, done, _ = model.step(action)
            sim_mobilization.append(model.get_state()[0])
            
            if done:
                # Pad remaining days
                sim_mobilization.extend([sim_mobilization[-1]] * (event['duration'] - day - 1))
                break
        
        # Compare
        obs = event['observed_mobilization']
        min_len = min(len(obs), len(sim_mobilization))
        obs = obs[:min_len]
        sim = sim_mobilization[:min_len]
        
        rmse = np.sqrt(mean_squared_error(obs, sim))
        correlation = np.corrcoef(obs, sim)[0, 1] if len(obs) > 1 else 0.0
        
        logger.info(f"✓ Validation Results: RMSE={rmse:.4f}, Correlation={correlation:.4f}")
        
        return {
            'event': event_name,
            'rmse': rmse,
            'correlation': correlation,
            'observed': obs,
            'simulated': sim
        }

# ═══════════════════════════════════════════════════════════════════════════
# DASHBOARD (WARROOM)
# ═══════════════════════════════════════════════════════════════════════════

def render_radar_chart(state):
    categories = ['Mobilization', 'Inflation', 'Sanctions', 'Legitimacy', 'Morale', 'Trauma']
    values = [
        state[0],  # Mobilization
        min(1.0, state[1] / 1.5),  # Inflation (normalized)
        state[4],  # Sanctions
        state[5],  # Legitimacy
        state[12],  # Security Morale
        state[13]  # Social Scars
    ]
    values = [max(0.0, min(1.0, v)) for v in values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Current State',
        line_color='#ff0000' if state[0] > 0.5 else '#ffaa00' if state[0] > 0.3 else '#00ff00'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        height=300
    )
    
    return fig

def run_warroom_dashboard():
    st.set_page_config(
        page_title="SIMORGH V20 COMMAND CENTER",
        page_icon="🦅",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.markdown("""
        <style>
        .stApp {background-color: #0e1117;}
        .metric-card {
            background-color: #262730;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #00ff00;
        }
        .critical {border-left-color: #ff0000 !important;}
        .warning {border-left-color: #ffaa00 !important;}
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🦅 SIMORGH V20: NATIONAL SECURITY DIGITAL TWIN")
    st.caption("Advanced Crisis Simulation with Information Warfare & Behavioral Economics")
    
    # Initialize session state
    if 'model' not in st.session_state:
        with st.spinner("Initializing simulation engine..."):
            st.session_state.model = SimorghTwinV20(mode='eval')
            st.session_state.history = defaultdict(list)
            st.session_state.day = 0
            st.session_state.action_history = []
    
    model = st.session_state.model
    state = model.get_state()
    
    # === TOP KPI METRICS ===
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        risk_class = "critical" if state[0] > 0.5 else "warning" if state[0] > 0.3 else ""
        st.markdown(f'<div class="metric-card {risk_class}">', unsafe_allow_html=True)
        st.metric("🚨 Mobilization Risk", f"{state[0]:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📈 Inflation (Food)", f"{state[1]:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("⚖️ Legitimacy", f"{state[5]:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        morale_class = "critical" if state[12] < 0.3 else "warning" if state[12] < 0.5 else ""
        st.markdown(f'<div class="metric-card {morale_class}">', unsafe_allow_html=True)
        st.metric("🛡️ Security Morale", f"{state[12]:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="metric-card critical">', unsafe_allow_html=True)
        st.metric("💔 Social Trauma", f"{state[13]:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # === MAIN DASHBOARD ===
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📊 Threat Assessment Radar")
        st.plotly_chart(render_radar_chart(state), use_container_width=True)
        
        # Mobilization History
        if len(st.session_state.history['day']) > 0:
            st.subheader("📈 Mobilization Trend")
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=st.session_state.history['day'],
                y=st.session_state.history['mobilization'],
                mode='lines',
                name='Mobilization',
                line=dict(color='#ff5555', width=2)
            ))
            fig_trend.update_layout(
                height=200,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig_trend, use_container_width=True)
    
    with col_right:
        st.subheader("🎯 Action Selection")
        
        # Action buttons in grid
        col_a, col_b = st.columns(2)
        
        selected_action = 0
        
        with col_a:
            if st.button("⏳ WAIT", use_container_width=True):
                selected_action = 0
            if st.button("🚔 RIOT POLICE", use_container_width=True):
                selected_action = 1
            if st.button("✂️ NET CUT", use_container_width=True):
                selected_action = 2
            if st.button("🕵️ INFILTRATE", use_container_width=True):
                selected_action = 3
            if st.button("🤝 REFORM", use_container_width=True):
                selected_action = 4
        
        with col_b:
            if st.button("📢 PROPAGANDA", use_container_width=True):
                selected_action = 5
            if st.button("💰 GREY ZONE", use_container_width=True):
                selected_action = 6
            if st.button("🎯 TARGETED ARREST", use_container_width=True):
                selected_action = 7
            if st.button("🛡️ CYBER DEFENSE", use_container_width=True):
                selected_action = 8
            if st.button("🕊️ DIPLOMACY", use_container_width=True):
                selected_action = 9
        
        if st.button("▶️ EXECUTE & ADVANCE", type="primary", use_container_width=True):
            action_vec = np.zeros(10)
            action_vec[selected_action] = 1.0
            
            with st.spinner("Simulating consequences..."):
                next_state, reward, done, info = model.step(action_vec)
                st.session_state.day += 1
                st.session_state.history['day'].append(st.session_state.day)
                st.session_state.history['mobilization'].append(next_state[0])
                st.session_state.action_history.append(ACT_NAMES[selected_action])
                
                if done:
                    st.error(f"⚠️ SIMULATION TERMINATED: {info.get('termination_reason', 'Unknown')}")
                    if st.button("🔄 RESET SIMULATION"):
                        st.session_state.model.reset()
                        st.session_state.history = defaultdict(list)
                        st.session_state.day = 0
                        st.session_state.action_history = []
                        st.rerun()
                else:
                    st.rerun()
        
        # Recent Actions
        st.subheader("📜 Recent Actions")
        if st.session_state.action_history:
            for i, action in enumerate(st.session_state.action_history[-5:][::-1]):
                st.caption(f"Day {st.session_state.day - i}: {action}")
    
    # === ADVANCED METRICS ===
    st.divider()
    st.subheader("🔬 Advanced Metrics")
    
    col_adv1, col_adv2, col_adv3, col_adv4 = st.columns(4)
    
    with col_adv1:
        st.metric("🌐 Internet Integrity", f"{state[7]:.1%}")
        st.metric("📡 External Agitation", f"{state[16]:.1%}")
    
    with col_adv2:
        st.metric("⚡ Violence Index", f"{state[14]:.1%}")
        st.metric("🔥 Radicalization", f"{state[15]:.1%}")
    
    with col_adv3:
        st.metric("💪 Avg Stamina", f"{state[18]:.1%}")
        st.metric("🤜 Collective Courage", f"{state[19]:.1%}")
    
    with col_adv4:
        st.metric("💵 Forex Rate", f"{state[3] * 100000:.0f} IRR")
        st.metric("🏦 Reserves", f"${state[6] * 100:.1f}B")

# ═══════════════════════════════════════════════════════════════════════════
# API SERVER
# ═══════════════════════════════════════════════════════════════════════════

def run_production_api(port=8000, checkpoint=None):
    if not FASTAPI_AVAILABLE:
        logger.error("FastAPI not installed. Run: pip install fastapi uvicorn")
        return
    
    app = FastAPI(title="Simorgh V20 API", version="20.0.0")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Global model instance
    model = SimorghTwinV20(mode='eval')
    if checkpoint:
        model.load_agent(checkpoint)
    
    class ActionRequest(BaseModel):
        action_idx: int = Field(..., ge=0, le=9, description="Action index (0-9)")
    
    @app.get("/")
    def read_root():
        return {
            "status": "online",
            "version": "20.0.0",
            "model": "Simorgh V20",
            "actions": ACT_NAMES
        }
    
    @app.get("/state")
    def get_state():
        state = model.get_state()
        return {
            "mobilization": float(state[0]),
            "inflation_food": float(state[1]),
            "inflation_housing": float(state[2]),
            "forex_rate": float(state[3] * 100000),
            "sanctions": float(state[4]),
            "legitimacy": float(state[5]),
            "reserves": float(state[6] * 100),
            "internet_integrity": float(state[7]),
            "grievance_avg": float(state[8]),
            "panic_avg": float(state[9]),
            "budget_deficit": float(state[10] * 100),
            "day": int(state[11] * 365),
            "security_morale": float(state[12]),
            "social_scars": float(state[13]),
            "violence_index": float(state[14]),
            "radicalization": float(state[15]),
            "external_agitation": float(state[16]),
            "info_saturation": float(state[17]),
            "stamina_avg": float(state[18]),
            "collective_courage": float(state[19])
        }
    
    @app.post("/step")
    def step_simulation(request: ActionRequest):
        action_vec = np.zeros(10)
        action_vec[request.action_idx] = 1.0
        
        try:
            next_state, reward, done, info = model.step(action_vec)
            
            return {
                "success": True,
                "state": {
                    "mobilization": float(next_state[0]),
                    "inflation_food": float(next_state[1]),
                    "legitimacy": float(next_state[5]),
                    "security_morale": float(next_state[12]),
                    "social_scars": float(next_state[13])
                },
                "reward": float(reward),
                "done": done,
                "info": info
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/reset")
    def reset_simulation():
        model.reset()
        return {"success": True, "message": "Simulation reset"}
    
    logger.info(f"🚀 Starting Simorgh V20 API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SIMORGH V20: Advanced National Security Digital Twin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Simorgh_v20.py --mode train --episodes 500
  python Simorgh_v20.py --mode warroom
  streamlit run Simorgh_v20.py -- --mode warroom
  python Simorgh_v20.py --mode validate --event 2022_mahsa
  python Simorgh_v20.py --mode api --port 8000 --checkpoint ./checkpoints/actor_final.pt
        """
    )
    
    parser.add_argument(
        '--mode', type=str, default='warroom',
        choices=['train', 'warroom', 'api', 'validate'],
        help='Operating mode'
    )
    
    parser.add_argument(
        '--episodes', type=int, default=500,
        help='Number of training episodes (train mode)'
    )
    
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to trained agent checkpoint'
    )
    
    parser.add_argument(
        '--port', type=int, default=8000,
        help='Server port (api mode)'
    )
    
    parser.add_argument(
        '--event', type=str, default='2022_mahsa',
        choices=['2009_green', '2022_mahsa'],
        help='Historical event for validation'
    )
    
    parser.add_argument(
        '--n-actors', type=int, default=None,
        help='Override number of actors in simulation'
    )
    
    # Handle Streamlit's extra arguments
    try:
        args = parser.parse_args()
    except SystemExit:
        # If argparse fails (due to streamlit args), use defaults
        args = parser.parse_args([])
    
    # Apply overrides
    if args.n_actors:
        CFG.n_actors = args.n_actors
        logger.info(f"Overriding n_actors to {args.n_actors}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # MODE: TRAINING
    # ═══════════════════════════════════════════════════════════════════════
    if args.mode == 'train':
        logger.info("=" * 80)
        logger.info("🚀 SIMORGH V20 - TRAINING MODE")
        logger.info("=" * 80)
        
        model = SimorghTwinV20(mode='train')
        
        if args.checkpoint:
            model.load_agent(args.checkpoint)
            logger.info(f"📥 Loaded checkpoint: {args.checkpoint}")
        
        model.train(n_episodes=args.episodes)
    
    # ═══════════════════════════════════════════════════════════════════════
    # MODE: WARROOM (Dashboard)
    # ═══════════════════════════════════════════════════════════════════════
    elif args.mode == 'warroom':
        if not STREAMLIT_AVAILABLE:
            logger.error("❌ Streamlit not installed. Run: pip install streamlit plotly")
            return
        
        # Check if running inside Streamlit
        is_running_in_streamlit = False
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            if get_script_run_ctx() is not None:
                is_running_in_streamlit = True
        except:
            try:
                import streamlit as st
                if hasattr(st, '_is_running_with_streamlit') and st._is_running_with_streamlit:
                    is_running_in_streamlit = True
            except:
                pass
        
        if is_running_in_streamlit:
            # Already inside Streamlit, just run the dashboard
            run_warroom_dashboard()
        else:
            # Launch Streamlit
            logger.info("=" * 80)
            logger.info("🦅 LAUNCHING SIMORGH V20 WARROOM DASHBOARD")
            logger.info("=" * 80)
            logger.info("Dashboard will open in your browser...")
            
            try:
                from streamlit.web import cli as stcli
                sys.argv = ["streamlit", "run", __file__, "--", "--mode", "warroom"]
                sys.exit(stcli.main())
            except ImportError:
                logger.error("Could not auto-launch Streamlit.")
                logger.error(f"Please run manually: streamlit run {__file__} -- --mode warroom")
    
    # ═══════════════════════════════════════════════════════════════════════
    # MODE: API
    # ═══════════════════════════════════════════════════════════════════════
    elif args.mode == 'api':
        logger.info("=" * 80)
        logger.info("🌐 SIMORGH V20 - API MODE")
        logger.info("=" * 80)
        
        run_production_api(port=args.port, checkpoint=args.checkpoint)
    
    # ═══════════════════════════════════════════════════════════════════════
    # MODE: VALIDATION
    # ═══════════════════════════════════════════════════════════════════════
    elif args.mode == 'validate':
        logger.info("=" * 80)
        logger.info("📊 SIMORGH V20 - HISTORICAL VALIDATION")
        logger.info("=" * 80)
        
        model = SimorghTwinV20(mode='eval')
        if args.checkpoint:
            model.load_agent(args.checkpoint)
        
        validator = HistoricalValidator()
        results = validator.validate(model, event_name=args.event)
        
        if results:
            logger.info("")
            logger.info("=" * 80)
            logger.info("VALIDATION RESULTS")
            logger.info("=" * 80)
            logger.info(f"Event: {results['event']}")
            logger.info(f"RMSE: {results['rmse']:.4f}")
            logger.info(f"Correlation: {results['correlation']:.4f}")
            logger.info("=" * 80)

if __name__ == "__main__":
    # Display banner
    banner = """
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║   ███████╗██╗███╗   ██╗ ██████╗ ██████╗  ██████╗ ██╗  ██╗          ║
    ║   ██╔════╝██║████╗  ██║██╔═══██╗██╔══██╗██╔════╝ ██║  ██║          ║
    ║   ███████╗██║██╔██╗ ██║██║   ██║██████╔╝██║  ███╗███████║          ║
    ║   ╚════██║██║██║╚██╗██║██║   ██║██╔══██╗██║   ██║██╔══██║          ║
    ║   ███████║██║██║ ╚████║╚██████╔╝██║  ██║╚██████╔╝██║  ██║          ║
    ║   ╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝          ║
    ║                                                                       ║
    ║              NATIONAL SECURITY DIGITAL TWIN V20                      ║
    ║                                                                       ║
    ║   Features:                                                          ║
    ║   • Information Warfare Engine                                       ║
    ║   • Behavioral Economics (Rumors, Panic Buying)                      ║
    ║   • Trauma Memory & Social Scars                                     ║
    ║   • Security Force Morale & Institutional Learning                   ║
    ║   • Threshold Model of Collective Action                             ║
    ║   • Protest Fatigue & Radicalization Dynamics                        ║
    ║   • Grey Zone Operations                                             ║
    ║   • Geopolitical Event Shocks                                        ║
    ║   • 10 Strategic Actions                                             ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)
    
    main()