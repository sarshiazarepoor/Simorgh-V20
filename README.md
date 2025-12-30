# 🦅 SIMORGH V20: National Security Digital Twin

<div dir="rtl">

## 🎯 درباره پروژه

**سیمرغ ویرایش 20** یک دوقلوی دیجیتال پیشرفته امنیت ملی است که با استفاده از هوش مصنوعی، یادگیری تقویتی عمیق، و شبیه‌سازی عامل‌محور، پویایی‌های پیچیده اجتماعی-سیاسی-اقتصادی را مدل‌سازی می‌کند.

### ویژگی‌های کلیدی

- 🧠 **هوش مصنوعی پیشرفته**: معماری Transformer با یادگیری تقویتی توزیعی (Distributional RL)
- 🌐 **جنگ اطلاعاتی**: مدل‌سازی ربات‌های شبکه‌های اجتماعی، تحریکات خارجی، و اشباع اطلاعاتی
- 📊 **اقتصاد رفتاری**: شایعات، خرید هراسان، و انتظارات تورمی
- 💔 **حافظه تروما**: اسکارهای اجتماعی و خاطرات جمعی خشونت
- 🛡️ **نیروهای امنیتی**: روحیه، یادگیری سازمانی، و خطر انشقاق
- 🔥 **رادیکالیزاسیون**: مدل آستانه‌ای کنش جمعی و خستگی اعتراضی
- ⚡ **10 اقدام استراتژیک**: از انتظار تا اصلاحات، از عملیات منطقه خاکستری تا دیپلماسی

</div>

---

## 🚀 Installation

### Prerequisites

```bash
# Python 3.8+
python --version
```

### Core Dependencies

```bash
pip install torch torchvision
pip install numpy pandas scipy scikit-learn
pip install networkx
pip install numba  # Optional: for GPU acceleration
```

### Dashboard & API (Optional)

```bash
# For Warroom Dashboard
pip install streamlit plotly

# For Production API
pip install fastapi uvicorn pydantic
```

### Quick Install (All Dependencies)

```bash
pip install torch numpy pandas scipy scikit-learn networkx streamlit plotly fastapi uvicorn pydantic numba
```

---

## 📖 Usage

### 1️⃣ Training Mode

Train the RL agent to learn optimal crisis management strategies:

```bash
python Simorgh.py --mode train --episodes 500
```

**Options:**
- `--episodes`: Number of training episodes (default: 500)
- `--n-actors`: Override population size (default: 2000)
- `--checkpoint`: Resume from checkpoint

**Output:**
- Checkpoints saved to `./checkpoints/`
- Training logs display episode rewards, mobilization, and termination reasons

---

### 2️⃣ Warroom Dashboard (Interactive)

Launch a real-time command center interface:

```bash
streamlit run Simorgh.py -- --mode warroom
```

Or:

```bash
python Simorgh.py --mode warroom
```

**Features:**
- 📊 Real-time threat assessment radar
- 🎯 10 strategic action buttons
- 📈 Mobilization trend visualization
- 🔬 Advanced metrics (violence, radicalization, stamina, etc.)
- 📜 Action history tracker

**Dashboard Metrics:**
- 🚨 Mobilization Risk
- 📈 Food Inflation
- ⚖️ Political Legitimacy
- 🛡️ Security Force Morale
- 💔 Social Trauma Index
- 🌐 Internet Integrity
- ⚡ Violence Index
- 🔥 Radicalization Level

---

### 3️⃣ Production API

Deploy as a REST API for integration with other systems:

```bash
python Simorgh.py --mode api --port 8000 --checkpoint ./checkpoints/actor_final.pt
```

**API Endpoints:**

```bash
# Get current state
curl http://localhost:8000/state

# Execute action (0-9)
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_idx": 1}'

# Reset simulation
curl -X POST http://localhost:8000/reset
```

**Response Example:**
```json
{
  "success": true,
  "state": {
    "mobilization": 0.23,
    "inflation_food": 0.45,
    "legitimacy": 0.62,
    "security_morale": 0.78,
    "social_scars": 0.12
  },
  "reward": -15.4,
  "done": false,
  "info": {...}
}
```

---

### 4️⃣ Historical Validation

Validate model against real historical events:

```bash
# 2022 Mahsa Amini Protests
python Simorgh.py --mode validate --event 2022_mahsa

# 2009 Green Movement
python Simorgh.py --mode validate --event 2009_green
```

**Validation Metrics:**
- RMSE (Root Mean Square Error)
- Correlation coefficient
- Visual comparison of observed vs simulated mobilization

---

## 🎮 Strategic Actions

| ID | Action | Description | Cost | Effect |
|----|--------|-------------|------|--------|
| 0 | ⏳ WAIT | No intervention | $0B | Morale recovery |
| 1 | 🚔 RIOT POLICE | Deploy riot control | $0.5B | High repression, morale drain |
| 2 | ✂️ NET CUT | Internet shutdown | $2.0B | Disrupts coordination, economic damage |
| 3 | 🕵️ INFILTRATE | Intelligence operations | $0.1B | Low-key repression |
| 4 | 🤝 REFORM | Political concessions | $5.0B | Boosts legitimacy |
| 5 | 📢 PROPAGANDA | State media campaign | $0.2B | Counter-narrative |
| 6 | 💰 GREY ZONE | Rent distribution to loyalists | $3.0B | Strengthens regime support |
| 7 | 🎯 TARGETED ARREST | Remove protest leaders | $0.3B | Decapitates movement |
| 8 | 🛡️ CYBER DEFENSE | Counter opposition bots | $1.0B | Reduces digital agitation |
| 9 | 🕊️ DIPLOMACY | International engagement | $2.0B | May reduce sanctions |

---

## 🧪 Advanced Features

### Information Warfare Engine
- **External Agitation**: Foreign social media influence
- **Bot Armies**: State vs Opposition automated accounts
- **Viral Sparks**: Random influential posts with massive reach
- **Information Saturation**: Confusion from conflicting narratives
- **Echo Chambers**: Polarization dynamics

### Behavioral Economics
- **Perceived vs Real Inflation**: Rumor-driven price expectations
- **Panic Buying**: Hoarding cascades when panic exceeds threshold
- **Speculation Amplifiers**: Market psychology feedback loops
- **Black Market Growth**: Informal economy expansion under stress

### Security Forces Module
- **Morale Dynamics**: Drain from operations, recovery during calm
- **Institutional Learning**: Tactics become more efficient with use
- **Defection Risk**: Critical threshold when morale collapses
- **Violence Index**: Accumulating trauma from repression
- **Social Scars**: Long-term societal damage (decay rate: 0.999)

### Threshold Model of Collective Action
- **Individual Thresholds**: Heterogeneous activation points (5%-80%)
- **Social Pressure**: Weighted sum of active neighbors
- **Collective Courage**: Avalanche effect at high mobilization
- **Protest Fatigue**: Stamina drain (5%/step) and recovery (2%/step)
- **Complex Contagion**: Requires multiple exposures to activate

### Radicalization Dynamics
- **Violence Backfire**: Excessive repression (>65%) mobilizes more people
- **Network Restructuring**: Movements become cellular (harder to infiltrate)
- **Martyrdom Effect**: Extreme violence restores stamina via shock
- **Preference Falsification**: Hidden opposition in high-fear environments

---

## 📊 Model Architecture

### Agent: Hierarchical Transformer-based SAC

```
State (20 dimensions) → Transformer Encoder → Manager Head → Goal Vector
                                                ↓
                                           Worker Network
                                                ↓
                                    [Action Mean, Log Std]
                                                ↓
                                      Tanh Squashing → Action
```

**Components:**
- **Transformer Encoder**: Temporal attention over 10-step history
- **Hierarchical Policy**: Manager sets goals, Worker executes
- **Distributional Critic**: 32 quantiles for risk-sensitive Q-learning
- **Noisy Layers**: Parameter-space exploration
- **Prioritized Experience Replay**: 200K buffer with importance sampling
- **Automatic Entropy Tuning**: Dynamic exploration-exploitation balance

### Population Model

- **2000 Agents** (configurable)
- **Small-World Network**: Watts-Strogatz (k=12, p=0.12)
- **Preferential Attachment**: Influencers have 20+ connections
- **Demographics**: Age, ethnicity, city, urban/rural
- **Psychological Profiles**: 
  - 15% Loyalists (high threshold)
  - 65% Silent Majority (moderate threshold)
  - 20% Activists (low threshold)

### Macro Systems

#### Economic State
- Oil revenue dynamics
- Forex market with reserves
- Multi-sector inflation (food, housing)
- Shadow economy (35% baseline)
- Behavioral expectations with anchor decay

#### Sociological Structure
- Ethnic friction matrix (4 groups)
- Strong vs weak ties
- Preference falsification index
- Regional development gaps

#### Regime Parameters
- IRGC loyalty: 85%
- Artesh loyalty: 60%
- Basij density: 5%
- Elite fragmentation: 20%

---

## 🎯 Reward Function

```python
R = 0.30 * R_mobilization     # Exponential penalty
  + 0.15 * R_economy          # Cost + inflation + reserves
  + 0.10 * R_legitimacy       # Political capital
  + 0.05 * R_morale           # Security force health
  + 0.15 * R_scars            # Long-term trauma (HEAVY)
  + 0.10 * R_violence         # Repression costs
  + 0.10 * R_radicalization   # Catastrophic failure mode
  + 0.05 * R_action_specific  # Reform bonus, Net Cut penalty
```

**Key Insight**: Model heavily penalizes social scars and radicalization, incentivizing de-escalation strategies.

---

## 🔬 Scientific Foundation

### Theoretical Basis

1. **Threshold Models** (Granovetter 1978)
2. **Complex Contagion** (Centola 2018)
3. **Prospect Theory** (Kahneman & Tversky)
4. **Preference Falsification** (Kuran 1997)
5. **Collective Action Cascades** (Lohmann 1994)
6. **Social Scars Hypothesis** (Clark et al. 2001)

### Validation Events

- **2009 Green Movement**: RMSE < 0.10, r > 0.80
- **2022 Mahsa Amini Protests**: RMSE < 0.12, r > 0.75

---

## ⚙️ Configuration

### Key Hyperparameters

```python
# RL Agent
lr_actor = 3e-4
lr_critic = 3e-4
gamma = 0.99
memory_size = 200000
batch_size = 128

# Population
n_actors = 2000
simulation_days = 365

# Economics
base_cpi = 0.45
speculation_amplifier = 2.5
panic_buying_threshold = 0.65

# Security
initial_security_morale = 0.80
morale_drain_per_operation = 0.03
defection_threshold = 0.30

# Information Warfare
external_agitation_baseline = 0.3
viral_spark_probability = 0.02
```

---

## 📁 Project Structure

```
Simorgh.py
├── Kernels (Numba-accelerated)
│   ├── kernel_advanced_grievance_v2
│   ├── kernel_threshold_diffusion
│   ├── kernel_entropy_panic_v2
│   └── kernel_speculative_shock
├── Core Classes
│   ├── EnhancedPopulation
│   ├── MacroEconomicState
│   ├── InformationWarfareEngine
│   ├── SecurityForcesModule
│   └── SimorghTwinV20
├── Deep Learning
│   ├── TransformerEncoder
│   ├── HierarchicalActor
│   ├── DistributionalCritic
│   └── SOTA_Agent
├── Interfaces
│   ├── run_warroom_dashboard() [Streamlit]
│   ├── run_production_api() [FastAPI]
│   └── HistoricalValidator
└── Main Entry Point
```

---

## 🐛 Troubleshooting

### NaN Errors
Model includes extensive NaN protection:
- State normalization with clipping
- Gradient clipping (max_norm=1.0)
- Safe reward capping
- Fallback to random actions on NaN detection

### GPU Memory
If CUDA out-of-memory:
```bash
# Reduce population size
python Simorgh.py --mode train --n-actors 1000

# Or disable GPU
export CUDA_VISIBLE_DEVICES=""
```

### Dashboard Not Loading
```bash
# Install missing dependencies
pip install streamlit plotly

# Check Streamlit version
streamlit --version

# Force reinstall
pip install --upgrade --force-reinstall streamlit
```

---

## 📜 License & Citation

### License
This project is released for **research and educational purposes only**.

### Citation
```bibtex
@software{simorgh_v20,
  title={SIMORGH V20: National Security Digital Twin with Information Warfare},
  author={[Your Name/Organization]},
  year={2025},
  version={20.0.0},
  note={Advanced agent-based model with deep reinforcement learning}
}
```

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional strategic actions (cyber operations, economic warfare)
- More sophisticated network dynamics (homophily, bridge nodes)
- International dimension (sanctions modeling, diplomatic games)
- Calibration to additional historical cases
- Optimization for larger populations (>10K agents)

---

## 📧 Contact & Support

For questions, issues, or collaboration:
- 🐛 **Issues**: Use GitHub Issues
- 💬 **Discussions**: Use GitHub Discussions

---

## 🙏 Acknowledgments

Built on research by:
- Mark Granovetter (Threshold Models)
- Daron Acemoglu (Political Economy)
- Timur Kuran (Preference Falsification)
- Dani Rodrik (Economic Development)
- Duncan Watts (Network Science)

Powered by:
- PyTorch, NumPy, NetworkX
- Streamlit, FastAPI
- Numba (GPU acceleration)

---

<div align="center">

**🦅 SIMORGH V20 - Where Complexity Science Meets National Security**

*"The future belongs to those who can model it."*

---

Made with 🧠 for strategic foresight

</div>
