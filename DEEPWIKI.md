# Meridian MMM Framework - Deep Architecture Documentation

## Table of Contents

1. [Framework Overview](#framework-overview)
2. [Core Architecture](#core-architecture)
3. [Model Module - Bayesian Inference Engine](#model-module---bayesian-inference-engine)
4. [Data Module - Input Processing Pipeline](#data-module---input-processing-pipeline)
5. [Analysis Module - Insights and Optimization](#analysis-module---insights-and-optimization)
6. [Mathematical Foundations](#mathematical-foundations)
7. [Configuration and Constants](#configuration-and-constants)
8. [Technical Implementation Details](#technical-implementation-details)
9. [Usage Patterns and Workflows](#usage-patterns-and-workflows)
10. [Advanced Features](#advanced-features)

## Framework Overview

Meridian is a comprehensive Marketing Mix Modeling (MMM) framework developed by Google that enables advertisers to set up and run their own in-house Bayesian causal inference models. The framework is designed to handle large-scale geo-level data while also supporting national-level modeling, providing clear insights and visualizations to inform business decisions around marketing budget and planning.

### Key Capabilities

- **Bayesian Hierarchical Modeling**: Uses advanced Bayesian causal inference with MCMC sampling
- **Geo-Level Analysis**: Supports both geo-level and national-level modeling approaches
- **Media Attribution**: Measures impact across marketing channels accounting for non-marketing factors
- **Budget Optimization**: Provides methodologies for optimizing marketing budget allocation
- **Reach & Frequency Support**: Handles both traditional media data and reach/frequency data
- **Experimental Calibration**: Supports calibration with experiments and prior information
- **GPU Acceleration**: Built-in GPU support using TensorFlow for real-time optimization

## Core Architecture

The Meridian framework is organized into four main modules, each serving a specific purpose in the MMM pipeline:

```
meridian/
├── model/          # Bayesian inference engine and core modeling
├── data/           # Input data processing and validation
├── analysis/       # Post-modeling analysis and optimization
└── mlflow/         # Experiment tracking integration
```

### Main Entry Points

The framework exposes its functionality through the main `meridian` package:

```python
from meridian import analysis, data, model
from meridian.version import __version__
```

The primary workflow involves:
1. **Data Preparation**: Using `data.InputData` to structure and validate input data
2. **Model Specification**: Configuring model parameters and priors
3. **Model Fitting**: Using `model.Meridian` to fit the Bayesian model via MCMC
4. **Analysis**: Using `analysis.Analyzer` for insights and `analysis.BudgetOptimizer` for optimization

## Model Module - Bayesian Inference Engine

### Core Meridian Class

The `Meridian` class in `meridian/model/model.py` serves as the main interface for the MMM framework. It encapsulates the entire Bayesian hierarchical model and provides methods for fitting, prediction, and analysis.

**Key Attributes:**
- `input_data`: InputData object containing structured input data
- `model_spec`: ModelSpec object defining model configuration
- `inference_data`: ArviZ InferenceData object with MCMC results
- `n_geos`, `n_media_channels`, `n_rf_channels`: Data dimensions
- `is_national`: Boolean indicating national vs geo-level modeling

**Core Methods:**
- `fit()`: Fits the model using MCMC sampling
- `predict()`: Generates predictions for new scenarios
- `save()` / `load()`: Model persistence functionality

### Bayesian Inference Implementation

The framework implements a sophisticated Bayesian hierarchical model using the `PosteriorMCMCSampler` class:

#### MCMC Sampling with NUTS

The posterior sampling uses the No U-Turn Sampler (NUTS) algorithm via TensorFlow Probability:

```python
class PosteriorMCMCSampler:
    def __init__(self, meridian: "model.Meridian"):
        self._meridian = meridian
    
    def __call__(self, n_chains: int, n_warmup: int, n_samples: int, ...):
        # Implements windowed adaptive NUTS sampling
        # Returns ArviZ InferenceData with posterior samples
```

**Key Features:**
- XLA compilation for performance (`@tf.function(jit_compile=True)`)
- Adaptive step size and mass matrix tuning
- Multiple chain sampling for convergence diagnostics
- Memory-optimized implementation for large models

#### Joint Distribution Structure

The model defines a joint probability distribution over all parameters using TensorFlow Probability's `JointDistributionCoroutineAutoBatched`. The distribution includes:

- **Media Parameters**: ROI, adstock (α), hill saturation (EC, slope)
- **Geo Parameters**: Geographic random effects (τ_g)
- **Time Parameters**: Temporal trends via spline knots
- **Control Parameters**: Coefficients for control variables
- **Noise Parameters**: Observation noise (σ)

### Media Effects Modeling

#### Adstock Transformation

The `AdstockTransformer` class implements media carryover effects:

```python
class AdstockTransformer(AdstockHillTransformer):
    def __init__(self, alpha: tf.Tensor, max_lag: int, n_times_output: int):
        # alpha: decay parameter [0, 1)
        # max_lag: maximum carryover periods
        
    def forward(self, media: tf.Tensor) -> tf.Tensor:
        # Computes: adstock_{g,t,m} = Σ_{i=0}^{max_lag} media_{g,t-i,m} * α^i
```

**Mathematical Implementation:**
- Geometric decay weighting with normalization
- Memory-optimized windowed calculation
- Supports both speed and memory optimization modes

#### Hill Saturation Transformation

The `HillTransformer` class models diminishing returns:

```python
class HillTransformer(AdstockHillTransformer):
    def __init__(self, ec: tf.Tensor, slope: tf.Tensor):
        # ec: half-saturation point
        # slope: saturation curve steepness
        
    def forward(self, media: tf.Tensor) -> tf.Tensor:
        # Computes: media^slope / (media^slope + ec^slope)
```

### Model Specifications and Priors

The framework supports flexible prior specification through the `PriorDistribution` class:

**Prior Types for Media Channels:**
- `roi`: Return on Investment priors
- `mroi`: Marginal ROI priors (1% spend increase)
- `coefficient`: Direct coefficient priors
- `contribution`: Contribution percentage priors

**Distribution Options:**
- Normal distributions for standard effects
- Log-normal distributions for strictly positive effects
- Custom distributions for specific business constraints

## Data Module - Input Processing Pipeline

### InputData Class Architecture

The `InputData` class in `meridian/data/input_data.py` serves as the central data container and validation engine:

```python
@dataclasses.dataclass
class InputData:
    # Required arrays
    kpi: xr.DataArray                    # Key performance indicator
    population: xr.DataArray             # Population data
    
    # Media arrays
    media: xr.DataArray | None           # Traditional media impressions
    media_spend: xr.DataArray | None     # Media spend data
    
    # Reach & Frequency arrays
    reach: xr.DataArray | None           # Reach data
    frequency: xr.DataArray | None       # Frequency data
    rf_spend: xr.DataArray | None        # RF spend data
    
    # Additional arrays
    controls: xr.DataArray | None        # Control variables
    organic_media: xr.DataArray | None   # Organic media impressions
    non_media_treatments: xr.DataArray | None  # Non-media treatments
    revenue_per_kpi: xr.DataArray | None # Revenue per KPI unit
```

### Data Validation Framework

The InputData class implements comprehensive validation:

#### Dimension Validation
- Ensures consistent geo, time, and channel dimensions across arrays
- Validates coordinate alignment between related arrays
- Checks for required vs optional data arrays

#### Data Type Validation
- Supports both traditional media and reach/frequency data
- Validates spend data can be provided at channel or geo×time×channel level
- Ensures proper data types and value ranges

#### Coordinate System
The framework uses a standardized coordinate system:
- `geo`: Geographic regions (can be national for aggregated models)
- `time`: Time periods in YYYY-MM-DD format
- `media_time`: Media execution time periods
- `*_channel`: Channel dimensions for different media types

### Data Transformation Pipeline

#### Scaling and Normalization
The framework applies several transformations:

1. **KPI Scaling**: Uses `transformers.KpiTransformer` for outcome scaling
2. **Control Scaling**: Standardizes control variables
3. **Media Scaling**: Normalizes media variables for numerical stability

#### Time Coordinate Handling
The `TimeCoordinates` class manages temporal aspects:
- Converts between different time formats
- Handles media execution vs outcome measurement timing
- Supports flexible time period aggregation

## Analysis Module - Insights and Optimization

### Analyzer Class - Core Analytics Engine

The `Analyzer` class in `meridian/analysis/analyzer.py` provides comprehensive post-modeling analysis:

```python
class Analyzer:
    def __init__(self, meridian: model.Meridian):
        self._meridian = meridian
        # Provides access to fitted model and inference data
```

#### Key Analysis Methods

**Incremental Outcome Analysis:**
```python
def incremental_outcome(
    self,
    data_tensors: DataTensors,
    use_posterior: bool = True,
    selected_geos: Sequence[str] | None = None,
    selected_times: Sequence[str] | None = None,
    aggregate_geos: bool = False,
    confidence_level: float = 0.9,
) -> xr.Dataset:
    # Computes incremental KPI/revenue attributed to each channel
```

**ROI Calculation:**
```python
def roi(self, ...) -> xr.Dataset:
    # Computes return on investment for each channel
    
def marginal_roi(self, ...) -> xr.Dataset:
    # Computes marginal ROI (1% spend increase impact)
```

**Response Curves:**
```python
def response_curves(
    self,
    spend_multipliers: Sequence[float] | None = None,
    use_posterior: bool = True,
    by_reach: bool = False,
) -> pd.DataFrame:
    # Generates media response curves for optimization
```

#### DataTensors Container

The `DataTensors` class provides a structured way to pass data for analysis:

```python
class DataTensors(tf.experimental.ExtensionType):
    media: Optional[tf.Tensor]
    media_spend: Optional[tf.Tensor]
    reach: Optional[tf.Tensor]
    frequency: Optional[tf.Tensor]
    rf_spend: Optional[tf.Tensor]
    organic_media: Optional[tf.Tensor]
    # ... additional tensor fields
```

### BudgetOptimizer - Media Planning Engine

The `BudgetOptimizer` class implements sophisticated budget allocation optimization:

```python
class BudgetOptimizer:
    def __init__(self, meridian: model.Meridian):
        self._meridian = meridian
        
    def optimize(
        self,
        n_time_periods: int,
        spend_constraints: Mapping[str, _SpendConstraint] | None = None,
        fixed_budget: float | None = None,
        target_roi: float | None = None,
        target_mroi: float | None = None,
        # ... additional parameters
    ) -> OptimizationResults:
```

#### Optimization Scenarios

**Fixed Budget Optimization:**
```python
@dataclasses.dataclass(frozen=True)
class FixedBudgetScenario:
    total_budget: float | None = None
    # Optimizes allocation within fixed total budget
```

**Flexible Budget Optimization:**
```python
@dataclasses.dataclass(frozen=True)
class FlexibleBudgetScenario:
    target_metric: str  # 'roi' or 'mroi'
    target_value: float
    # Finds optimal budget to achieve target ROI/mROI
```

#### Optimization Grid System

The `OptimizationGrid` class implements grid search optimization:

1. **Grid Generation**: Creates spend allocation grids based on constraints
2. **Response Calculation**: Computes incremental outcomes for each grid point
3. **Constraint Checking**: Validates spend bounds and target metrics
4. **Optimization**: Finds optimal allocation via grid search

### Visualization and Reporting

The analysis module includes comprehensive visualization capabilities:

- **Response Curves**: Media saturation and efficiency curves
- **Budget Allocation**: Optimal vs historical spend comparison
- **Incremental Outcomes**: Channel contribution analysis
- **Model Diagnostics**: Convergence and fit quality metrics

## Mathematical Foundations

### Bayesian Hierarchical Model Structure

The Meridian model implements a sophisticated Bayesian hierarchical structure:

#### Level 1: Observation Model
```
KPI_{g,t} ~ Normal(μ_{g,t}, σ_g)
```

Where μ_{g,t} is the linear predictor combining:
- Media effects (with adstock and saturation)
- Control variable effects
- Geographic random effects
- Temporal trends

#### Level 2: Media Effects Model
```
Media Effect_{g,t,m} = β_{g,m} × Hill(Adstock(Media_{g,t,m}))
```

With transformations:
- **Adstock**: `Σ_{i=0}^{max_lag} Media_{g,t-i,m} × α_m^i`
- **Hill**: `Media^{slope_m} / (Media^{slope_m} + EC_m^{slope_m})`

#### Level 3: Hierarchical Priors
```
β_{g,m} ~ Normal(β_m, τ_g)
β_m ~ Prior distribution (ROI, mROI, coefficient, or contribution)
τ_g ~ Prior distribution for geo variation
```

### MCMC Sampling Mathematics

The framework uses Hamiltonian Monte Carlo (HMC) with the No U-Turn Sampler (NUTS):

1. **Hamiltonian Dynamics**: Simulates particle movement in parameter space
2. **Automatic Differentiation**: TensorFlow computes gradients for HMC
3. **Adaptive Tuning**: Automatically tunes step size and mass matrix
4. **No U-Turn Criterion**: Prevents excessive trajectory simulation

### Prior Distribution Framework

The framework supports multiple prior specification approaches:

#### ROI Priors
```python
# Direct ROI specification
roi_prior = tfp.distributions.Normal(loc=roi_mean, scale=roi_std)
```

#### Marginal ROI Priors
```python
# 1% spend increase impact
mroi_factor = 1.01
mroi_prior = tfp.distributions.Normal(loc=mroi_mean, scale=mroi_std)
```

#### Contribution Priors
```python
# Percentage of total KPI attributed to channel
contribution_prior = tfp.distributions.Beta(alpha=a, beta=b)
```

## Configuration and Constants

### Constants Framework

The `meridian/constants.py` file defines the comprehensive configuration system:

#### Data Array Names
```python
# Required input data
REQUIRED_INPUT_DATA_ARRAY_NAMES = (KPI, POPULATION)

# Media data
MEDIA_INPUT_DATA_ARRAY_NAMES = (MEDIA, MEDIA_SPEND)

# Reach & Frequency data
RF_INPUT_DATA_ARRAY_NAMES = (REACH, FREQUENCY, RF_SPEND)

# Optional data
OPTIONAL_INPUT_DATA_ARRAY_NAMES = (
    CONTROLS, REVENUE_PER_KPI, ORGANIC_MEDIA, 
    ORGANIC_REACH, ORGANIC_FREQUENCY, NON_MEDIA_TREATMENTS
)
```

#### Model Parameters
```python
# Media parameters
MEDIA_PARAMETERS = (ROI_M, MROI_M, CONTRIBUTION_M, BETA_M, 
                   ETA_M, ALPHA_M, EC_M, SLOPE_M)

# Geographic parameters  
GEO_PARAMETERS = (TAU_G,)

# Time parameters
TIME_PARAMETERS = (MU_T,)
```

#### Prior Types
```python
TREATMENT_PRIOR_TYPE_ROI = 'roi'
TREATMENT_PRIOR_TYPE_MROI = 'mroi'
TREATMENT_PRIOR_TYPE_COEFFICIENT = 'coefficient'
TREATMENT_PRIOR_TYPE_CONTRIBUTION = 'contribution'
```

### Model Specification Options

#### Media Effects Distribution
```python
MEDIA_EFFECTS_NORMAL = 'normal'        # Standard normal effects
MEDIA_EFFECTS_LOG_NORMAL = 'log_normal' # Strictly positive effects
```

#### National Model Defaults
```python
NATIONAL_MODEL_SPEC_ARGS = {
    MEDIA_EFFECTS_DIST: MEDIA_EFFECTS_NORMAL,
    UNIQUE_SIGMA_FOR_EACH_GEO: False,
}
```

## Technical Implementation Details

### TensorFlow Integration

The framework is built on TensorFlow 2.x with TensorFlow Probability:

#### GPU Acceleration
- Automatic GPU detection and utilization
- XLA compilation for performance optimization
- Memory-optimized tensor operations

#### Tensor Operations
```python
# Example: Adstock calculation with broadcasting
weights = tf.expand_dims(alpha, -1) ** l_range
normalization_factors = tf.expand_dims(
    (1 - alpha ** window_size) / (1 - alpha), -1
)
weights = tf.divide(weights, normalization_factors)
result = tf.einsum('...mw,w...gtm->...gtm', weights, windowed)
```

### Memory Management

The framework implements several memory optimization strategies:

1. **Adstock Memory Optimization**: Configurable memory vs speed tradeoff
2. **Batch Processing**: Handles large datasets through batching
3. **Lazy Evaluation**: Defers computation until needed
4. **Tensor Reuse**: Minimizes memory allocation

### Error Handling

Comprehensive error handling includes:

- `MCMCSamplingError`: MCMC sampling failures
- `MCMCOOMError`: Out-of-memory errors during sampling
- `NotFittedModelError`: Accessing unfitted model results
- Data validation errors with detailed messages

## Usage Patterns and Workflows

### Standard MMM Workflow

#### 1. Data Preparation
```python
import meridian

# Create InputData object
input_data = meridian.data.InputData(
    kpi=kpi_array,
    population=population_array,
    media=media_array,
    media_spend=spend_array,
    controls=controls_array,
    # ... additional arrays
)
```

#### 2. Model Specification
```python
# Define model specification
model_spec = meridian.model.ModelSpec(
    max_lag=13,  # 13-week adstock
    hill_before_adstock=True,
    media_effects_dist='normal',
    # ... additional parameters
)
```

#### 3. Model Fitting
```python
# Create and fit model
mmm = meridian.model.Meridian(
    input_data=input_data,
    model_spec=model_spec
)

# Fit using MCMC
mmm.fit(
    n_chains=4,
    n_warmup=1000,
    n_samples=1000,
    seed=42
)
```

#### 4. Analysis and Insights
```python
# Create analyzer
analyzer = meridian.analysis.Analyzer(mmm)

# Compute incremental outcomes
incremental = analyzer.incremental_outcome(
    data_tensors=analyzer.get_data_tensors(),
    use_posterior=True,
    confidence_level=0.9
)

# Calculate ROI
roi_results = analyzer.roi(
    data_tensors=analyzer.get_data_tensors(),
    use_posterior=True
)
```

#### 5. Budget Optimization
```python
# Create optimizer
optimizer = meridian.analysis.BudgetOptimizer(mmm)

# Optimize budget allocation
optimization_results = optimizer.optimize(
    n_time_periods=52,  # 1 year
    fixed_budget=1000000,  # $1M budget
    spend_constraints={
        'TV': [0.2, 0.6],      # 20-60% of budget
        'Digital': [0.3, 0.8], # 30-80% of budget
    }
)
```

### Advanced Usage Patterns

#### Reach & Frequency Modeling
```python
# RF-specific input data
input_data = meridian.data.InputData(
    kpi=kpi_array,
    population=population_array,
    reach=reach_array,
    frequency=frequency_array,
    rf_spend=rf_spend_array,
    revenue_per_kpi=revenue_per_kpi_array
)

# RF-specific analysis
rf_roi = analyzer.roi(
    data_tensors=analyzer.get_data_tensors(),
    use_posterior=True,
    by_reach=True  # RF-specific calculation
)
```

#### National vs Geo-Level Modeling
```python
# National model (aggregated)
national_input = input_data.aggregate_to_national()
national_mmm = meridian.model.Meridian(
    input_data=national_input,
    model_spec=model_spec
)

# Geo-level model (full geographic detail)
geo_mmm = meridian.model.Meridian(
    input_data=input_data,  # Full geo-level data
    model_spec=model_spec
)
```

#### Model Calibration with Experiments
```python
# Incorporate experimental results as priors
experimental_priors = {
    'TV': meridian.model.PriorDistribution(
        distribution='normal',
        loc=2.5,  # Experimental ROI estimate
        scale=0.5  # Uncertainty
    )
}

model_spec = meridian.model.ModelSpec(
    prior=experimental_priors,
    # ... other parameters
)
```

## Advanced Features

### MLflow Integration

The framework includes MLflow integration for experiment tracking:

```python
import meridian.mlflow

# Automatic experiment logging
with meridian.mlflow.start_run():
    mmm.fit(n_chains=4, n_warmup=1000, n_samples=1000)
    # Model metrics and artifacts automatically logged
```

### Model Diagnostics

Comprehensive model diagnostics include:

#### Convergence Diagnostics
```python
# R-hat statistics for convergence
rhat_summary = analyzer.rhat_summary()

# Effective sample size
ess_summary = analyzer.get_effective_sample_size()
```

#### Model Fit Assessment
```python
# Predictive accuracy metrics
accuracy = analyzer.predictive_accuracy(
    holdout_id='test_period'
)

# Expected vs actual comparison
fit_data = analyzer.expected_vs_actual_data()
```

### Custom Transformations

The framework supports custom media transformations:

```python
# Custom adstock-hill transformation
class CustomTransformer(meridian.model.AdstockHillTransformer):
    def forward(self, media: tf.Tensor) -> tf.Tensor:
        # Custom transformation logic
        return transformed_media
```

### Extensibility Points

The framework provides several extension points:

1. **Custom Prior Distributions**: Define domain-specific priors
2. **Custom Media Transformations**: Implement specialized transformations
3. **Custom Optimization Objectives**: Define business-specific optimization goals
4. **Custom Visualizations**: Extend the visualization framework

### Performance Optimization

#### GPU Configuration
```python
# Configure GPU memory growth
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

#### Batch Processing
```python
# Large dataset handling
optimizer = meridian.analysis.BudgetOptimizer(mmm)
results = optimizer.optimize(
    batch_size=1000,  # Process in batches
    # ... other parameters
)
```

This comprehensive documentation provides deep insights into the Meridian MMM framework's architecture, implementation, and usage patterns. The framework represents a sophisticated approach to marketing mix modeling that combines advanced Bayesian statistics, modern machine learning infrastructure, and practical business applications.
