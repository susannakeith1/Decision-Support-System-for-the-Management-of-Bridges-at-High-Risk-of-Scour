[README.md](https://github.com/user-attachments/files/25846713/README.md)
# Decision Support System for Bridges at High Risk of Scour

This repository contains the code used in the research paper: 

"**Decision Support System for the Management of Bridges at High Risk of Scour**"

The project develops a **decision support framework** to help bridge managers determine whether it is economically beneficial to conduct additional site investigations (such as coring) to determine the true foundaiton depth of a masonry arch bridge.

The framework combines **structural reliability analysis**, **decision analysis**, and **Value of Information (VoI)** to evaluate whether collecting additional informaiton improves decision making. 

## Methodology
The analysis consists of three main components:

### 1. **Reliability Analysis** 

A reliability model is used to estimate the **probability of bridge failure due to scour**. 

This probability is evaluated under uncertainty in the **foundation depth**, which is represented as a probability distribution.

### 2. **Decision problem** 

A decison model evaluates the **expected cost of different management decisions**, such as intervention or no intervention.

Two information scenarios are considered:
- **Prior case** - foundaiton depth is uncertain
- **Posterior case** - foundaiton depth becomes known after testing

### 3. **Value of Information** 

The Value of Information quantifies the **economic benefit of collecting addtional site-specific data**.

$$ VoI = EC_{prior} - EC_{preposterior} $$

where:
- $EC_{prior}$ = expected cost when foundaiton depth is unknown.
- $EC_{preposterior}$ = expected cost when foundaiton depth is known

A positive $VoI$ indicates that collecting additional information is economically beneficial.


## Repository Structure

```
.
├── Functions.py                        # Core reliability, decision, and VoI calculations
├── ProbabilityFailure.ipynb            # Example notebook demonstrating the model
├── WaterLevel_30Years_Q_May26.csv      # Example water level dataset
└── README.md                           # Project documentation

```
## Requirements
```
numpy
scipy
pandas
matplotlib
jupyter
```

## Running the Code:
The analysis is performed using the function `run_DT_failure_model` contained in `Functions.py`.

Example usage:


```python
from Functions import *
import pandas as pd

DF_known=1

r, Pf_prior, E_Pf_post, P_fail_post, DF_list = run_DT_failure_model(
    waterlevel_file="WaterLevel_30Years_Q_May26.csv",
    y0=1.0,
    y200 = 2.5,
    Wp=2,
    f_PS=1,
    f_PA=1,
    f_d=1,
    bias=1.0,
    sigma_DT_error=0.3,
    mu= 4,
    sigma=1.38,
    DF_known=DF_known,
    DFmedian=1.349,
    sigma_ln_DF=0.703
)
```
This function returns:
- `r` - dictionary containing all the decision analysis and Value of Information results. 
- `Pf_prior` - probability of failure in the prior case
- `E_Pf_post` - expected posterior probability of failure
- `P_fail_post` - posterior probabilities of failure for each possible foundation depth
- `DF_list` - sampled foundation depth values used in the analysis

## Example Output

The example notebook `ProbabilityFailure.ipynb` demonstrates the full workflow of the framework. It:

   1. Runs the reliability and decision model
   2. Prints the calculated probabilities of failure and expected utilities
   3. Computes the Value of Information
   4. Generates a fugure illustrating the results used in the paper. 
   
The printed output includes:
- Prior probability of failure
- Posterir probabilities of faiure
- Expected utilties (costs)
- Value of Information.

Example output structure:
```
Prior
------------------------------
Pf_prior               : 0.01768
EC_M                   : 1.63174
EC_DN_prior            : 1.59667
EC_RR_prior            : 0.04000
EC_prior               : 0.04000

Posterior
------------------------------
Expected_Pf_Post       : 0.01775
EC_pp                  : 0.01364
Number of optimal DN   : 14034
Number of optimal M    : 0
Number of optimal RR   : 5966

VoI
------------------------------
VoI                    : 0.0264
VoI £                  : 52800
```


The Value of Information represents the **maximum amount that should be spent on coring to determine the true foundation depth**. 
