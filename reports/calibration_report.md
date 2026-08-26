# Model Probability Calibration & Reliability Report

**Project**: Intain Campus FinTech Challenge 2026 — AI Track
**Module**: Probability Calibration & Reliability Diagnostics
**Validation Cohort**: Out-of-time (2020-01 to 2021-12) | N = 95,563 rows

---

## Executive Summary

Probability calibration ensures that a predicted score of $p$ matches the empirical true outcome frequency $p$ in practice.
Post-hoc Platt Sigmoid Scaling (`CalibratedClassifierCV`) was applied to all LightGBM models, optimizing Brier scores and minimizing Expected Calibration Error (ECE).

## Target: `next_3m_delinquency_flag`

- **Brier Score Loss**: `0.0297` (lower is better)
- **Expected Calibration Error (ECE)**: `0.0018`
- **Maximum Calibration Error (MCE)**: `0.0504`

### Empirical Binned Reliability Table

| Probability Bin | Loan Count | Mean Predicted Prob | Empirical True Rate | Calibration Gap |
| :--- | :--- | :--- | :--- | :--- |
| [0.00, 0.10) | 93,864 | 0.0285 | 0.0276 | 0.0009 |
| [0.60, 0.70) | 1,699 | 0.6989 | 0.7493 | 0.0504 |

### Reliability Diagram

```
  Empirical Rate vs Predicted Probability (Ideal = Diagonal /)
  1.0 |
 1.0 |                   ·
     |                 ·  
     |             █ ·    
 0.7 |             ·      
     |           ·        
     |         ·          
 0.4 |       ·            
     |     ·              
     |   ·                
 0.1 | █                  
  0.0 +--------------------
       0.0  0.2  0.4  0.6  0.8  1.0 (Predicted Prob)
  Legend: [·] = Perfectly Calibrated Diagonal | [█] = Empirical Model Bin
```

---

## Target: `next_6m_delinquency_flag`

- **Brier Score Loss**: `0.0500` (lower is better)
- **Expected Calibration Error (ECE)**: `0.0046`
- **Maximum Calibration Error (MCE)**: `0.0972`

### Empirical Binned Reliability Table

| Probability Bin | Loan Count | Mean Predicted Prob | Empirical True Rate | Calibration Gap |
| :--- | :--- | :--- | :--- | :--- |
| [0.00, 0.10) | 86,771 | 0.0454 | 0.0460 | 0.0006 |
| [0.10, 0.20) | 7,093 | 0.1317 | 0.1011 | 0.0307 |
| [0.60, 0.70) | 1,699 | 0.6621 | 0.7593 | 0.0972 |

### Reliability Diagram

```
  Empirical Rate vs Predicted Probability (Ideal = Diagonal /)
  1.0 |
 1.0 |                   ·
     |                 ·  
     |             █ ·    
 0.7 |             ·      
     |           ·        
     |         ·          
 0.4 |       ·            
     |     ·              
     |   █                
 0.1 | █                  
  0.0 +--------------------
       0.0  0.2  0.4  0.6  0.8  1.0 (Predicted Prob)
  Legend: [·] = Perfectly Calibrated Diagonal | [█] = Empirical Model Bin
```

---

## Target: `next_12m_default_flag`

- **Brier Score Loss**: `0.0418` (lower is better)
- **Expected Calibration Error (ECE)**: `0.0016`
- **Maximum Calibration Error (MCE)**: `0.1858`

### Empirical Binned Reliability Table

| Probability Bin | Loan Count | Mean Predicted Prob | Empirical True Rate | Calibration Gap |
| :--- | :--- | :--- | :--- | :--- |
| [0.00, 0.10) | 94,525 | 0.0424 | 0.0416 | 0.0008 |
| [0.10, 0.20) | 28 | 0.1713 | 0.3571 | 0.1858 |
| [0.20, 0.30) | 1,010 | 0.2927 | 0.3644 | 0.0717 |

### Reliability Diagram

```
  Empirical Rate vs Predicted Probability (Ideal = Diagonal /)
  1.0 |
 1.0 |                   ·
     |                 ·  
     |               ·    
 0.7 |             ·      
     |           ·        
     |         ·          
 0.4 |   █ █ ·            
     |     ·              
     |   ·                
 0.1 | █                  
  0.0 +--------------------
       0.0  0.2  0.4  0.6  0.8  1.0 (Predicted Prob)
  Legend: [·] = Perfectly Calibrated Diagonal | [█] = Empirical Model Bin
```

---

## Target: `next_12m_prepayment_flag`

- **Brier Score Loss**: `0.0446` (lower is better)
- **Expected Calibration Error (ECE)**: `0.0000`
- **Maximum Calibration Error (MCE)**: `0.0055`

### Empirical Binned Reliability Table

| Probability Bin | Loan Count | Mean Predicted Prob | Empirical True Rate | Calibration Gap |
| :--- | :--- | :--- | :--- | :--- |
| [0.00, 0.10) | 95,553 | 0.0470 | 0.0470 | 0.0000 |
| [0.20, 0.30) | 10 | 0.2055 | 0.2000 | 0.0055 |

### Reliability Diagram

```
  Empirical Rate vs Predicted Probability (Ideal = Diagonal /)
  1.0 |
 1.0 |                   ·
     |                 ·  
     |               ·    
 0.7 |             ·      
     |           ·        
     |         ·          
 0.4 |       ·            
     |     █              
     |   ·                
 0.1 | █                  
  0.0 +--------------------
       0.0  0.2  0.4  0.6  0.8  1.0 (Predicted Prob)
  Legend: [·] = Perfectly Calibrated Diagonal | [█] = Empirical Model Bin
```

---
