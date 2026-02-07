# LFM Galaxy Rotation Curves - Complete Honest Assessment

## Summary of Experiments

### GOV-04 (Quasi-static Poisson limit)
**File**: lfm_gov04_proper.py
**Result**: DECLINING curves (Keplerian falloff)
**Why**: GOV-04 is structurally identical to Newtonian Poisson equation

### GOV-03 (Chi memory with tau-averaging)
**File**: lfm_gov03_memory.py, lfm_gov03_scan.py
**Result**: Can produce FLAT curves with parameter tuning
**Problem**: 
- Requires 2 free parameters (g, tau_scale)
- Shape is wrong (rising, not flat throughout)
- Not predictive without deriving tau from fundamentals

### RAR with LFM a
**File**: lfm_sparc_rotation_rar.py
**Result**: Excellent fits (4/5 flat, 11.5% avg error)
**Problem**: RAR functional form is EMPIRICAL, not derived

## What LFM Actually Provides

| Quantity | Status | Source |
|----------|--------|--------|
| a = cH/(2π) |  DERIVED | LFM cosmology |
| a value match |  10% | Compared to observed g |
| RAR shape |  BORROWED | McGaugh et al. 2016 |
| Flat curves from GOV-03 |  POSSIBLE | With tuning |
| Predictive power |  MISSING | Need tau derivation |

## The Gap That Needs Closing

**Required derivation**: Show that coupled GOV-01 + GOV-02 dynamics produce the RAR interpolating function:

lfm_rar_derivation_attempt.py\nu(x) = \frac{1}{1 - e^{-\sqrt{x}}}lfm_rar_derivation_attempt.py

where  = g_{bar}/a_0$ and  = cH_0/(2\pi)$.

**Current status**: We can MATCH the scale (a) but not DERIVE the shape (ν(x)).

## Conclusion

LFM rotation curve predictions are currently a **partial success**:
-  Correct acceleration scale from cosmology
-  RAR shape not derived from first principles
-  GOV-03 can produce flat curves but requires tuning

This is an honest gap that should be addressed in future work.
