# A549 three-drug response table

Imported unchanged from `origin/devin/1791043097-zimmer-a549`.
The 512 conditions combine eight concentrations each of taxol (paclitaxel),
cisplatin and doxorubicin. The objective is to minimise survival percent.

Study: Zimmer et al., *Prediction of multidimensional drug dose responses based
on measurements of drug pairs*, PNAS (2016), https://doi.org/10.1073/pnas.1606301113.
The converter documents the secondary source archive and dose mapping.

`survival_sd=3.679` percentage points is estimated from neighbour residuals.
It is **not measured replicate variability**. Fresh simulated preparations sample
this assumed noise and independent delivery error. The demo uses assumed 100 uM
stocks and a 100 uL final volume; this is a simulation recipe, not a reconstruction
of the publication's dispensing, solvent, incubation or readout protocol.
