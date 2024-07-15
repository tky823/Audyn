audyn.criterion.sdr
===================

``audyn.criterion.sdr`` includes criteria related to source-to-distortion ratio (SDR).

Classes
-------

SISDR
^^^^^

Scale-invariant SDR (SI-SDR).

.. autoclass:: audyn.criterion.sdr.SISDR
   :members: forward

NegSISDR
^^^^^^^^

Negative SI-SDR used for minimization in training of source separation models.

.. autoclass:: audyn.criterion.sdr.NegSISDR
   :members: forward

PITNegSISDR
^^^^^^^^^^^

Negative SI-SDR with permutation invariant training (PIT).

.. autoclass:: audyn.criterion.sdr.PITNegSISDR
   :members: forward
