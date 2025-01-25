audyn.amp
=========

``audyn.amp`` provides utils for mixed precision training using ``torch.amp``.

.. code-block:: python

   >>> import torch
   >>> from audyn.amp import 
   >>> torch.manual_seed(0)
   >>> model = ...
   >>> input = torch.randn((batch_size, in_channels, length))
   >>> device_type = get_autocast_device_type()
   >>> with autocast(device_type, enabled=self.enable_amp, dtype=self.amp_dtype):
           output = self.model(input)
