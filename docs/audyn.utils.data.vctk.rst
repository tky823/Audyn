audyn.utils.data.vctk
=====================

``audyn.utils.data.vctk`` provides utilities for VCTK dataset.

.. code-block:: python

    >>> from audyn.utils.data.vctk import speakers, num_speakers
    >>> print(speakers[80:90])
    ['p312', 'p313', 'p314', 'p315', 'p316', 'p317', 'p318', 'p323', 'p326', 'p329']
    >>> print(num_speakers)
    110
    >>> from audyn.utils.data.vctk import valid_speakers, num_valid_speakers
    >>> print(valid_speakers[80:90])
    ['p313', 'p314', 'p316', 'p317', 'p318', 'p323', 'p326', 'p329', 'p330', 'p333']
    >>> print(num_valid_speakers)
    108
    >>> print(set(speakers) - set(valid_speakers))
    {'p280', 'p315'}
