.. _configuration_file:

Configuration File
==================

The config file defines how experiments will be run

Data parameters
---------------

datapull_date
^^^^^^^^^^^^^

mgdiagnose expects the data to be versioned and stored in csv files with the following file name convention: `yyyymmddhhMMss.csv`

* `yyyy`: 4-digit year
* `mm`: 2-digit month number
* `dd`: 2-digit day
* `hh`: 2-digit hour in 24h format
* `MM`: 2-digit minute
* `ss`: 2-digit second



This datetime stamp is defined here as ``datapull_date`` and is a mandatory parameter in the configuration.

Feature parameters
------------------

muscles
^^^^^^^

List of muscle names to be used.

.. TODO: Add a section explaining the expected data format better
.. warning:: mgdiagnose expects to find 3 columns for each muscle: one for mean scores, one for left-side scores and one for right-side scores. The left and right-side columns must me named after the muscle followed by ``_l`` and ``_r`` respectively. 

non_muscle_columns
^^^^^^^^^^^^^^^^^^

Other columns to be loaded and used as features or for processing.

.. warning:: mgdiagnose expects some columns describing the sample scores, as ``side``, ``scale``.

label_col
^^^^^^^^^

Column containing the class labels

non_train_cols
^^^^^^^^^^^^^^

Column that shall not be used as input features

Processing parameters
---------------------

data_operations
^^^^^^^^^^^^^^^

Reshaping operations for merging and expanding columns. Each operation requires the following parameters:

type
""""
Type of operation, either ``merge`` (See :func:`mgdiagnose.process.process.merge`), ``expand`` (See :func:`mgdiagnose.process.process.expand`) or ``combine_labels`` (See :func:`mgdiagnose.process.process.combine_labels`).

input
"""""

Column input name (if ``expand``) or names (if ``merge``) to be used

output
""""""

Column output name (if ``merge``) or names (if ``expand``) to be used

asymmetry
^^^^^^^^^

Bool indicating if asymmetry should be calculated (See :func:`mgdiagnose.process.process.asymmetry`).


bilateral_to_mean
^^^^^^^^^^^^^^^^^

Bool indicating if bilateral muscle scores should be averaged (See :func:`mgdiagnose.process.process.bilateral_to_mean`)

remove_unscored
^^^^^^^^^^^^^^^

If a ``bool`` is provided, it indicates if samples without any muscle fat score should be removed (See :func:`mgdiagnose.process.process.remove_unscored`).

If a ``float`` is provided, it indicates the threshold (range 0. to 1.) of non-missing values required to keep a sample. I.e if ``remove_unscored=0.2`` then all samples with over an 80% of missing data will be removed.

scale_scores
^^^^^^^^^^^^

Wether to rescale the fat scores or not. If ``Null``, no rescaling is applied. If a list is passed, the first and second elements are assumed to be the lower an upper values of the new range. (See :func:`mgdiagnose.process.process.scale_scores`)

.. TODO: Add a section explaining the expected data format better
.. warning:: to scale the scores, mgdiagnose expects to find a column named ``scale`` labelling the scale used for each sample.