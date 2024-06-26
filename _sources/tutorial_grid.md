::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`note;1.5em;sd-mr-1` Automated Refinement

Using [BGMN](http://www.bgmn.de) as a backend, Dara provides a simple and automated way
to refine phases in powder X-ray diffraction (XRD) data.

+++
[Go to tutorial »](notebooks/automated_refinement)
:::

:::{grid-item-card} {octicon}`search;1.5em;sd-mr-1` Phase searching

Dara provides a parallelilzed tree search algorithm to search for phases in powder X-ray
diffraction data. It needs only two inputs:
(1) the raw X-ray diffraction pattern and (2) the reference phases. For the latter, Dara
also implements `CODDatabase` and `ICSDDatabase` to
help users query the reference CIFs in chemical system of interest in the COD and ICSD databases.

+++
[Go to tutorial »](notebooks/phase_search)
:::

::::
