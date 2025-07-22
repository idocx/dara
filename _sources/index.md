---
sd_hide_title: true
---

<style>
    .keynote {
        font-size: 2em;
        font-weight: 300;
        text-align: center;
        margin: 24px 5%;
    }
</style>

# Overview

```{image} /_static/logo-with-text.svg
:width: 100%
:align: center
```

<p class="keynote">
Dara is a Python package for phase analysis and refinement of powder X-ray diffraction (XRD) data.
</p>

Some features of Dara include:

* **Automated refinement**: Automatically refine a powder XRD pattern with the Rietveld
  method by calling the BGMN program from Python.
* **Phase identification**: Identify phases in a powder XRD pattern using a parallelized
  tree search procedure customized to use any crystallographic database or provided
  CIFs.
* **Multiple hypotheses**: Compare between multiple possible fits (hypotheses) during phase
  identification performed on a powder XRD pattern.
* **Data visualization**: Plot powder XRD data and the results of phase identification
  and refinement, including missing/extra peaks.
* **Constrained search**: Constrain phase search space using suggested pre-filtering of
  crystallographic databases, as well as reaction energetics and predicted products.

::::{admonition} Installation (recommended)
:class: note
The easiest way to install the latest release of Dara is via pip from PyPI:

```
pip install dara
```

+++
[More installation options »](install.md)
::::

---

## Wanna have a quick try?
::::{admonition} 🚀 Try the Dara Web Server!
:class: tip
For the fastest way to experience Dara, launch the **web server** and use the browser-based interface:

```bash
dara server
```

Then open your browser and navigate to `http://localhost:8898`. You will have a full application with all the features of Dara, including data management, phase analysis, and refinement.
::::


## Tutorials

```{include} tutorial_grid.md
```

---
:::{seealso}
This project gets a lot of inspiration from the [Profex](https://www.profex-xrd.org/)
project led by Nicola Dobelin. We are very grateful for their work.
:::

```{toctree}
:maxdepth: 2
:hidden:
Installation<install>
Web Server<web_server>
Tutorials<tutorials>
API Docs<modules>
```
