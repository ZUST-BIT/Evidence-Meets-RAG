# Paper figures

Figures 1 and 2 use the authors' original editable PowerPoint diagrams. Their layouts, native objects, fonts, images, and connectors remain in the source files. Statistical plots are generated separately from experiment outputs.

| Paper figure | Editable source | Publication PDF |
| --- | --- | --- |
| Figure 1: motivation and comparison | `figures/source/intro-motivation.pptx` | `figures/intro-motivation.pdf` |
| Figure 2: framework overview | `figures/source/overview_framework.pptx` | `figures/overview_framework.pdf` |

## Figure 1 values

The committed editable source contains the current 571-question values. It does not contain the older numeric labels from the original draft.

| Metric | Document corpus | Semantic KG | Evidence graph |
| --- | ---: | ---: | ---: |
| Component recall | 0.23 | 0.22 | 0.60 |
| Answering accuracy | 0.34 | 0.42 | 0.44 |

The two annotations, `+167%` and `+30%`, compare the evidence graph with the document corpus. They use full-precision results before rounding, so they should not be recomputed from the two-decimal labels above.

`figures/figure1_values.json` stores the full-precision values and references their analysis inputs. Only its `metrics` field contains current results. The separate `template_scale` constants calibrate the original bar geometry and are not experimental results.

To update the existing editable objects from this configuration:

```bash
python scripts/update_concept_figures.py
```

This writes `figures/generated/intro-motivation.pptx`. It changes six numeric labels, six native bar widths, and two gain labels, and moves numeric labels with their bar endpoints. It retains all other diagram objects. The source file remains untouched. The helper removes the old cached thumbnail rather than distributing an outdated preview.

To use freshly recomputed 571-question analysis outputs:

```bash
python scripts/update_concept_figures.py \
  --metrics-dir dataset/rag/analysis/unified_571
```

The directory must contain `retrieval_571.json` and `main_and_ablation.json`. The helper checks that retrieval and answering results cover all 571 questions.

## PDF export

PPTX updating uses Python's standard library. Optional PDF export additionally requires LibreOffice (`soffice` or `libreoffice`) and `pypdf`:

```bash
python scripts/update_concept_figures.py --export-pdf
```

The resulting `figures/generated/intro-motivation.pdf` retains the original publication crop box. Use `--soffice /path/to/soffice` when the executable is outside `PATH`. PowerPoint can also export the editable source to PDF; preserve the publication crop when replacing the committed PDF. Review fonts and text placement after exporting with a different office application.

Figure 2's committed PDF is the unchanged original export, paired with its original editable source. It has no result values to recompute. Re-exporting its PPTX is optional; keep the supplied PDF when an identical rendering is required.

## Relationship to analysis plots

`scripts/update_concept_figures.py` updates Figure 1's diagram. It does not run experiments or generate analysis plots. After completing the experiments and their analyses, use:

```bash
python scripts/build_unified_571_figures.py
```

This generates the existing Figure 3 radar, Figure 4 retrieval/use analysis, and Figure 5 component ablation designs under `outputs/figures/`. Figure 6's conditional-effect matrices come from the stage analysis/plotting entry point in `scripts/expand_stage_571.py`. Other analysis-specific plotting commands are documented alongside the experiment workflow. Concept diagrams and analysis plots therefore share experimental values where applicable while retaining their separate editable or programmatic sources.
