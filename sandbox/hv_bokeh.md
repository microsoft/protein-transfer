Notes and links for adding dot size and shade to Holoviews Bokeh backend plot legends

* Holoviews bokeh legend GitHub issues [here](https://github.com/bokeh/bokeh/issues/2603) and [here](GitHub issue) as well as a Holoviz [discourse](https://discourse.holoviz.org/t/point-size-legend/4330) with a `matplotlib` getaway
https://github.com/holoviz/holoviews/issues/5305
* Just `matplotlib` [scatter_with_legend](https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_with_legend.html)

So Bokeh likely will NOT work and should have used good old day matplotlib... But some resources for the future:

### Holoviews
* Holoviews `renderer-module` [reference_manual](https://holoviews.org/reference_manual/holoviews.plotting.bokeh.html#renderer-module)
* Holoviews `Plotting_with_Bokeh` [user_guide](https://holoviews.org/user_guide/Plotting_with_Bokeh.html#working-with-bokeh-directly)
* Holoviews `NdOverlay` [reference](https://holoviews.org/reference/containers/bokeh/NdOverlay.html) for legend creation and `stackoverflow` [attampt](https://stackoverflow.com/questions/64744222/holoviews-ndoverlay-legend)
* Holoviews modify legend manually [discourse](https://discourse.holoviz.org/t/manually-modify-legend/2128/2)
* Holoviews `Applying_Customizations` [user_guide](https://holoviews.org/user_guide/Applying_Customizations.html#split-into-style-plot-and-norm-options)

### Bokeh
* [customizing-glyphs](https://docs.bokeh.org/en/latest/docs/first_steps/first_steps_2.html#customizing-glyphs)
* [adding-and-styling-a-legend](https://docs.bokeh.org/en/latest/docs/first_steps/first_steps_3.html#adding-and-styling-a-legend)
* `legends` [examples](https://docs.bokeh.org/en/latest/docs/examples/models/legends.html)
* `styling-legends` [user_guide](https://docs.bokeh.org/en/latest/docs/user_guide/styling/plots.html#styling-legends)
* `annotations` of the `legends` [user_guide](https://docs.bokeh.org/en/latest/docs/user_guide/basic/annotations.html#legends) with the multiline example and [manual-legends](https://docs.bokeh.org/en/latest/docs/user_guide/basic/annotations.html#manual-legends)
* Circles with different sizes example [here](https://docs.bokeh.org/en/latest/docs/reference/models/glyphs/circle.html#circle)
* [bokeh.models.GlyphRenderer](https://docs.bokeh.org/en/latest/docs/reference/models/renderers.html#bokeh.models.GlyphRenderer)
* [bokeh.plotting.figure](https://docs.bokeh.org/en/latest/docs/reference/plotting/figure.html)
* [get-glyphrenderer-for-annotation](https://stackoverflow.com/questions/55187376/bokeh-how-to-get-glyphrenderer-for-annotation)
* [interactive-legend-hiding-glyphs](https://stackoverflow.com/questions/62020096/python-bokeh-interactive-legend-hiding-glyphs-not-working)