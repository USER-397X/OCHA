# Claude Code Starting Prompt — Ukraine Media Attention Demo

Paste everything below this line into Claude Code after launching it from your repo root.

---

## Context

It's Saturday 11 PM at ETH Datathon 2026. Noon Sunday is the hard deadline. I am validating whether a "media attention vs humanitarian funding" research direction is worth pursuing for the UN OCHA challenge. I need a plot in the next 30–45 minutes that shows whether GDELT media-attention data correctly captures the Ukraine invasion as a massive spike on 24 February 2022. That single plot decides whether I invest the next 6 hours in this direction or pivot.

## Non-goals tonight

To keep you focused, these are explicitly OUT of scope for this session:

- No package structure, no module layout, no `__init__.py` files
- No integration with the rest of the repo
- No schema design, no output-format work for teammates
- No tests
- No virtualenv setup — use system Python with `pip install --user`
- No BigQuery, no GCP, no authentication of any kind
- No multi-country generalization yet
- No type hints, no docstrings beyond one-liners
- No refactoring of anything you produce

Every one of these items is a trap that sounds responsible but will cost me the demo window. We do all of it tomorrow if and only if the demo works.

## Approach

Use the free, auth-free GDELT DOC 2.0 API via the `gdeltdoc` Python library:

- PyPI: `pip install gdeltdoc`
- GitHub (docs + examples): https://github.com/alex9smith/gdelt-doc-api
- Relevant mode: `timelinevolraw` returns actual daily article counts matching the filter, not percentages

## The single file to build

Create exactly one file: `scratch/ukraine_demo.py`.

It should do this, in this order:

1. Install `gdeltdoc`, `pandas`, and `matplotlib` via pip if they aren't already available. Print what it installs so I can see.
2. Use `gdeltdoc.Filters` with `keyword="Ukraine"`, `language="English"`, `start_date="2021-06-01"`, `end_date="2023-06-01"`. This gives us 8 months of pre-invasion baseline and 16 months of post-invasion signal, which is more than enough to see the spike.
3. Call `gd.timeline_search("timelinevolraw", f)` to get a pandas DataFrame of daily raw article counts.
4. Plot raw article count on the y-axis against date on the x-axis using matplotlib. Make the line thin and dark so the spike reads clearly.
5. Draw a vertical red dashed line at 2022-02-24 with a text label "Invasion (24 Feb 2022)" anchored above the line.
6. Save the figure as `scratch/ukraine_demo.png` at 150 DPI.
7. Print three numbers to stdout, each on its own labeled line: the date of peak article volume, the peak value, and the ratio of peak value to the mean daily volume of the pre-invasion baseline (2021-06-01 through 2022-02-23).

## Success criteria

The plot must show a dramatic, unmissable spike starting around 24 February 2022, sustained well above the pre-invasion baseline for months. The peak-to-baseline ratio should be at least 10x. If it isn't, something is wrong with the query — not with the underlying premise — and we debug the query before concluding the approach fails.

## If something goes wrong

- If `gdeltdoc` errors on the full 2-year range, retry once with a 1-year range (2021-09-01 to 2022-09-01) before debugging further.
- If the API rate-limits you, wait 60 seconds and retry once.
- If either of those fails again, stop and tell me exactly what error you saw. Don't invent fallback paths.

## One clarifying question

Before you write code, you may ask me one clarifying question if anything above is genuinely ambiguous. Otherwise, start building and show me the plot when it's done.