# dataframe-benchmarking
Benchmarking a few Dataframe python frameworks;

This repo was mostly inspired by the claim that [FireDucks](https://fireducks-dev.github.io/) would be a faster drop-in replacement of [Pandas](https://pandas.pydata.org/), I've made some not too complex test cases based on that claim. For more complex or long-running tasks, I usually resort to [Polars](https://pola.rs/), which requires meaningfully different syntax and usage, but is known for being generally faster.

The goal was to make some tests that would allow a comparison between those frameworks that is realistic while employing a not-so-heavy use case in terms of numbers of manipulations (something that could favor lazy-evaluated frameworks) but using a realistic data amount.  think a developer should always build something like this in house before proposing a tool migration. You are looking for the best tool for your problem, not the best tool for a problem you've never faced yourself.

# Usage 

With [uv](https://docs.astral.sh/uv/) installed, run the command `bash run.sh`.

There is a github action that executes it and provides artifacts for analysis.

# Other notes:
I suspected `__pycache__` or some Just-in-Time (JIT) compilation artifact could have an impact on performance. Removing the `__pycache__` folder could help with the former, but other than repeating operations, I don't see a proper way of testing the ladder.