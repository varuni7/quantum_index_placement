# Elasticsearch harness

`QuboBenchDriverTests.java` runs Elasticsearch's own shard allocator
(`DesiredBalanceComputer` with `BalancedShardsAllocator`, production deciders, shipped
balance factors) on an instance we supply, and writes the placement it chooses. It is a
test class only so that it can use the setup Elasticsearch's own allocation tests use; it
does not change Elasticsearch.

Elasticsearch's shipped settings are used throughout, including its balance factors.
Two details make the allocator behave as it does in a real cluster:

* **Write-load forecasts.** The open-source build's default forecaster returns no forecast
  (forecasting is a licensed feature), so the allocator would never see write load. The
  harness supplies a forecaster that returns the forecast stored on each index.
* **Running to convergence.** While new shards are being assigned, Elasticsearch stops a
  balance computation after `max_balance_computation_time_during_index_creation` (1 s by
  default), publishes the intermediate result, and the next computation continues from
  it. The harness does the same, calling `compute()` from the previous result until it
  reports `CONVERGED`. Each output file records the finish reason and the number of calls.

## Setup

1. Check out Elasticsearch 9.5.3:
   ```bash
   git clone --branch v9.5.3 --depth 1 https://github.com/elastic/elasticsearch
   ```
2. Copy the harness into the server module's tests:
   ```bash
   cp QuboBenchDriverTests.java \
      elasticsearch/server/src/test/java/org/elasticsearch/cluster/routing/allocation/allocator/
   ```
3. Point this repository at the checkout and run:
   ```bash
   export ES_REPO=/path/to/elasticsearch
   python run.py configs/smaller/es_allocator.json
   ```
   `ES_GRADLE` overrides the Gradle wrapper if you use your own (default
   `$ES_REPO/gradlew`). Elasticsearch's build needs the JDK version its documentation
   names.

The first call compiles the server tests and takes several minutes; later calls take
about 30 s each, most of it Gradle and JVM startup.
