/*
 * Runs Elasticsearch's DesiredBalanceComputer on an instance we supply and writes the
 * placement it chooses. A test class only so that it can use the setup Elasticsearch's
 * allocation tests use; it does not change Elasticsearch.
 *
 * Input, one line per index:  indexName <TAB> shardSizeBytes <TAB> writeLoad
 * Output, one line per copy:  indexName <TAB> shardNum <TAB> nodeId
 * Batch mode: -Dqubo.batch=manifest.tsv, one run per line:
 *     inFile <TAB> outFile <TAB> nodes <TAB> replicas [ <TAB> diskPerNode ]
 */
package org.elasticsearch.cluster.routing.allocation.allocator;

import org.elasticsearch.cluster.ClusterInfo;
import org.elasticsearch.cluster.routing.allocation.decider.AllocationDeciders;
import org.elasticsearch.cluster.ClusterModule;
import org.elasticsearch.cluster.ESAllocationTestCase;
import org.elasticsearch.cluster.ClusterInfo.NodeAndShard;
import org.elasticsearch.cluster.ClusterName;
import org.elasticsearch.cluster.ClusterState;
import org.elasticsearch.cluster.DiskUsage;
import org.elasticsearch.cluster.TestShardRoutingRoleStrategies;
import org.elasticsearch.cluster.metadata.IndexMetadata;
import org.elasticsearch.cluster.metadata.Metadata;
import org.elasticsearch.cluster.node.DiscoveryNodes;
import org.elasticsearch.common.transport.TransportAddress;
import org.elasticsearch.cluster.node.DiscoveryNodeUtils;
import org.elasticsearch.cluster.routing.AllocationId;
import org.elasticsearch.cluster.routing.IndexRoutingTable;
import org.elasticsearch.cluster.routing.RoutingTable;
import org.elasticsearch.cluster.routing.allocation.TestRoutingAllocationFactory;
import org.elasticsearch.cluster.routing.allocation.command.MoveAllocationCommand;
import org.elasticsearch.common.UUIDs;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.time.TimeProviderUtils;
import org.elasticsearch.index.IndexVersion;
import org.elasticsearch.index.shard.ShardId;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import org.elasticsearch.cluster.routing.allocation.WriteLoadForecaster;
import org.elasticsearch.cluster.metadata.ProjectMetadata;
import java.util.OptionalDouble;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;

import static org.elasticsearch.cluster.ClusterInfo.shardIdentifierFromRouting;
import static org.elasticsearch.cluster.routing.ShardRoutingState.STARTED;
import static org.elasticsearch.cluster.routing.TestShardRouting.shardRoutingBuilder;
import static org.elasticsearch.common.settings.ClusterSettings.createBuiltInClusterSettings;

public class QuboBenchDriverTests extends ESAllocationTestCase {

    private record Idx(String name, long size, double writeLoad, List<String> current) {}

    public void testRunInstance() throws Exception {
        final String batch = System.getProperty("qubo.batch");
        if (batch != null) {
            long t0 = System.nanoTime();
            int n = 0;
            for (String line : Files.readAllLines(Path.of(batch))) {
                if (line.isBlank() || line.startsWith("#")) continue;
                String[] f = line.split("\t");
                long disk = f.length > 4 ? Long.parseLong(f[4]) : 0L;
                runOne(f[0], f[1], Integer.parseInt(f[2]), Integer.parseInt(f[3]), disk);
                n++;
            }
            logger.info("QuboBenchDriverTests: batch of {} instances in {} ms",
                n, (System.nanoTime() - t0) / 1_000_000);
            return;
        }
        final String in = System.getProperty("qubo.in");
        final String out = System.getProperty("qubo.out");
        if (in == null || out == null) {
            return;
        }
        runOne(in, out,
            Integer.parseInt(System.getProperty("qubo.nodes", "6")),
            Integer.parseInt(System.getProperty("qubo.replicas", "1")),
            Long.parseLong(System.getProperty("qubo.diskPerNode", "0")));
    }

    private void runOne(String in, String out, int nodes, int replicas, long diskPerNode) throws Exception {
        Files.writeString(Path.of(out), solveToString(in, nodes, replicas, diskPerNode));
    }

    private String solveToString(String in, int nodes, int replicas, long diskPerNode) throws Exception {
        final int copies = replicas + 1;

        final List<Idx> idxs = new ArrayList<>();
        for (String line : Files.readAllLines(Path.of(in))) {
            if (line.isBlank() || line.startsWith("#")) continue;
            String[] f = line.split("\t", -1);
            List<String> inc = (f.length > 3 && f[3].isBlank() == false) ? List.of(f[3].split(",")) : List.of();
            idxs.add(new Idx(f[0], Long.parseLong(f[1]), Double.parseDouble(f[2]), inc));
        }

        var discoveryNodes = DiscoveryNodes.builder();
        final List<String> nodeIds = new ArrayList<>();
        final Map<String, Long> used = new HashMap<>();
        for (int i = 0; i < nodes; i++) {
            String id = "node-" + i;
            nodeIds.add(id);
            used.put(id, 0L);
            discoveryNodes.add(
                DiscoveryNodeUtils.builder(id)
                    .address(new TransportAddress(TransportAddress.META_ADDRESS, 9300 + i))
                    // fixed ids, so runs on identical input are processed in the same order
                    .ephemeralId("eph-" + id)
                    .roles(MASTER_DATA_ROLES)
                    .build()
            );
        }

        var metadataBuilder = Metadata.builder();
        var routingTableBuilder = RoutingTable.builder(TestShardRoutingRoleStrategies.DEFAULT_ROLE_ONLY);
        var shardSizes = new HashMap<String, Long>();
        var dataPath = new HashMap<NodeAndShard, String>();
        long totalBytes = 0;

        for (Idx idx : idxs) {
            var inSyncIds = new ArrayList<String>();
            for (int c = 0; c < copies; c++) {
                inSyncIds.add(idx.name() + "-alloc-" + c);
            }
            var imBuilder = IndexMetadata.builder(idx.name())
                .settings(indexSettings(IndexVersion.current(), "uuid-" + idx.name(), 1, replicas))
                .shardSizeInBytesForecast(idx.size())
                .putInSyncAllocationIds(0, new LinkedHashSet<>(inSyncIds));
            if (idx.writeLoad() > 0) {
                imBuilder.indexWriteLoadForecast(idx.writeLoad());
            }
            metadataBuilder.put(imBuilder);
            var indexMetadata = metadataBuilder.get(idx.name());
            var shardId = new ShardId(indexMetadata.getIndex(), 0);

            shardSizes.put(shardIdentifierFromRouting(shardId, true), idx.size());
            shardSizes.put(shardIdentifierFromRouting(shardId, false), idx.size());
            totalBytes += idx.size() * (long) copies;

            if (idx.current().isEmpty()) {
                routingTableBuilder.addAsNew(indexMetadata);       // fresh placement
            } else {
                var irt = IndexRoutingTable.builder(indexMetadata.getIndex());
                for (int c = 0; c < copies; c++) {
                    String nodeId = idx.current().get(c % idx.current().size());
                    dataPath.put(new NodeAndShard(nodeId, shardId), "/data");
                    used.merge(nodeId, idx.size(), Long::sum);
                    irt.addShard(
                        shardRoutingBuilder(shardId, nodeId, c == 0, STARTED)
                            .withAllocationId(AllocationId.newInitializing(inSyncIds.get(c)))
                            .build()
                    );
                }
                routingTableBuilder.add(irt);
            }
        }

        var clusterState = ClusterState.builder(ClusterName.DEFAULT)
            .nodes(discoveryNodes.masterNodeId("node-0").localNodeId("node-0"))
            .metadata(metadataBuilder)
            .routingTable(routingTableBuilder)
            .build();

        final long disk = diskPerNode > 0
            ? diskPerNode
            : Math.max(totalBytes / nodes, used.values().stream().max(Long::compare).orElse(0L)) * 2;
        var diskUsage = new HashMap<String, DiskUsage>();
        for (String id : nodeIds) {
            diskUsage.put(id, new DiskUsage(id, id, "/data", disk, disk - used.getOrDefault(id, 0L)));
        }
        var clusterInfo = ClusterInfo.builder()
            .leastAvailableSpaceUsage(diskUsage)
            .mostAvailableSpaceUsage(diskUsage)
            .shardSizes(shardSizes)
            .dataPath(dataPath)
            .build();

        var settings = Settings.EMPTY;     // Elasticsearch's shipped defaults
        var allocation = TestRoutingAllocationFactory.forClusterState(clusterState)
            // production deciders, in production order
            .allocationDeciders(
                new AllocationDeciders(
                    ClusterModule.createAllocationDeciders(settings, createBuiltInClusterSettings(settings), List.of())
                )
            )
            .clusterInfo(clusterInfo)
            .build();

        var computer = new DesiredBalanceComputer(
            createBuiltInClusterSettings(settings),
            TimeProviderUtils.create(System::currentTimeMillis),
            // returns the write-load forecast stored on each index (the default forecaster
            // in the open-source build returns none)
            new BalancedShardsAllocator(new BalancerSettings(settings), new WriteLoadForecaster() {
                @Override
                public ProjectMetadata.Builder withWriteLoadForecastForWriteIndex(String ds, ProjectMetadata.Builder md) {
                    return md;
                }

                @Override
                public OptionalDouble getForecastedWriteLoad(IndexMetadata indexMetadata) {
                    return indexMetadata.getForecastedWriteLoad();
                }

                @Override
                public void refreshLicense() {}
            }),
            TEST_ONLY_EXPLAINER
        );
        Queue<List<MoveAllocationCommand>> pending = new LinkedList<>(List.of(List.of()));
        // As in production: after an early stop (max_balance_computation_time_during_index_creation,
        // 1 s by default) the next computation continues from the previous result, until CONVERGED.
        final long solveT0 = System.nanoTime();
        DesiredBalance desiredBalance = DesiredBalance.BECOME_MASTER_INITIAL;
        int calls = 0;
        do {
            desiredBalance = computer.compute(
                desiredBalance,
                new DesiredBalanceInput(calls, allocation, List.of()),
                pending,
                ignored -> true
            );
            calls++;
        } while (desiredBalance.finishReason() != DesiredBalance.ComputationFinishReason.CONVERGED);
        final long solveNs = System.nanoTime() - solveT0;

        var sb = new StringBuilder();
        sb.append("#solve_ns\t").append(solveNs).append('\n');
        sb.append("#finish\t").append(desiredBalance.finishReason()).append('\t').append(calls).append('\n');
        for (Map.Entry<ShardId, ShardAssignment> e : desiredBalance.assignments().entrySet()) {
            for (String nodeId : e.getValue().nodeIds()) {
                sb.append(e.getKey().getIndexName()).append('\t').append(e.getKey().id()).append('\t').append(nodeId).append('\n');
            }
        }
        return sb.toString();
    }
}
