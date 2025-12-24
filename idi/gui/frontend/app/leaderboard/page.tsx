"use client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Trophy, RefreshCw, Copy, Crown, Medal, Radio } from "lucide-react";
import { useToast } from "@/components/ui/toast";
import { cn } from "@/lib/utils";
import { useLeaderboard, LeaderboardEntry } from "@/hooks/useLeaderboard";

export default function LeaderboardPage() {
    const toast = useToast();

    const {
        entries,
        loading,
        lastUpdated,
        refresh,
        isLive,
        connectionStatus,
    } = useLeaderboard({
        onContributionAccepted: (data) => {
            if (data.is_new_leader) {
                toast.success(`🏆 New #1: Agent ${data.pack_hash.substring(0, 8)} with score ${data.score.toFixed(4)}!`);
            } else if (data.leaderboard_position) {
                toast.success(`Agent accepted! Ranked #${data.leaderboard_position} (score: ${data.score.toFixed(4)})`);
            } else {
                toast.info(`Agent accepted (score: ${data.score.toFixed(4)})`);
            }
        },
    });

    const handleRefresh = () => {
        refresh();
        toast.info("Refreshing leaderboard...");
    };

    const copyHash = (hash: string) => {
        navigator.clipboard.writeText(hash);
        toast.success("Hash copied to clipboard");
    };

    const getRankIcon = (rank: number) => {
        if (rank === 1) return <Crown className="w-4 h-4 text-yellow-400" />;
        if (rank === 2) return <Medal className="w-4 h-4 text-gray-300" />;
        if (rank === 3) return <Medal className="w-4 h-4 text-amber-600" />;
        return null;
    };

    const getRankStyle = (rank: number) => {
        if (rank === 1) return "bg-yellow-500/10 border-yellow-500/30";
        if (rank === 2) return "bg-gray-400/10 border-gray-400/30";
        if (rank === 3) return "bg-amber-600/10 border-amber-600/30";
        return "";
    };

    return (
        <div className="space-y-6">
            {/* Header with Refresh */}
            <div className="flex items-center justify-between">
                <h1 className="text-3xl font-bold flex items-center gap-2">
                    <Trophy className="text-yellow-500" /> Leaderboard
                </h1>
                <div className="flex items-center gap-4">
                    {/* Live indicator */}
                    <div className={cn(
                        "flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium",
                        isLive
                            ? "bg-green-500/20 text-green-400 border border-green-500/30"
                            : "bg-gray-500/20 text-gray-400 border border-gray-500/30"
                    )}>
                        <Radio className={cn("w-3 h-3", isLive && "animate-pulse")} />
                        {isLive ? "Live" : connectionStatus}
                    </div>

                    {lastUpdated && (
                        <span className="text-xs text-muted-foreground">
                            Updated: {lastUpdated.toLocaleTimeString()}
                        </span>
                    )}
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={handleRefresh}
                        disabled={loading}
                    >
                        <RefreshCw className={cn("w-4 h-4 mr-2", loading && "animate-spin")} />
                        Refresh
                    </Button>
                </div>
            </div>

            <Card className="glass">
                <CardHeader>
                    <CardTitle>Top Agents</CardTitle>
                </CardHeader>
                <CardContent>
                    {/* Simple Table */}
                    <div className="w-full overflow-x-auto">
                        <table className="w-full text-sm text-left">
                            <thead className="text-xs uppercase bg-secondary/30 text-muted-foreground">
                                <tr>
                                    <th className="px-6 py-3">Rank</th>
                                    <th className="px-6 py-3">Agent Hash</th>
                                    <th className="px-6 py-3">Score</th>
                                    <th className="px-6 py-3">Episodes</th>
                                    <th className="px-6 py-3">Timestamp</th>
                                </tr>
                            </thead>
                            <tbody>
                                {entries.map((entry, i) => {
                                    const rank = i + 1;
                                    return (
                                        <tr
                                            key={entry.pack_hash}
                                            className={cn(
                                                "border-b border-border/50 hover:bg-secondary/10 transition-colors",
                                                getRankStyle(rank)
                                            )}
                                        >
                                            <td className="px-6 py-4 font-mono flex items-center gap-2">
                                                {getRankIcon(rank)}
                                                #{rank}
                                            </td>
                                            <td className="px-6 py-4 font-mono text-xs group">
                                                <span className="cursor-help" title={entry.pack_hash}>
                                                    {entry.pack_hash.substring(0, 12)}...
                                                </span>
                                                <button
                                                    onClick={() => copyHash(entry.pack_hash)}
                                                    className="ml-2 opacity-0 group-hover:opacity-100 transition-opacity"
                                                    title="Copy full hash"
                                                >
                                                    <Copy className="w-3 h-3 hover:text-cyan-400" />
                                                </button>
                                            </td>
                                            <td className={cn(
                                                "px-6 py-4 font-bold",
                                                rank <= 3 ? "text-cyan-400" : "text-foreground"
                                            )}>
                                                {entry.score.toFixed(4)}
                                            </td>
                                            <td className="px-6 py-4">{entry.metrics.episodes_run}</td>
                                            <td className="px-6 py-4 text-muted-foreground">
                                                {new Date(entry.timestamp_ms).toLocaleString()}
                                            </td>
                                        </tr>
                                    );
                                })}
                                {entries.length === 0 && (
                                    <tr>
                                        <td colSpan={5} className="px-6 py-8 text-center text-muted-foreground">
                                            No entries yet. Train an agent to see it here!
                                        </td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
