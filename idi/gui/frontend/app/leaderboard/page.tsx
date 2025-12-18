"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Navbar } from "@/components/Navbar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Trophy } from "lucide-react";

export default function LeaderboardPage() {
    const [entries, setEntries] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        api.leaderboard.get().then(setEntries).finally(() => setLoading(false));
    }, []);

    return (
        <main className="min-h-screen bg-slate-950 text-slate-100 pb-20">
            <Navbar />
            <div className="container mx-auto p-8">
                <h1 className="text-3xl font-bold mb-6 flex items-center gap-2">
                    <Trophy className="text-yellow-500" /> Leaderboard
                </h1>
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
                                    {entries.map((entry, i) => (
                                        <tr key={i} className="border-b border-border/50 hover:bg-secondary/10">
                                            <td className="px-6 py-4 font-mono">#{i + 1}</td>
                                            <td className="px-6 py-4 font-mono text-xs">{entry.pack_hash.substring(0, 12)}...</td>
                                            <td className="px-6 py-4 font-bold text-cyan-400">{entry.score.toFixed(4)}</td>
                                            <td className="px-6 py-4">{entry.metrics.episodes_run}</td>
                                            <td className="px-6 py-4 text-muted-foreground">{new Date(entry.timestamp_ms).toLocaleString()}</td>
                                        </tr>
                                    ))}
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
        </main>
    );
}
